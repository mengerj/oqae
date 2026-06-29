"""
Pure evaluation metrics for the OQAE benchmarking harness.

These functions quantify the three things PR #9 cares about when comparing
likelihood / codebook configurations:

- **Reconstruction quality** — how well the model reproduces held-out
  expression (:func:`reconstruction_metrics`).
- **Codebook utilization** — whether the discrete bottleneck stays diverse
  (high perplexity / utilization) rather than collapsing onto a few entries
  (:func:`codebook_usage`).
- **Downstream separability** — whether the learned latent keeps biologically
  distinct cells apart, measured against known labels
  (:func:`separability_score`).

Everything here is dependency-light (NumPy + Torch only) and offline. The
codebook and separability metrics are *pure* functions of arrays and are unit
tested directly; :func:`reconstruction_metrics` runs a trained model in
``eval`` + ``no_grad`` (so it never mutates the EMA codebooks).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from omvqvae.data.normalize import CountMatrix, compute_size_factors, to_dense
from omvqvae.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omvqvae.models.vqvae import OmicsVQVAE

logger = get_logger(__name__)

#: Array-like accepted as discrete codes ``(n_cells, n_codebooks)``.
CodesLike = Union[Tensor, np.ndarray]
#: Array-like accepted as latent vectors ``(n_cells, n_latent)``.
LatentLike = Union[Tensor, np.ndarray]
#: Array-like accepted as per-cell size factors ``(n_cells,)``.
SizeFactorsLike = Union[Tensor, np.ndarray]

__all__ = [
    "CodebookUsage",
    "ReconstructionMetrics",
    "codebook_usage",
    "reconstruction_metrics",
    "separability_score",
]


@dataclass
class CodebookUsage:
    """
    Codebook diversity / collapse metrics over a set of encoded cells.

    Attributes
    ----------
    perplexity : float
        Mean per-level perplexity (``exp`` of the index-assignment entropy). A
        value near ``codebook_size`` means the codes are used uniformly; a value
        near ``1`` signals collapse onto a single entry.
    utilization : float
        Mean fraction of codebook entries used at least once, in ``[0, 1]``.
    perplexities : List[float]
        Per-level perplexity, one entry per residual codebook.
    utilizations : List[float]
        Per-level utilization fraction, one entry per residual codebook.
    codebook_size : int
        Number of entries in each codebook (the perplexity ceiling).
    """

    perplexity: float
    utilization: float
    perplexities: List[float]
    utilizations: List[float]
    codebook_size: int


@dataclass
class ReconstructionMetrics:
    """
    Reconstruction-quality metrics over a set of held-out cells.

    Attributes
    ----------
    nll : float
        Mean reconstruction negative log-likelihood per cell (the model's own
        likelihood). Comparable across codebook sweeps for a *fixed* likelihood;
        not directly comparable across different likelihoods (different units).
    mae : float
        Mean absolute error between the model's expected reconstruction and the
        target, in the head's native target space (raw counts for NB/ZINB,
        ``log1p`` expression for the Gaussian head).
    """

    nll: float
    mae: float


def _to_numpy_int(codes: CodesLike) -> np.ndarray:
    """Coerce codes to a 2-D ``int64`` NumPy array."""
    if isinstance(codes, Tensor):
        array = codes.detach().cpu().numpy()
    else:
        array = np.asarray(codes)
    array = np.asarray(array, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError(
            f"codes must be 2-D (n_cells, n_codebooks); got {array.ndim}-D."
        )
    return array


def _to_numpy_float(latent: LatentLike) -> np.ndarray:
    """Coerce latent vectors to a 2-D ``float64`` NumPy array."""
    if isinstance(latent, Tensor):
        array = latent.detach().cpu().numpy()
    else:
        array = np.asarray(latent)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"latent must be 2-D (n_cells, n_latent); got {array.ndim}-D.")
    return array


def codebook_usage(codes: CodesLike, codebook_size: int) -> CodebookUsage:
    """
    Summarize codebook diversity for a batch of discrete codes.

    Parameters
    ----------
    codes : torch.Tensor or numpy.ndarray
        Per-cell codes of shape ``(n_cells, n_codebooks)`` (integer codebook
        indices, e.g. :attr:`omvqvae.inference.EncodedCells.codes`).
    codebook_size : int
        Number of entries in each codebook (the index range ``[0, codebook_size)``
        and the perplexity ceiling).

    Returns
    -------
    CodebookUsage
        Per-level and mean perplexity / utilization.

    Raises
    ------
    ValueError
        If ``codes`` is not 2-D or ``codebook_size`` is not positive.

    Notes
    -----
    Perplexity is computed over the full set of cells (not per minibatch), so it
    reflects the dataset-level code distribution and is a more reliable
    collapse signal than the per-forward metric logged during training.
    """
    if codebook_size < 1:
        raise ValueError("codebook_size must be a positive integer.")
    array = _to_numpy_int(codes)
    perplexities: List[float] = []
    utilizations: List[float] = []
    for level in range(array.shape[1]):
        counts = np.bincount(array[:, level], minlength=codebook_size).astype(
            np.float64
        )
        if counts.shape[0] > codebook_size:
            raise ValueError(
                "codes contain an index >= codebook_size "
                f"({int(array[:, level].max())} >= {codebook_size})."
            )
        total = counts.sum()
        if total <= 0:
            perplexities.append(0.0)
            utilizations.append(0.0)
            continue
        probs = counts / total
        nonzero = probs[probs > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum())
        perplexities.append(float(np.exp(entropy)))
        utilizations.append(float((counts > 0).sum()) / codebook_size)

    mean_perplexity = float(np.mean(perplexities)) if perplexities else 0.0
    mean_utilization = float(np.mean(utilizations)) if utilizations else 0.0
    return CodebookUsage(
        perplexity=mean_perplexity,
        utilization=mean_utilization,
        perplexities=perplexities,
        utilizations=utilizations,
        codebook_size=codebook_size,
    )


def separability_score(latent: LatentLike, labels: Sequence[object]) -> float:
    """
    Nearest-centroid separability of a latent space given known labels.

    Each label group's centroid is computed in latent space, then every cell is
    assigned to its nearest centroid; the score is the fraction correctly
    assigned. This is a fast, deterministic, dependency-free proxy for
    "downstream separability": a value near ``1`` means the labels form
    well-separated clusters in the latent, while a value near chance
    (``1 / n_labels``) means the latent does not encode the label structure.

    Parameters
    ----------
    latent : torch.Tensor or numpy.ndarray
        Latent vectors of shape ``(n_cells, n_latent)`` (e.g. the continuous
        pre-quantization latent from :func:`omvqvae.inference.encode`).
    labels : Sequence
        Per-cell labels of length ``n_cells`` (e.g. cell type or program id).

    Returns
    -------
    float
        Nearest-centroid accuracy in ``[0, 1]``, or ``nan`` when there are fewer
        than two distinct labels (separability is undefined).

    Raises
    ------
    ValueError
        If ``latent`` is not 2-D, or ``labels`` length does not match the number
        of cells.

    Notes
    -----
    The centroids are fit on the same cells they score (resubstitution), so this
    is an optimistic upper bound; it is intended for *relative* comparison across
    benchmark configurations rather than as an absolute classifier accuracy.
    """
    array = _to_numpy_float(latent)
    labels_arr = np.asarray(list(labels), dtype=object)
    if labels_arr.shape[0] != array.shape[0]:
        raise ValueError(
            f"labels has length {labels_arr.shape[0]} but there are "
            f"{array.shape[0]} cells."
        )
    if array.shape[0] == 0:
        return float("nan")
    unique = np.unique(labels_arr)
    if unique.size < 2:
        return float("nan")
    centroids = np.stack([array[labels_arr == value].mean(axis=0) for value in unique])
    # Squared Euclidean distance from each cell to each centroid: (n_cells, n_labels).
    distances = ((array[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    predicted = unique[distances.argmin(axis=1)]
    return float((predicted == labels_arr).mean())


@contextmanager
def _eval_mode(model: "OmicsVQVAE") -> Iterator[None]:
    """Run ``model`` in ``eval`` + ``no_grad``, restoring its prior mode."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        if was_training:
            model.train()


def reconstruction_metrics(
    model: "OmicsVQVAE",
    counts: CountMatrix,
    *,
    size_factors: Optional[SizeFactorsLike] = None,
    size_factor_mode: str = "total",
    batch_size: int = 512,
) -> ReconstructionMetrics:
    """
    Evaluate reconstruction quality of a trained model on held-out counts.

    The model is run in ``eval`` + ``no_grad`` (the EMA codebooks are not
    mutated), batched over the cells, and the per-cell reconstruction NLL and the
    expected-reconstruction MAE are accumulated.

    Parameters
    ----------
    model : OmicsVQVAE
        A trained model whose ``n_genes`` matches the columns of ``counts``.
    counts : numpy.ndarray or scipy.sparse.spmatrix
        Raw counts of shape ``(n_cells, n_genes)``.
    size_factors : torch.Tensor or numpy.ndarray, optional
        Per-cell size factors of shape ``(n_cells,)``; computed from the counts
        (via ``size_factor_mode``) when omitted.
    size_factor_mode : str, default "total"
        Mode passed to :func:`omvqvae.data.normalize.compute_size_factors`.
    batch_size : int, default 512
        Number of cells per forward pass.

    Returns
    -------
    ReconstructionMetrics
        Mean per-cell NLL and expected-reconstruction MAE.

    Raises
    ------
    ValueError
        If ``counts`` is not 2-D, its gene dimension disagrees with
        ``model.n_genes``, or ``size_factors`` has the wrong length.
    """
    dense = to_dense(counts)
    if dense.ndim != 2:
        raise ValueError(f"counts must be 2-D (n_cells, n_genes); got {dense.ndim}-D.")
    if dense.shape[1] != model.n_genes:
        raise ValueError(
            f"counts has {dense.shape[1]} genes but the model expects "
            f"{model.n_genes}."
        )
    n_cells = dense.shape[0]
    if n_cells == 0:
        return ReconstructionMetrics(nll=float("nan"), mae=float("nan"))

    if size_factors is None:
        factors_np = compute_size_factors(dense, mode=size_factor_mode)
    else:
        if isinstance(size_factors, Tensor):
            factors_np = size_factors.detach().cpu().to(torch.float32).numpy()
        else:
            factors_np = np.asarray(size_factors, dtype=np.float32)
        factors_np = factors_np.ravel()
        if factors_np.shape[0] != n_cells:
            raise ValueError(
                f"size_factors has length {factors_np.shape[0]} but there are "
                f"{n_cells} cells."
            )

    counts_t = torch.from_numpy(np.ascontiguousarray(dense, dtype=np.float32))
    factors_t = torch.from_numpy(np.ascontiguousarray(factors_np, dtype=np.float32))
    gaussian = model.likelihood == "gaussian"

    nll_total = 0.0
    mae_total = 0.0
    with _eval_mode(model):
        for start in range(0, n_cells, batch_size):
            stop = min(start + batch_size, n_cells)
            batch_counts = counts_t[start:stop]
            batch_factors = factors_t[start:stop]
            output = model(batch_counts, batch_factors)
            expected = model.decode_codes(output.indices, batch_factors)
            target = torch.log1p(batch_counts) if gaussian else batch_counts
            count = stop - start
            nll_total += float(output.reconstruction_loss.item()) * count
            mae_total += float((expected - target).abs().mean().item()) * count

    return ReconstructionMetrics(
        nll=nll_total / n_cells,
        mae=mae_total / n_cells,
    )
