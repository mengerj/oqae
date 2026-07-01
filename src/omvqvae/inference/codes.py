"""
Discrete-code inference API for OQAE.

This is the user-facing latent interface on top of a trained
:class:`~omvqvae.models.vqvae.OmicsVQVAE`: it turns expression into the model's
**discrete universal codes** and turns codes back into expression.

.. code-block:: text

    counts ──encode──► codes (n_cells, n_codebooks)  ──decode──► expected counts
                       + per-cell size factor

The two halves enable the universal-latent use cases:

- **Compression / representation** — each cell becomes a tiny integer code
  ``(n_codebooks,)`` (one codebook index per residual level) plus a scalar size
  factor; that pair is everything needed to reconstruct it.
- **Generation / decoding** — feed any codes (encoded or hand-constructed) back
  through the generative decoder to produce expression profiles.

Code-vector format
------------------
``encode`` returns an :class:`EncodedCells` bundle whose ``codes`` field is an
``int64`` tensor of shape ``(n_cells, n_codebooks)``. Row ``i`` is cell ``i``'s
discrete code; column ``j`` is the index it selected from residual codebook ``j``
(each in ``[0, codebook_size)``). Decoding additionally needs a per-cell
``size_factor`` (observed sequencing depth) so the decoder reconstructs
depth-appropriate counts; ``encode`` returns these alongside the codes.

Everything here runs the model in ``eval`` mode under ``torch.no_grad`` so the
EMA codebooks / dead-code statistics are **not** mutated during inference; the
model's previous train/eval mode is restored on exit.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, Optional, Union

import numpy as np
import torch
from torch import Tensor

from omvqvae.data.anndata_io import extract_counts
from omvqvae.data.dataset import align_to_reference
from omvqvae.data.normalize import CountMatrix, compute_size_factors, to_dense
from omvqvae.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anndata import AnnData

    from omvqvae.hf_utils import LoadedModel
    from omvqvae.models.vqvae import OmicsVQVAE

logger = get_logger(__name__)

#: An array-like accepted as raw counts (dense, sparse, or a torch tensor).
CountsLike = Union[Tensor, np.ndarray, "CountMatrix"]
#: An array-like accepted as per-cell size factors.
SizeFactorsLike = Union[Tensor, np.ndarray]
#: An array-like accepted as discrete codes (or an :class:`EncodedCells` bundle).
CodesLike = Union[Tensor, np.ndarray, "EncodedCells"]

__all__ = [
    "EncodedCells",
    "encode",
    "encode_anndata",
    "decode",
    "decode_to_params",
]


@dataclass
class EncodedCells:
    """
    The discrete encoding of a batch of cells.

    Attributes
    ----------
    codes : torch.Tensor
        Per-cell discrete codes of shape ``(n_cells, n_codebooks)`` (``int64``);
        column ``j`` is the index selected from residual codebook ``j``. This is
        the compact universal representation.
    size_factors : torch.Tensor
        Per-cell size factors of shape ``(n_cells,)`` (``float32``) — the
        observed sequencing depth needed to decode depth-appropriate counts.
    latent : torch.Tensor
        The continuous pre-quantization latent of shape ``(n_cells, n_latent)``
        (``float32``); useful for inspecting quantization error but not required
        for decoding.
    quantized : torch.Tensor
        The continuous *post*-quantization latent of shape
        ``(n_cells, n_latent)`` (``float32``) — the sum of the per-level codebook
        vectors selected by ``codes``. This is the model's discrete
        representation embedded back into latent space; comparing metrics on
        ``latent`` vs ``quantized`` quantifies how much the codebook bottleneck
        costs.
    """

    codes: Tensor
    size_factors: Tensor
    latent: Tensor
    quantized: Tensor

    def __len__(self) -> int:
        return int(self.codes.shape[0])


@contextmanager
def _inference_mode(model: "OmicsVQVAE") -> Iterator[None]:
    """Run ``model`` in ``eval`` + ``no_grad``, restoring its prior mode."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        if was_training:
            model.train()


def _counts_to_array(counts: CountsLike) -> np.ndarray:
    """Coerce dense/sparse/tensor counts to a 2-D dense ``float32`` array."""
    if isinstance(counts, Tensor):
        dense = counts.detach().cpu().to(torch.float32).numpy()
    else:
        dense = to_dense(counts)
    if dense.ndim != 2:
        raise ValueError(f"counts must be 2-D (n_cells, n_genes); got {dense.ndim}-D.")
    return dense


def _resolve_size_factors(
    dense: np.ndarray,
    size_factors: Optional[SizeFactorsLike],
    size_factor_mode: str,
) -> Tensor:
    """Return per-cell size factors as a ``float32`` tensor (computed if absent)."""
    if size_factors is None:
        factors = compute_size_factors(dense, mode=size_factor_mode)
    else:
        if isinstance(size_factors, Tensor):
            factors = size_factors.detach().cpu().to(torch.float32).numpy()
        else:
            factors = np.asarray(size_factors, dtype=np.float32)
        factors = factors.ravel()
        if factors.shape[0] != dense.shape[0]:
            raise ValueError(
                f"size_factors has length {factors.shape[0]} but there are "
                f"{dense.shape[0]} cells."
            )
    return torch.from_numpy(np.ascontiguousarray(factors, dtype=np.float32))


def encode(
    model: "OmicsVQVAE",
    counts: CountsLike,
    *,
    size_factors: Optional[SizeFactorsLike] = None,
    size_factor_mode: str = "total",
    batch_size: int = 1024,
) -> EncodedCells:
    """
    Encode raw counts to their discrete codes.

    Parameters
    ----------
    model : OmicsVQVAE
        A trained model; its ``n_genes`` must match the columns of ``counts``
        (align local data with :func:`encode_anndata` first if it does not).
    counts : torch.Tensor, numpy.ndarray, or scipy.sparse.spmatrix
        Raw counts of shape ``(n_cells, n_genes)``.
    size_factors : torch.Tensor or numpy.ndarray, optional
        Per-cell size factors of shape ``(n_cells,)``. Computed from the counts
        (via ``size_factor_mode``) when omitted.
    size_factor_mode : str, default "total"
        Mode passed to :func:`omvqvae.data.normalize.compute_size_factors` when
        ``size_factors`` is not given.
    batch_size : int, default 1024
        Number of cells encoded per forward pass.

    Returns
    -------
    EncodedCells
        The discrete codes, the size factors used, and the continuous latent.

    Raises
    ------
    ValueError
        If ``counts`` is not 2-D, its gene dimension disagrees with
        ``model.n_genes``, or ``size_factors`` has the wrong length.
    """
    dense = _counts_to_array(counts)
    if dense.shape[1] != model.n_genes:
        raise ValueError(
            f"counts has {dense.shape[1]} genes but the model expects "
            f"{model.n_genes}."
        )
    factors = _resolve_size_factors(dense, size_factors, size_factor_mode)

    n_cells = dense.shape[0]
    counts_t = torch.from_numpy(np.ascontiguousarray(dense, dtype=np.float32))
    codes = torch.empty((n_cells, model.n_codebooks), dtype=torch.long)
    latent = torch.empty((n_cells, model.n_latent), dtype=torch.float32)
    quantized = torch.empty((n_cells, model.n_latent), dtype=torch.float32)

    with _inference_mode(model):
        for start in range(0, n_cells, batch_size):
            stop = min(start + batch_size, n_cells)
            z = model.encode(counts_t[start:stop])
            rvq = model.quantize(z)
            codes[start:stop] = rvq.indices
            latent[start:stop] = z
            quantized[start:stop] = rvq.quantized

    return EncodedCells(
        codes=codes, size_factors=factors, latent=latent, quantized=quantized
    )


def encode_anndata(
    loaded: "LoadedModel",
    adata: "AnnData",
    *,
    layer: Optional[str] = None,
    var_key: Optional[str] = None,
    min_overlap: float = 0.0,
    size_factor_mode: str = "total",
    batch_size: int = 1024,
) -> EncodedCells:
    """
    Encode a local AnnData to discrete codes on the model's feature space.

    The AnnData is aligned to the model's
    :class:`~omvqvae.data.dataset.GeneVocabulary` (missing genes zero-filled,
    extra genes dropped) before encoding, so it works regardless of the file's
    own gene ordering.

    Parameters
    ----------
    loaded : LoadedModel
        A model + its gene vocabulary as returned by
        :func:`omvqvae.hf_utils.load_pretrained` / ``from_pretrained``.
    adata : anndata.AnnData
        Cells to encode (raw counts).
    layer : str, optional
        AnnData layer to read counts from (default ``adata.X``).
    var_key : str, optional
        ``adata.var`` column providing gene ids (default ``var_names``).
    min_overlap : float, default 0.0
        Warn-below threshold for reference-gene coverage during alignment.
    size_factor_mode : str, default "total"
        Size-factor mode (see :func:`omvqvae.data.normalize.compute_size_factors`).
        Size factors are computed from the **aligned** counts.
    batch_size : int, default 1024
        Number of cells encoded per forward pass.

    Returns
    -------
    EncodedCells
        The discrete codes, the size factors used, and the continuous latent.
    """
    matrix, gene_ids = extract_counts(adata, layer=layer, var_key=var_key)
    aligned = align_to_reference(
        matrix, gene_ids, loaded.vocabulary, min_overlap=min_overlap
    )
    return encode(
        loaded.model,
        aligned,
        size_factor_mode=size_factor_mode,
        batch_size=batch_size,
    )


def _resolve_codes(
    codes: CodesLike,
    size_factors: Optional[SizeFactorsLike],
    n_codebooks: int,
) -> tuple[Tensor, Tensor]:
    """Normalize a codes argument (+ size factors) into aligned tensors."""
    resolved_sf: Optional[SizeFactorsLike]
    if isinstance(codes, EncodedCells):
        codes_t = codes.codes
        resolved_sf = codes.size_factors if size_factors is None else size_factors
    else:
        codes_t = torch.as_tensor(codes)
        resolved_sf = size_factors
    codes_t = codes_t.to(torch.long)
    if codes_t.ndim != 2:
        raise ValueError(
            f"codes must be 2-D (n_cells, n_codebooks); got {codes_t.ndim}-D."
        )
    if codes_t.shape[1] != n_codebooks:
        raise ValueError(
            f"codes has {codes_t.shape[1]} levels but the model has "
            f"{n_codebooks} codebooks."
        )
    if resolved_sf is None:
        raise ValueError(
            "size_factors is required when decoding from a raw codes array "
            "(pass an EncodedCells bundle to reuse its size factors)."
        )
    if isinstance(resolved_sf, Tensor):
        sf = resolved_sf.detach().cpu().to(torch.float32)
    else:
        sf = torch.as_tensor(np.asarray(resolved_sf, dtype=np.float32))
    sf = sf.reshape(-1)
    if sf.shape[0] != codes_t.shape[0]:
        raise ValueError(
            f"size_factors has length {sf.shape[0]} but there are "
            f"{codes_t.shape[0]} cells."
        )
    return codes_t, sf


def decode(
    model: "OmicsVQVAE",
    codes: CodesLike,
    size_factors: Optional[SizeFactorsLike] = None,
    *,
    batch_size: int = 1024,
) -> Tensor:
    """
    Decode discrete codes back to expected expression.

    Parameters
    ----------
    model : OmicsVQVAE
        The trained model the codes were produced by (same codebooks).
    codes : torch.Tensor, numpy.ndarray, or EncodedCells
        Per-cell codes of shape ``(n_cells, n_codebooks)`` (``int64``), or an
        :class:`EncodedCells` bundle (whose ``size_factors`` are used unless
        overridden).
    size_factors : torch.Tensor or numpy.ndarray, optional
        Per-cell size factors of shape ``(n_cells,)``. Required for a raw codes
        array; defaults to the bundle's factors when ``codes`` is an
        :class:`EncodedCells`.
    batch_size : int, default 1024
        Number of cells decoded per forward pass.

    Returns
    -------
    torch.Tensor
        The reconstruction mean of shape ``(n_cells, n_genes)`` (expected counts
        for NB/ZINB; ``log1p`` expression for the Gaussian head).

    Raises
    ------
    ValueError
        If ``codes`` is not 2-D, its level count disagrees with the model, or
        size factors are missing / mismatched.
    """
    codes_t, sf = _resolve_codes(codes, size_factors, model.n_codebooks)
    n_cells = codes_t.shape[0]
    out = torch.empty((n_cells, model.n_genes), dtype=torch.float32)
    with _inference_mode(model):
        for start in range(0, n_cells, batch_size):
            stop = min(start + batch_size, n_cells)
            out[start:stop] = model.decode_codes(codes_t[start:stop], sf[start:stop])
    return out


def decode_to_params(
    model: "OmicsVQVAE",
    codes: CodesLike,
    size_factors: Optional[SizeFactorsLike] = None,
) -> Dict[str, Tensor]:
    """
    Decode discrete codes to the full reconstruction-likelihood parameters.

    Unlike :func:`decode` (which returns only the mean), this exposes every
    per-gene distribution parameter of the configured head — useful for sampling
    synthetic profiles or inspecting dispersion / dropout.

    Parameters
    ----------
    model : OmicsVQVAE
        The trained model the codes were produced by.
    codes : torch.Tensor, numpy.ndarray, or EncodedCells
        Per-cell codes of shape ``(n_cells, n_codebooks)``, or an
        :class:`EncodedCells` bundle.
    size_factors : torch.Tensor or numpy.ndarray, optional
        Per-cell size factors; defaults to the bundle's factors when ``codes`` is
        an :class:`EncodedCells`.

    Returns
    -------
    Dict[str, torch.Tensor]
        The per-gene distribution parameters (see the configured head).
    """
    codes_t, sf = _resolve_codes(codes, size_factors, model.n_codebooks)
    with _inference_mode(model):
        return model.codes_to_params(codes_t, sf)
