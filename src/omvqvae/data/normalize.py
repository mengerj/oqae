"""
Internal normalization and size-factor helpers for OQAE.

OQAE ingests **raw counts** and performs normalization *internally* (scVI-style):
the model reconstructs counts with a count likelihood (NB/ZINB) using an observed
size factor, while the encoder applies an internal ``log1p`` (and optional
per-cell normalization) purely for numerical stability. None of these are
user-facing preprocessing steps — they live here so the rest of the pipeline can
share a single, tested implementation.

All helpers accept either a dense :class:`numpy.ndarray` or a SciPy sparse matrix
(cell x gene) and return dense :class:`numpy.ndarray` results.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from scipy import sparse

from omvqvae.utils.logging import get_logger

logger = get_logger(__name__)

# A cell x gene count matrix, dense or sparse.
CountMatrix = Union[np.ndarray, "sparse.spmatrix"]

__all__ = [
    "CountMatrix",
    "compute_size_factors",
    "normalize_counts",
    "to_dense",
]


def to_dense(matrix: CountMatrix) -> np.ndarray:
    """
    Return a dense ``float32`` view/copy of ``matrix``.

    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse.spmatrix
        Cell x gene count matrix.

    Returns
    -------
    numpy.ndarray
        Dense ``float32`` array with the same shape as ``matrix``.
    """
    if sparse.issparse(matrix):
        dense: np.ndarray = matrix.toarray()  # type: ignore[union-attr]
    else:
        dense = np.asarray(matrix)
    return dense.astype(np.float32, copy=False)


def _row_sums(matrix: CountMatrix) -> np.ndarray:
    """Compute per-cell (row) total counts as a 1-D ``float64`` array."""
    if sparse.issparse(matrix):
        sums = np.asarray(matrix.sum(axis=1)).ravel()
    else:
        sums = np.asarray(matrix).sum(axis=1).ravel()
    return sums.astype(np.float64, copy=False)


def compute_size_factors(
    matrix: CountMatrix,
    *,
    mode: str = "total",
    target_sum: Union[float, None] = None,
) -> np.ndarray:
    """
    Compute a per-cell size factor from observed total counts.

    The size factor captures sequencing-depth / library-size variation so the
    generative decoder can reconstruct depth-appropriate counts.

    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse.spmatrix
        Raw count matrix of shape ``(n_cells, n_genes)``.
    mode : {"total", "ratio"}, default "total"
        ``"total"`` returns the raw per-cell total counts. ``"ratio"`` returns
        each cell's total divided by ``target_sum`` (or the median total when
        ``target_sum`` is ``None``), yielding factors centred near 1.0.
    target_sum : float, optional
        Reference depth used when ``mode="ratio"``. Defaults to the median of the
        per-cell totals.

    Returns
    -------
    numpy.ndarray
        Size factors of shape ``(n_cells,)`` as ``float32``. Cells with zero
        total counts receive a size factor of 1.0 to avoid divide-by-zero
        downstream.

    Raises
    ------
    ValueError
        If ``mode`` is not one of ``{"total", "ratio"}``.
    """
    totals = _row_sums(matrix)

    if mode == "total":
        factors = totals
    elif mode == "ratio":
        reference = target_sum
        if reference is None:
            positive = totals[totals > 0]
            reference = float(np.median(positive)) if positive.size else 1.0
        if reference <= 0:
            raise ValueError("target_sum must be positive.")
        factors = totals / reference
    else:
        raise ValueError(f"Unknown size-factor mode: {mode!r}.")

    # Guard against empty cells so downstream division stays finite.
    factors = np.where(totals > 0, factors, 1.0)
    return factors.astype(np.float32, copy=False)


def normalize_counts(
    matrix: CountMatrix,
    *,
    target_sum: Union[float, None] = 1e4,
    log1p: bool = True,
    size_factors: Union[np.ndarray, None] = None,
) -> np.ndarray:
    """
    Apply OQAE's internal normalization to raw counts.

    Each cell is optionally rescaled to a common depth and then ``log1p``
    transformed. This mirrors the encoder's internal transform and is provided
    here so it can be reused and unit-tested in isolation.

    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse.spmatrix
        Raw count matrix of shape ``(n_cells, n_genes)``.
    target_sum : float, optional
        Per-cell total to normalize to before the log transform. If ``None``, no
        depth normalization is applied (only the optional ``log1p``).
    log1p : bool, default True
        Whether to apply ``log1p`` after (optional) depth normalization.
    size_factors : numpy.ndarray, optional
        Pre-computed per-cell totals of shape ``(n_cells,)``. When ``None`` and
        ``target_sum`` is set, totals are computed from ``matrix``.

    Returns
    -------
    numpy.ndarray
        Normalized matrix of shape ``(n_cells, n_genes)`` as ``float32``.
    """
    dense = to_dense(matrix)

    if target_sum is not None:
        if size_factors is None:
            totals = dense.sum(axis=1)
        else:
            totals = np.asarray(size_factors, dtype=np.float64).ravel()
        safe_totals = np.where(totals > 0, totals, 1.0)
        dense = dense / safe_totals[:, None] * float(target_sum)

    if log1p:
        dense = np.log1p(dense)

    return dense.astype(np.float32, copy=False)


def _is_count_like(matrix: Any) -> bool:  # pragma: no cover - convenience guard
    """Best-effort check that values look like non-negative counts."""
    sample = to_dense(matrix) if not sparse.issparse(matrix) else matrix.data
    arr = np.asarray(sample)
    return bool(arr.size == 0 or arr.min() >= 0)
