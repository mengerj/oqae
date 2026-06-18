"""
Organism-aware gene vocabulary, gene-space alignment, and the shared minibatch
contract for OQAE.

Every data source (local AnnData now; CELLxGENE Census later) is normalized onto
a single, per-organism feature ordering defined by a :class:`GeneVocabulary`, and
yields the same ``(counts, covariates)`` minibatch via :class:`Minibatch`. The
v1 model is unconditional, but the covariates still travel with every batch so
conditioning can be enabled later without changing the data format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from omvqvae.data.normalize import CountMatrix, compute_size_factors, to_dense
from omvqvae.utils.logging import get_logger

logger = get_logger(__name__)

#: Organisms supported by OQAE v1 (one model per organism).
SUPPORTED_ORGANISMS = ("homo_sapiens", "mus_musculus")

__all__ = [
    "SUPPORTED_ORGANISMS",
    "GeneVocabulary",
    "align_to_reference",
    "Minibatch",
    "CountsDataset",
    "collate_minibatch",
]


class GeneVocabulary:
    """
    An ordered, per-organism reference gene set.

    The vocabulary fixes the feature ordering that the model expects; any data
    source is aligned to it (missing genes zero-filled, extra genes dropped).

    Parameters
    ----------
    organism : str
        Organism identifier, e.g. ``"homo_sapiens"`` or ``"mus_musculus"``.
    gene_ids : Iterable[str]
        Ordered, unique gene identifiers (e.g. Ensembl IDs). Order *is* the
        model's feature ordering.

    Raises
    ------
    ValueError
        If ``gene_ids`` is empty or contains duplicates.
    """

    def __init__(self, organism: str, gene_ids: Iterable[str]) -> None:
        ids = [str(g) for g in gene_ids]
        if not ids:
            raise ValueError("GeneVocabulary requires at least one gene id.")
        index: Dict[str, int] = {}
        for pos, gene in enumerate(ids):
            if gene in index:
                raise ValueError(f"Duplicate gene id in vocabulary: {gene!r}.")
            index[gene] = pos
        if organism not in SUPPORTED_ORGANISMS:
            logger.warning(
                "Organism %r is not in the v1-supported set %s.",
                organism,
                SUPPORTED_ORGANISMS,
            )
        self.organism = organism
        self._gene_ids = ids
        self._index = index

    @property
    def gene_ids(self) -> List[str]:
        """Ordered reference gene identifiers."""
        return list(self._gene_ids)

    @property
    def n_genes(self) -> int:
        """Number of genes in the reference."""
        return len(self._gene_ids)

    def position_of(self, gene_id: str) -> Optional[int]:
        """Return the reference position of ``gene_id`` (or ``None``)."""
        return self._index.get(str(gene_id))

    def __len__(self) -> int:
        return self.n_genes

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return f"GeneVocabulary(organism={self.organism!r}, " f"n_genes={self.n_genes})"


def align_to_reference(
    matrix: CountMatrix,
    source_gene_ids: Sequence[str],
    vocabulary: GeneVocabulary,
    *,
    min_overlap: float = 0.0,
) -> np.ndarray:
    """
    Align a source count matrix onto a reference gene vocabulary.

    Genes present in the reference but missing from the source are zero-filled;
    genes present in the source but absent from the reference are dropped.

    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse.spmatrix
        Source count matrix of shape ``(n_cells, len(source_gene_ids))``.
    source_gene_ids : Sequence[str]
        Gene identifiers labelling the columns of ``matrix``.
    vocabulary : GeneVocabulary
        Target reference defining the output column ordering.
    min_overlap : float, default 0.0
        Warn when the fraction of *reference* genes covered by the source falls
        below this threshold (in ``[0, 1]``).

    Returns
    -------
    numpy.ndarray
        Aligned matrix of shape ``(n_cells, vocabulary.n_genes)`` as ``float32``.

    Raises
    ------
    ValueError
        If ``matrix`` column count does not match ``len(source_gene_ids)``.
    """
    n_cells = matrix.shape[0]
    n_source_cols = matrix.shape[1]
    if n_source_cols != len(source_gene_ids):
        raise ValueError(
            "matrix has "
            f"{n_source_cols} columns but {len(source_gene_ids)} gene ids "
            "were provided."
        )

    src_positions: List[int] = []
    ref_positions: List[int] = []
    for col, gene in enumerate(source_gene_ids):
        ref_pos = vocabulary.position_of(gene)
        if ref_pos is not None:
            src_positions.append(col)
            ref_positions.append(ref_pos)

    n_matched = len(ref_positions)
    overlap = n_matched / vocabulary.n_genes if vocabulary.n_genes else 0.0
    if overlap < min_overlap:
        logger.warning(
            "Low gene overlap for organism %r: %d/%d reference genes matched "
            "(%.1f%% < %.1f%% threshold).",
            vocabulary.organism,
            n_matched,
            vocabulary.n_genes,
            overlap * 100.0,
            min_overlap * 100.0,
        )
    else:
        logger.info(
            "Aligned %d/%d reference genes (%.1f%% coverage) for organism %r.",
            n_matched,
            vocabulary.n_genes,
            overlap * 100.0,
            vocabulary.organism,
        )

    aligned = np.zeros((n_cells, vocabulary.n_genes), dtype=np.float32)
    if n_matched:
        dense = to_dense(matrix)
        aligned[:, ref_positions] = dense[:, src_positions]
    return aligned


@dataclass
class Minibatch:
    """
    The shared minibatch contract produced by every OQAE data source.

    Attributes
    ----------
    counts : torch.Tensor
        Raw counts of shape ``(batch, n_genes)`` (``float32``), aligned to the
        organism's :class:`GeneVocabulary`.
    size_factors : torch.Tensor
        Per-cell size factors of shape ``(batch,)`` (``float32``).
    covariates : Dict[str, List[object]]
        Per-cell metadata carried with the batch (always includes
        ``"organism"``; typically a ``"batch"`` / dataset id). Unused by the v1
        unconditional model but available for later conditioning.
    """

    counts: torch.Tensor
    size_factors: torch.Tensor
    covariates: Dict[str, List[object]]

    def __len__(self) -> int:
        return int(self.counts.shape[0])


class CountsDataset(Dataset[Mapping[str, object]]):
    """
    In-memory :class:`torch.utils.data.Dataset` over aligned raw counts.

    Parameters
    ----------
    counts : numpy.ndarray or scipy.sparse.spmatrix
        Count matrix of shape ``(n_cells, n_genes)``, already aligned to the
        reference vocabulary.
    organism : str
        Organism identifier recorded on every sample.
    batch_ids : Sequence, optional
        Per-cell batch / dataset identifier. Defaults to ``None`` for every cell.
    size_factor_mode : str, default "total"
        Mode passed to :func:`omvqvae.data.normalize.compute_size_factors`.

    Raises
    ------
    ValueError
        If ``batch_ids`` length does not match the number of cells.
    """

    def __init__(
        self,
        counts: CountMatrix,
        organism: str,
        *,
        batch_ids: Optional[Sequence[object]] = None,
        size_factor_mode: str = "total",
    ) -> None:
        self._counts = to_dense(counts)
        self.organism = organism
        n_cells = self._counts.shape[0]
        if batch_ids is None:
            self._batch_ids: List[object] = [None] * n_cells
        else:
            batch_list = list(batch_ids)
            if len(batch_list) != n_cells:
                raise ValueError(
                    f"batch_ids has length {len(batch_list)} but there are "
                    f"{n_cells} cells."
                )
            self._batch_ids = batch_list
        self._size_factors = compute_size_factors(self._counts, mode=size_factor_mode)

    def __len__(self) -> int:
        return int(self._counts.shape[0])

    def __getitem__(self, index: int) -> Dict[str, object]:
        return {
            "counts": torch.from_numpy(self._counts[index].copy()),
            "size_factor": torch.tensor(
                float(self._size_factors[index]), dtype=torch.float32
            ),
            "organism": self.organism,
            "batch": self._batch_ids[index],
        }


def collate_minibatch(samples: Sequence[Mapping[str, object]]) -> Minibatch:
    """
    Collate per-cell samples into a :class:`Minibatch`.

    Parameters
    ----------
    samples : Sequence[Mapping[str, object]]
        Samples as produced by :class:`CountsDataset` (or any source emitting the
        same keys: ``counts``, ``size_factor``, ``organism``, ``batch``).

    Returns
    -------
    Minibatch
        Stacked counts/size-factors with collated covariates.
    """
    counts = torch.stack([torch.as_tensor(s["counts"]) for s in samples])
    size_factors = torch.stack([torch.as_tensor(s["size_factor"]) for s in samples])
    covariates: Dict[str, List[object]] = {
        "organism": [s["organism"] for s in samples],
        "batch": [s["batch"] for s in samples],
    }
    return Minibatch(counts=counts, size_factors=size_factors, covariates=covariates)
