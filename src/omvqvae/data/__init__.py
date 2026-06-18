"""
Data layer for OQAE.

Organism-aware loading of raw single-cell counts onto a shared minibatch
contract. This slice covers local AnnData (``.h5ad`` / ``.zarr``) plus the
gene-space alignment, normalization, and minibatch primitives; CELLxGENE Census
streaming lands next, reusing the same :class:`Minibatch` contract.
"""

from omvqvae.data.anndata_io import (
    build_anndata_dataloader,
    extract_counts,
    load_anndata,
)
from omvqvae.data.dataset import (
    SUPPORTED_ORGANISMS,
    CountsDataset,
    GeneVocabulary,
    Minibatch,
    align_to_reference,
    collate_minibatch,
)
from omvqvae.data.normalize import (
    compute_size_factors,
    normalize_counts,
    to_dense,
)

__all__ = [
    # vocabulary / alignment / contract
    "SUPPORTED_ORGANISMS",
    "GeneVocabulary",
    "align_to_reference",
    "Minibatch",
    "CountsDataset",
    "collate_minibatch",
    # normalization
    "compute_size_factors",
    "normalize_counts",
    "to_dense",
    # local AnnData
    "load_anndata",
    "extract_counts",
    "build_anndata_dataloader",
]
