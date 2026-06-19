"""
Data layer for OQAE.

Organism-aware loading of raw single-cell counts onto a shared minibatch
contract. This covers local AnnData (``.h5ad`` / ``.zarr``) and CELLxGENE Census
streaming (TileDB-SOMA), plus the gene-space alignment, normalization, and
minibatch primitives — every source reuses the same :class:`Minibatch` contract.
"""

from omvqvae.data.anndata_io import (
    build_anndata_dataloader,
    extract_counts,
    load_anndata,
)
from omvqvae.data.census import (
    CensusMinibatchLoader,
    build_census_dataloader,
    census_batch_to_minibatch,
    gene_vocabulary_from_var,
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
    # CELLxGENE Census streaming
    "census_batch_to_minibatch",
    "gene_vocabulary_from_var",
    "CensusMinibatchLoader",
    "build_census_dataloader",
]
