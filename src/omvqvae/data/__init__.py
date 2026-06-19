"""
Data layer for OQAE.

Organism-aware loading of raw single-cell counts onto a shared minibatch
contract. This covers local AnnData (``.h5ad`` / ``.zarr``) and CELLxGENE Census
streaming (TileDB-SOMA), plus the gene-space alignment, normalization, and
minibatch primitives — every source yields the same :class:`Minibatch`.
"""

from omvqvae.data.anndata_io import (
    build_anndata_dataloader,
    extract_counts,
    load_anndata,
)
from omvqvae.data.census import (
    DEFAULT_CENSUS_VERSION,
    CensusMinibatchLoader,
    build_census_dataloader,
    census_chunk_to_minibatch,
    census_experiment_name,
    census_gene_vocabulary,
    open_census,
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
    "DEFAULT_CENSUS_VERSION",
    "census_experiment_name",
    "open_census",
    "census_gene_vocabulary",
    "census_chunk_to_minibatch",
    "CensusMinibatchLoader",
    "build_census_dataloader",
]
