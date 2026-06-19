"""
CELLxGENE Census streaming loaders (TileDB-SOMA) for OQAE.

This module streams **raw single-cell counts** directly from the CZ CELLxGENE
Census without materializing the full corpus in memory, exposing the *same*
:class:`~omvqvae.data.dataset.Minibatch` contract as the local-AnnData path.

It is organism-aware: ``organism`` selects the ``homo_sapiens`` /
``mus_musculus`` Census experiment, and each organism's reference gene set is
built from that experiment's ``var`` index. The pipeline is:

``cellxgene_census.open_soma`` → ``ExperimentAxisQuery`` (``obs``/``var`` value
filters, ``"raw"`` layer) → ``tiledbsoma_ml.ExperimentDataset`` +
``experiment_dataloader`` → aligned :class:`Minibatch`.

The heavy TileDB-SOMA wiring requires a live Census and is exercised by a
network-gated test; the chunk-to-:class:`Minibatch` glue is pure-Python and
fully unit-tested offline with synthetic ``(X, obs)`` fixtures.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
)

import numpy as np
import torch

from omvqvae.data.dataset import (
    SUPPORTED_ORGANISMS,
    GeneVocabulary,
    Minibatch,
    align_to_reference,
)
from omvqvae.data.normalize import compute_size_factors
from omvqvae.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

logger = get_logger(__name__)

#: Pinned default Census LTS release (newest long-term-support build).
DEFAULT_CENSUS_VERSION = "2025-11-08"

#: Census measurement holding scRNA-seq data.
DEFAULT_MEASUREMENT = "RNA"

#: Census ``X`` layer carrying raw counts.
RAW_LAYER = "raw"

#: ``var`` column providing stable gene identifiers (Ensembl IDs).
FEATURE_ID_COLUMN = "feature_id"

__all__ = [
    "DEFAULT_CENSUS_VERSION",
    "census_experiment_name",
    "open_census",
    "census_gene_vocabulary",
    "census_chunk_to_minibatch",
    "CensusMinibatchLoader",
    "build_census_dataloader",
]


def census_experiment_name(organism: str) -> str:
    """
    Map an OQAE organism identifier to its Census experiment name.

    Parameters
    ----------
    organism : str
        Organism identifier, e.g. ``"homo_sapiens"`` or ``"mus_musculus"``.

    Returns
    -------
    str
        The Census experiment name (identical to ``organism`` for the v1
        supported organisms).

    Raises
    ------
    ValueError
        If ``organism`` is not one of :data:`SUPPORTED_ORGANISMS`.
    """
    if organism not in SUPPORTED_ORGANISMS:
        raise ValueError(
            f"Unsupported organism {organism!r}; expected one of "
            f"{SUPPORTED_ORGANISMS}."
        )
    return organism


@contextmanager
def open_census(
    census_version: str = DEFAULT_CENSUS_VERSION,
    **open_kwargs: Any,
) -> Iterator[Any]:  # pragma: no cover - requires live Census/TileDB-SOMA
    """
    Open a pinned CELLxGENE Census as a context manager.

    Parameters
    ----------
    census_version : str, default :data:`DEFAULT_CENSUS_VERSION`
        Census release to open (a date string such as ``"2025-11-08"``, or an
        alias like ``"stable"`` / ``"latest"``). Pin a concrete date for
        reproducibility.
    **open_kwargs
        Extra keyword arguments forwarded to ``cellxgene_census.open_soma``.

    Yields
    ------
    Any
        The open Census SOMA ``Collection`` handle; closed on exit.
    """
    import cellxgene_census

    census = cellxgene_census.open_soma(census_version=census_version, **open_kwargs)
    try:
        yield census
    finally:
        census.close()


def census_gene_vocabulary(
    census: Any,
    organism: str,
    *,
    var_value_filter: Optional[str] = None,
    column_name: str = FEATURE_ID_COLUMN,
) -> GeneVocabulary:  # pragma: no cover - requires live Census/TileDB-SOMA
    """
    Build a per-organism :class:`GeneVocabulary` from the Census ``var`` index.

    Parameters
    ----------
    census : Any
        An open Census handle (see :func:`open_census`).
    organism : str
        Organism identifier selecting the Census experiment.
    var_value_filter : str, optional
        TileDB-SOMA value filter restricting the gene panel (e.g.
        ``"feature_biotype == 'gene'"``). ``None`` keeps every gene.
    column_name : str, default :data:`FEATURE_ID_COLUMN`
        ``var`` column providing the gene identifiers (and feature ordering).

    Returns
    -------
    GeneVocabulary
        Reference gene set for ``organism`` in Census ``var`` order.
    """
    import cellxgene_census

    var_df = cellxgene_census.get_var(
        census,
        census_experiment_name(organism),
        value_filter=var_value_filter,
        column_names=[column_name],
    )
    gene_ids = [str(g) for g in var_df[column_name].tolist()]
    return GeneVocabulary(organism, gene_ids)


def census_chunk_to_minibatch(
    counts: Any,
    obs: "pd.DataFrame",
    source_gene_ids: Sequence[str],
    vocabulary: GeneVocabulary,
    *,
    organism: str,
    batch_key: Optional[str] = None,
    size_factor_mode: str = "total",
    min_overlap: float = 0.0,
) -> Minibatch:
    """
    Convert one Census ``(X, obs)`` chunk into the shared :class:`Minibatch`.

    This is the pure-Python glue between ``tiledbsoma_ml`` mini-batches and
    OQAE's contract: it aligns the chunk's counts onto ``vocabulary``, derives
    per-cell size factors, and attaches the ``organism`` / ``batch`` covariates.

    Parameters
    ----------
    counts : numpy.ndarray
        Raw counts for the chunk, shape ``(n_cells, len(source_gene_ids))``. A
        1-D array (a single cell) is treated as one row.
    obs : pandas.DataFrame
        Matching ``obs`` rows for the chunk (as returned by
        ``ExperimentDataset``).
    source_gene_ids : Sequence[str]
        Gene identifiers labelling the columns of ``counts`` (the query's
        ``var`` order).
    vocabulary : GeneVocabulary
        Target reference defining the output gene ordering.
    organism : str
        Organism identifier recorded on every cell of the batch.
    batch_key : str, optional
        ``obs`` column carried as the per-cell ``"batch"`` covariate. When
        ``None`` or absent from ``obs``, the batch covariate is ``None``.
    size_factor_mode : str, default "total"
        Mode passed to
        :func:`omvqvae.data.normalize.compute_size_factors`.
    min_overlap : float, default 0.0
        Warn-below threshold for reference-gene coverage during alignment.

    Returns
    -------
    Minibatch
        Aligned counts, size factors, and covariates for the chunk.
    """
    matrix = np.asarray(counts)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    aligned = align_to_reference(
        matrix, source_gene_ids, vocabulary, min_overlap=min_overlap
    )
    size_factors = compute_size_factors(aligned, mode=size_factor_mode)
    n_cells = aligned.shape[0]

    if batch_key is not None and batch_key in obs:
        batch_ids: List[object] = [obj for obj in obs[batch_key].tolist()]
    else:
        batch_ids = [None] * n_cells

    covariates: Dict[str, List[object]] = {
        "organism": [organism] * n_cells,
        "batch": batch_ids,
    }
    return Minibatch(
        counts=torch.from_numpy(aligned),
        size_factors=torch.from_numpy(size_factors),
        covariates=covariates,
    )


class CensusMinibatchLoader:
    """
    Iterable adapter turning a Census ``ExperimentDataset`` into Minibatches.

    Wrapping the ``experiment_dataloader`` iterator, this yields
    :class:`Minibatch` objects — the same contract as the local-AnnData
    DataLoader — so downstream code iterates either source identically.

    Parameters
    ----------
    dataloader : Iterable
        An iterable of ``(counts, obs)`` chunks (a ``tiledbsoma_ml``
        ``experiment_dataloader``, or any compatible iterable for testing).
    source_gene_ids : Sequence[str]
        Gene identifiers labelling the columns of each chunk's counts.
    vocabulary : GeneVocabulary
        Reference defining the output gene ordering.
    organism : str
        Organism recorded on every emitted batch.
    batch_key : str, optional
        ``obs`` column carried as the ``"batch"`` covariate.
    size_factor_mode : str, default "total"
        Size-factor mode (see
        :func:`omvqvae.data.normalize.compute_size_factors`).
    min_overlap : float, default 0.0
        Warn-below threshold for reference-gene coverage during alignment.
    query : Any, optional
        The underlying ``ExperimentAxisQuery`` to close on :meth:`close`.
    """

    def __init__(
        self,
        dataloader: Any,
        *,
        source_gene_ids: Sequence[str],
        vocabulary: GeneVocabulary,
        organism: str,
        batch_key: Optional[str] = None,
        size_factor_mode: str = "total",
        min_overlap: float = 0.0,
        query: Optional[Any] = None,
    ) -> None:
        self._dataloader = dataloader
        self._source_gene_ids = [str(g) for g in source_gene_ids]
        self._vocabulary = vocabulary
        self._organism = organism
        self._batch_key = batch_key
        self._size_factor_mode = size_factor_mode
        self._min_overlap = min_overlap
        self._query = query

    @property
    def vocabulary(self) -> GeneVocabulary:
        """Reference gene vocabulary the batches are aligned to."""
        return self._vocabulary

    def __iter__(self) -> Iterator[Minibatch]:
        for counts, obs in self._dataloader:
            yield census_chunk_to_minibatch(
                counts,
                obs,
                self._source_gene_ids,
                self._vocabulary,
                organism=self._organism,
                batch_key=self._batch_key,
                size_factor_mode=self._size_factor_mode,
                min_overlap=self._min_overlap,
            )

    def close(self) -> None:
        """Close the underlying ``ExperimentAxisQuery`` if one was supplied."""
        if self._query is not None:
            self._query.close()


def build_census_dataloader(
    census: Any,
    organism: str,
    vocabulary: Optional[GeneVocabulary] = None,
    *,
    measurement_name: str = DEFAULT_MEASUREMENT,
    obs_value_filter: Optional[str] = None,
    var_value_filter: Optional[str] = None,
    column_name: str = FEATURE_ID_COLUMN,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    batch_key: Optional[str] = None,
    size_factor_mode: str = "total",
    min_overlap: float = 0.0,
    seed: Optional[int] = None,
) -> CensusMinibatchLoader:  # pragma: no cover - requires live Census/TileDB-SOMA
    """
    Stream a Census slice as :class:`Minibatch` batches.

    Builds an ``ExperimentAxisQuery`` for ``organism`` (optionally filtered by
    ``obs``/``var`` value filters), reads raw counts via
    ``tiledbsoma_ml.ExperimentDataset``, and wraps the resulting dataloader in a
    :class:`CensusMinibatchLoader` aligned to ``vocabulary``.

    Parameters
    ----------
    census : Any
        An open Census handle (see :func:`open_census`). The handle and the
        query must stay open while iterating the returned loader.
    organism : str
        Organism identifier selecting the Census experiment.
    vocabulary : GeneVocabulary, optional
        Reference gene ordering. When ``None``, it is built from the query's
        ``var`` index (i.e. the Census genes selected by ``var_value_filter``).
    measurement_name : str, default :data:`DEFAULT_MEASUREMENT`
        Census measurement to read (``"RNA"`` for scRNA-seq).
    obs_value_filter : str, optional
        TileDB-SOMA value filter selecting cells (e.g.
        ``"tissue_general == 'blood'"``). ``None`` selects all cells.
    var_value_filter : str, optional
        TileDB-SOMA value filter selecting genes. ``None`` selects all genes.
    column_name : str, default :data:`FEATURE_ID_COLUMN`
        ``var`` column providing gene identifiers.
    batch_size : int, default 128
        Number of cells per emitted minibatch.
    shuffle : bool, default True
        Whether ``ExperimentDataset`` shuffles cells while streaming.
    num_workers : int, default 0
        Number of dataloader worker processes.
    batch_key : str, optional
        ``obs`` column carried as the per-cell ``"batch"`` covariate (also
        added to the streamed ``obs`` columns).
    size_factor_mode : str, default "total"
        Size-factor mode (see
        :func:`omvqvae.data.normalize.compute_size_factors`).
    min_overlap : float, default 0.0
        Warn-below threshold for reference-gene coverage during alignment.
    seed : int, optional
        Shuffle seed for reproducible streaming.

    Returns
    -------
    CensusMinibatchLoader
        Iterable yielding :class:`Minibatch` batches aligned to ``vocabulary``.
    """
    import tiledbsoma
    from tiledbsoma_ml import ExperimentDataset, experiment_dataloader

    experiment = census["census_data"][census_experiment_name(organism)]
    query = experiment.axis_query(
        measurement_name=measurement_name,
        obs_query=(
            tiledbsoma.AxisQuery(value_filter=obs_value_filter)
            if obs_value_filter is not None
            else tiledbsoma.AxisQuery()
        ),
        var_query=(
            tiledbsoma.AxisQuery(value_filter=var_value_filter)
            if var_value_filter is not None
            else tiledbsoma.AxisQuery()
        ),
    )

    var_df = query.var(column_names=[column_name]).concat().to_pandas()
    source_gene_ids = [str(g) for g in var_df[column_name].tolist()]
    if vocabulary is None:
        vocabulary = GeneVocabulary(organism, source_gene_ids)

    obs_column_names: List[str] = ["soma_joinid"]
    if batch_key is not None and batch_key not in obs_column_names:
        obs_column_names.append(batch_key)

    # tiledbsoma_ml defines ``ExperimentDataset`` via attrs with a custom
    # ``__init__`` (the documented ``{query, layer_name}`` constructor); mypy's
    # attrs plugin resolves the generated field signature instead, so the
    # public keywords need an explicit ignore.
    dataset = ExperimentDataset(  # type: ignore[call-arg]
        query,
        layer_name=RAW_LAYER,
        obs_column_names=obs_column_names,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    dataloader = experiment_dataloader(dataset, num_workers=num_workers)

    logger.info(
        "Streaming Census %r (%d genes) for organism %r.",
        measurement_name,
        vocabulary.n_genes,
        organism,
    )
    return CensusMinibatchLoader(
        dataloader,
        source_gene_ids=source_gene_ids,
        vocabulary=vocabulary,
        organism=organism,
        batch_key=batch_key,
        size_factor_mode=size_factor_mode,
        min_overlap=min_overlap,
        query=query,
    )
