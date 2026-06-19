"""
CELLxGENE Census streaming loaders for OQAE (TileDB-SOMA).

This module streams **raw counts** from the CZ CELLxGENE Census via the
maintained TileDB-SOMA stack (``cellxgene_census`` + ``tiledbsoma`` +
``tiledbsoma_ml``) and normalizes them onto OQAE's shared minibatch contract —
the *same* :class:`~omvqvae.data.dataset.Minibatch` the local AnnData loader
emits. ``organism`` selects the ``homo_sapiens`` / ``mus_musculus`` experiment,
``obs_value_filter`` slices the cells, and the per-organism reference gene set is
built from the Census ``var`` index.

The heavy TileDB-SOMA dependencies are imported lazily (inside functions), so
importing this module — and exercising the offline alignment/contract glue — does
not require them. Live streaming is network-gated and tested separately.
"""

from __future__ import annotations

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
from omvqvae.data.normalize import CountMatrix, compute_size_factors
from omvqvae.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

logger = get_logger(__name__)

#: Default pinned Census version (newest known stable at implementation time).
DEFAULT_CENSUS_VERSION = "2025-01-30"
#: Census measurement holding scRNA-seq data.
DEFAULT_MEASUREMENT = "RNA"
#: Census X layer carrying raw counts.
DEFAULT_X_LAYER = "raw"
#: Census ``var`` column holding the stable (Ensembl) gene id.
DEFAULT_GENE_ID_COLUMN = "feature_id"

__all__ = [
    "DEFAULT_CENSUS_VERSION",
    "DEFAULT_MEASUREMENT",
    "DEFAULT_X_LAYER",
    "DEFAULT_GENE_ID_COLUMN",
    "census_batch_to_minibatch",
    "gene_vocabulary_from_var",
    "CensusMinibatchLoader",
    "build_census_dataloader",
]


def _close_quietly(objs: Sequence[Any]) -> None:
    """Close each object, logging (not raising) on failure."""
    for obj in objs:
        try:
            obj.close()
        except Exception:  # best-effort cleanup
            logger.warning("Failed to close Census handle %r.", obj)


def _validate_organism(organism: str) -> None:
    """Raise if ``organism`` is not a v1-supported Census experiment."""
    if organism not in SUPPORTED_ORGANISMS:
        raise ValueError(
            f"Unsupported organism {organism!r}; OQAE v1 supports "
            f"{SUPPORTED_ORGANISMS}."
        )


def census_batch_to_minibatch(
    x: CountMatrix,
    obs: "pd.DataFrame",
    source_gene_ids: Sequence[str],
    vocabulary: GeneVocabulary,
    *,
    batch_key: Optional[str] = None,
    size_factor_mode: str = "total",
    min_overlap: float = 0.0,
) -> Minibatch:
    """
    Convert a raw TileDB-SOMA ``(X, obs)`` batch into a :class:`Minibatch`.

    This is the offline-testable glue between ``tiledbsoma_ml`` batches and
    OQAE's shared contract: it aligns the counts to the organism reference,
    derives per-cell size factors, and carries ``organism`` + ``batch``
    covariates.

    Parameters
    ----------
    x : numpy.ndarray or scipy.sparse.spmatrix
        Raw counts for the batch. Shape ``(batch, n_source_genes)``; a 1-D array
        (single cell) is treated as one row.
    obs : pandas.DataFrame
        Per-cell observation metadata for the batch (``tiledbsoma_ml`` yields one
        row per cell, matched to ``x``).
    source_gene_ids : Sequence[str]
        Gene identifiers labelling the columns of ``x`` (the Census ``var``
        ordering for the query).
    vocabulary : GeneVocabulary
        Organism reference defining the output gene ordering.
    batch_key : str, optional
        Column in ``obs`` carried as the per-cell ``"batch"`` covariate.
    size_factor_mode : str, default "total"
        Mode passed to :func:`omvqvae.data.normalize.compute_size_factors`.
    min_overlap : float, default 0.0
        Warn-below threshold for reference-gene coverage during alignment.

    Returns
    -------
    Minibatch
        Counts aligned to ``vocabulary`` with size factors and covariates.

    Raises
    ------
    ValueError
        If the number of rows in ``x`` and ``obs`` disagree.
    """
    matrix: CountMatrix = x
    if not hasattr(matrix, "ndim") or matrix.ndim == 1:  # single-cell batch
        matrix = np.atleast_2d(np.asarray(matrix))

    n_cells = matrix.shape[0]
    if len(obs) != n_cells:
        raise ValueError(
            f"X has {n_cells} rows but obs has {len(obs)} rows; they must match."
        )

    aligned = align_to_reference(
        matrix, source_gene_ids, vocabulary, min_overlap=min_overlap
    )
    size_factors = compute_size_factors(aligned, mode=size_factor_mode)

    if batch_key is not None:
        batch_ids: List[object] = [obj for obj in obs[batch_key].tolist()]
    else:
        batch_ids = [None] * n_cells

    covariates: Dict[str, List[object]] = {
        "organism": [vocabulary.organism] * n_cells,
        "batch": batch_ids,
    }
    return Minibatch(
        counts=torch.from_numpy(aligned),
        size_factors=torch.from_numpy(size_factors),
        covariates=covariates,
    )


def gene_vocabulary_from_var(
    var: "pd.DataFrame",
    organism: str,
    *,
    gene_id_column: str = DEFAULT_GENE_ID_COLUMN,
) -> GeneVocabulary:
    """
    Build a :class:`GeneVocabulary` from a Census ``var`` DataFrame.

    The reference gene ordering is taken directly from the (already ordered)
    ``var`` frame, using ``gene_id_column`` for the stable identifiers.

    Parameters
    ----------
    var : pandas.DataFrame
        The Census ``var`` frame for the organism / query.
    organism : str
        Organism identifier (``"homo_sapiens"`` / ``"mus_musculus"``).
    gene_id_column : str, default "feature_id"
        Column holding the stable (Ensembl) gene id. If absent, the frame index
        is used.

    Returns
    -------
    GeneVocabulary
        Reference vocabulary for ``organism``.
    """
    _validate_organism(organism)
    if gene_id_column in var.columns:
        gene_ids = [str(g) for g in var[gene_id_column].tolist()]
    else:
        gene_ids = [str(g) for g in var.index.tolist()]
    return GeneVocabulary(organism, gene_ids)


class CensusMinibatchLoader:
    """
    Iterable wrapper turning TileDB-SOMA batches into :class:`Minibatch` objects.

    Wraps an inner iterable of ``(X, obs)`` batches (a ``tiledbsoma_ml``
    DataLoader) and yields :class:`Minibatch` objects aligned to ``vocabulary`` —
    matching the contract shared with the local AnnData loader. Usable as a
    context manager to release any open Census handles.

    Parameters
    ----------
    inner : Iterable
        Iterable yielding ``(X, obs)`` tuples (e.g. the ``tiledbsoma_ml``
        ``experiment_dataloader``).
    source_gene_ids : Sequence[str]
        Gene identifiers for the columns of each batch's ``X``.
    vocabulary : GeneVocabulary
        Organism reference defining the output gene ordering.
    batch_key : str, optional
        ``obs`` column carried as the per-cell ``"batch"`` covariate.
    size_factor_mode : str, default "total"
        Size-factor mode (see
        :func:`omvqvae.data.normalize.compute_size_factors`).
    min_overlap : float, default 0.0
        Warn-below threshold for reference-gene coverage during alignment.
    closeables : Sequence, optional
        Objects with a ``close()`` method to release when this loader is closed
        (e.g. the open Census / experiment query).
    """

    def __init__(
        self,
        inner: Any,
        source_gene_ids: Sequence[str],
        vocabulary: GeneVocabulary,
        *,
        batch_key: Optional[str] = None,
        size_factor_mode: str = "total",
        min_overlap: float = 0.0,
        closeables: Optional[Sequence[Any]] = None,
    ) -> None:
        self._inner = inner
        self._source_gene_ids = list(source_gene_ids)
        self._vocabulary = vocabulary
        self._batch_key = batch_key
        self._size_factor_mode = size_factor_mode
        self._min_overlap = min_overlap
        self._closeables: List[Any] = list(closeables) if closeables else []

    @property
    def vocabulary(self) -> GeneVocabulary:
        """The organism reference vocabulary batches are aligned to."""
        return self._vocabulary

    def __iter__(self) -> Iterator[Minibatch]:
        for x, obs in self._inner:
            yield census_batch_to_minibatch(
                x,
                obs,
                self._source_gene_ids,
                self._vocabulary,
                batch_key=self._batch_key,
                size_factor_mode=self._size_factor_mode,
                min_overlap=self._min_overlap,
            )

    def close(self) -> None:
        """Release any open Census / query handles."""
        _close_quietly(self._closeables)
        self._closeables = []

    def __enter__(self) -> "CensusMinibatchLoader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _open_census(census_version: str) -> Any:  # pragma: no cover - network-backed
    """Open a pinned Census version (lazy import of ``cellxgene_census``)."""
    import cellxgene_census  # local import: heavy, network-backed

    logger.info("Opening CELLxGENE Census version %s.", census_version)
    return cellxgene_census.open_soma(census_version=census_version)


def build_census_dataloader(
    organism: str,
    *,
    census_version: str = DEFAULT_CENSUS_VERSION,
    obs_value_filter: Optional[str] = None,
    var_value_filter: Optional[str] = None,
    vocabulary: Optional[GeneVocabulary] = None,
    gene_id_column: str = DEFAULT_GENE_ID_COLUMN,
    batch_key: Optional[str] = None,
    obs_column_names: Optional[Sequence[str]] = None,
    measurement_name: str = DEFAULT_MEASUREMENT,
    x_layer_name: str = DEFAULT_X_LAYER,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: Optional[int] = None,
    size_factor_mode: str = "total",
    min_overlap: float = 0.0,
    census: Optional[Any] = None,
) -> CensusMinibatchLoader:
    """
    Stream a Census slice as :class:`Minibatch` batches (TileDB-SOMA).

    Opens the pinned Census, selects the per-organism experiment, applies the
    ``obs`` / ``var`` value filters, and streams raw counts through
    ``tiledbsoma_ml`` — yielding the *same* :class:`Minibatch` contract as the
    local AnnData loader. The per-organism reference gene set is built from the
    query's ``var`` index unless an explicit ``vocabulary`` is supplied.

    Parameters
    ----------
    organism : str
        ``"homo_sapiens"`` or ``"mus_musculus"`` (selects the Census experiment).
    census_version : str, default :data:`DEFAULT_CENSUS_VERSION`
        Pinned Census version to open.
    obs_value_filter : str, optional
        TileDB-SOMA value filter selecting cells (e.g.
        ``"tissue_general == 'lung'"``).
    var_value_filter : str, optional
        TileDB-SOMA value filter selecting genes.
    vocabulary : GeneVocabulary, optional
        Explicit reference vocabulary. When ``None``, it is built from the
        query's ``var`` index.
    gene_id_column : str, default "feature_id"
        ``var`` column providing the stable gene id used for alignment.
    batch_key : str, optional
        ``obs`` column carried as the per-cell ``"batch"`` covariate (also added
        to the requested ``obs`` columns).
    obs_column_names : Sequence[str], optional
        ``obs`` columns to fetch. Defaults to ``("soma_joinid",)`` plus
        ``batch_key`` when given.
    measurement_name : str, default "RNA"
        Census measurement to read.
    x_layer_name : str, default "raw"
        X layer to read (raw counts).
    batch_size : int, default 128
        Number of cells per streamed minibatch.
    shuffle : bool, default True
        Whether ``tiledbsoma_ml`` shuffles cells while streaming.
    num_workers : int, default 0
        DataLoader worker processes.
    seed : int, optional
        Shuffle seed for reproducibility.
    size_factor_mode : str, default "total"
        Size-factor mode (see
        :func:`omvqvae.data.normalize.compute_size_factors`).
    min_overlap : float, default 0.0
        Warn-below threshold for reference-gene coverage during alignment.
    census : optional
        An already-open Census handle (skips opening / closing internally).

    Returns
    -------
    CensusMinibatchLoader
        An iterable (and context manager) yielding :class:`Minibatch` batches.
    """
    import tiledbsoma  # local import: heavy
    import tiledbsoma_ml  # local import: heavy

    _validate_organism(organism)

    owns_census = census is None
    census_handle = census if census is not None else _open_census(census_version)
    closeables: List[Any] = [census_handle] if owns_census else []

    try:
        experiment = census_handle["census_data"][organism]

        obs_query = (
            tiledbsoma.AxisQuery(value_filter=obs_value_filter)
            if obs_value_filter
            else None
        )
        var_query = (
            tiledbsoma.AxisQuery(value_filter=var_value_filter)
            if var_value_filter
            else None
        )

        query = experiment.axis_query(
            measurement_name=measurement_name,
            obs_query=obs_query,
            var_query=var_query,
        )
        closeables.insert(0, query)

        var_df = query.var().concat().to_pandas()
        if vocabulary is None:
            vocab = gene_vocabulary_from_var(
                var_df, organism, gene_id_column=gene_id_column
            )
        else:
            vocab = vocabulary
        source_gene_ids = (
            [str(g) for g in var_df[gene_id_column].tolist()]
            if gene_id_column in var_df.columns
            else [str(g) for g in var_df.index.tolist()]
        )

        requested_cols = _resolve_obs_columns(obs_column_names, batch_key)
        # ``ExperimentDataset`` is an attrs class whose runtime ``__init__``
        # (query / layer_name) differs from the attrs-generated signature mypy
        # infers; route through ``Any`` so the real keywords type-check.
        experiment_dataset_cls: Any = tiledbsoma_ml.ExperimentDataset
        dataset = experiment_dataset_cls(
            query=query,
            layer_name=x_layer_name,
            obs_column_names=requested_cols,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )
        loader = tiledbsoma_ml.experiment_dataloader(dataset, num_workers=num_workers)
    except Exception:
        _close_quietly(closeables)
        raise

    return CensusMinibatchLoader(
        loader,
        source_gene_ids,
        vocab,
        batch_key=batch_key,
        size_factor_mode=size_factor_mode,
        min_overlap=min_overlap,
        closeables=closeables,
    )


def _resolve_obs_columns(
    obs_column_names: Optional[Sequence[str]],
    batch_key: Optional[str],
) -> List[str]:
    """Resolve the ``obs`` columns to fetch, always including ``batch_key``."""
    cols: List[str] = list(obs_column_names) if obs_column_names else ["soma_joinid"]
    if batch_key is not None and batch_key not in cols:
        cols.append(batch_key)
    return cols
