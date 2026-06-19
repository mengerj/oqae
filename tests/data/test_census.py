"""Tests for the CELLxGENE Census streaming loader.

The alignment/contract glue is tested fully offline with synthetic
``(X, obs)`` batches; the live Census streaming test is network-gated and
skipped by default.
"""

from __future__ import annotations

import os
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from omvqvae.data.census import (
    DEFAULT_CENSUS_VERSION,
    CensusMinibatchLoader,
    build_census_dataloader,
    census_batch_to_minibatch,
    gene_vocabulary_from_var,
)
from omvqvae.data.dataset import GeneVocabulary, Minibatch

RUN_LIVE_CENSUS = os.environ.get("OQAE_RUN_CENSUS_TESTS") == "1"


def _obs(n_cells: int, *, with_batch: bool = True) -> pd.DataFrame:
    data = {"soma_joinid": list(range(n_cells))}
    if with_batch:
        data["dataset_id"] = [f"d{i % 2}" for i in range(n_cells)]
    return pd.DataFrame(data)


def test_census_batch_to_minibatch_basic(human_gene_ids: List[str]) -> None:
    vocab = GeneVocabulary("homo_sapiens", human_gene_ids)
    x = np.arange(2 * len(human_gene_ids), dtype=np.float32).reshape(
        2, len(human_gene_ids)
    )
    batch = census_batch_to_minibatch(
        x, _obs(2), human_gene_ids, vocab, batch_key="dataset_id"
    )
    assert isinstance(batch, Minibatch)
    assert batch.counts.shape == (2, len(human_gene_ids))
    assert batch.counts.dtype == torch.float32
    assert batch.covariates["organism"] == ["homo_sapiens", "homo_sapiens"]
    assert batch.covariates["batch"] == ["d0", "d1"]
    # size factors == per-cell totals (default "total" mode)
    assert torch.allclose(batch.size_factors, batch.counts.sum(dim=1))


def test_census_batch_to_minibatch_aligns_and_reorders(
    human_gene_ids: List[str],
) -> None:
    # Source gene order differs from the reference and is missing one gene.
    reference = human_gene_ids  # ENSG1..ENSG4
    source_gene_ids = ["ENSG3", "ENSG1", "ENSG_UNKNOWN"]
    vocab = GeneVocabulary("homo_sapiens", reference)
    x = np.array([[3.0, 1.0, 99.0]], dtype=np.float32)  # 99 -> dropped gene
    batch = census_batch_to_minibatch(x, _obs(1), source_gene_ids, vocab)
    # ENSG1 -> col 0 == 1.0, ENSG3 -> col 2 == 3.0, ENSG2/ENSG4 zero-filled.
    expected = torch.tensor([[1.0, 0.0, 3.0, 0.0]])
    assert torch.allclose(batch.counts, expected)
    # The unknown source gene is dropped, so it does not enter the size factor.
    assert torch.allclose(batch.size_factors, torch.tensor([4.0]))


def test_census_batch_to_minibatch_no_batch_key(human_gene_ids: List[str]) -> None:
    vocab = GeneVocabulary("homo_sapiens", human_gene_ids)
    x = np.ones((3, len(human_gene_ids)), dtype=np.float32)
    batch = census_batch_to_minibatch(
        x, _obs(3, with_batch=False), human_gene_ids, vocab
    )
    assert batch.covariates["batch"] == [None, None, None]


def test_census_batch_to_minibatch_1d_single_cell(human_gene_ids: List[str]) -> None:
    vocab = GeneVocabulary("homo_sapiens", human_gene_ids)
    x = np.ones(len(human_gene_ids), dtype=np.float32)  # 1-D single cell
    batch = census_batch_to_minibatch(
        x, _obs(1, with_batch=False), human_gene_ids, vocab
    )
    assert batch.counts.shape == (1, len(human_gene_ids))


def test_census_batch_to_minibatch_sparse(human_gene_ids: List[str]) -> None:
    vocab = GeneVocabulary("homo_sapiens", human_gene_ids)
    x = sparse.csr_matrix(np.ones((2, len(human_gene_ids)), dtype=np.float32))
    batch = census_batch_to_minibatch(
        x, _obs(2, with_batch=False), human_gene_ids, vocab
    )
    assert batch.counts.shape == (2, len(human_gene_ids))
    assert float(batch.counts.sum()) == 2 * len(human_gene_ids)


def test_census_batch_to_minibatch_row_mismatch(human_gene_ids: List[str]) -> None:
    vocab = GeneVocabulary("homo_sapiens", human_gene_ids)
    x = np.ones((3, len(human_gene_ids)), dtype=np.float32)
    with pytest.raises(ValueError, match="must match"):
        census_batch_to_minibatch(x, _obs(2), human_gene_ids, vocab)


def test_gene_vocabulary_from_var_feature_id_column() -> None:
    var = pd.DataFrame({"feature_id": ["ENSG1", "ENSG2"], "feature_name": ["A", "B"]})
    vocab = gene_vocabulary_from_var(var, "homo_sapiens")
    assert vocab.gene_ids == ["ENSG1", "ENSG2"]
    assert vocab.organism == "homo_sapiens"


def test_gene_vocabulary_from_var_falls_back_to_index() -> None:
    var = pd.DataFrame({"feature_name": ["A", "B"]}, index=["ENSMUSG1", "ENSMUSG2"])
    vocab = gene_vocabulary_from_var(var, "mus_musculus")
    assert vocab.gene_ids == ["ENSMUSG1", "ENSMUSG2"]


def test_gene_vocabulary_from_var_rejects_unknown_organism() -> None:
    var = pd.DataFrame({"feature_id": ["G1"]})
    with pytest.raises(ValueError, match="Unsupported organism"):
        gene_vocabulary_from_var(var, "danio_rerio")


def test_build_census_dataloader_rejects_unknown_organism() -> None:
    with pytest.raises(ValueError, match="Unsupported organism"):
        build_census_dataloader("danio_rerio")


class _FakeInner:
    """A stand-in for the tiledbsoma_ml DataLoader (yields ``(X, obs)``)."""

    def __init__(self, batches: List[Tuple[np.ndarray, pd.DataFrame]]) -> None:
        self._batches = batches

    def __iter__(self) -> Iterator[Tuple[np.ndarray, pd.DataFrame]]:
        return iter(self._batches)


class _Closeable:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_census_minibatch_loader_maps_batches(human_gene_ids: List[str]) -> None:
    vocab = GeneVocabulary("homo_sapiens", human_gene_ids)
    inner = _FakeInner(
        [
            (np.ones((2, len(human_gene_ids)), dtype=np.float32), _obs(2)),
            (np.full((1, len(human_gene_ids)), 2.0, dtype=np.float32), _obs(1)),
        ]
    )
    loader = CensusMinibatchLoader(inner, human_gene_ids, vocab, batch_key="dataset_id")
    assert loader.vocabulary is vocab
    batches = list(loader)
    assert len(batches) == 2
    assert all(isinstance(b, Minibatch) for b in batches)
    assert sum(len(b) for b in batches) == 3
    assert batches[0].covariates["organism"] == ["homo_sapiens", "homo_sapiens"]


class _RaisingCloseable:
    def close(self) -> None:
        raise RuntimeError("cannot close")


def test_census_minibatch_loader_close_swallows_errors() -> None:
    vocab = GeneVocabulary("homo_sapiens", ["G1"])
    loader = CensusMinibatchLoader(
        _FakeInner([]), ["G1"], vocab, closeables=[_RaisingCloseable()]
    )
    # A failing close() is logged, not raised.
    loader.close()


def test_census_minibatch_loader_context_manager_closes() -> None:
    vocab = GeneVocabulary("homo_sapiens", ["G1"])
    handle = _Closeable()
    inner = _FakeInner([(np.ones((1, 1), dtype=np.float32), _obs(1, with_batch=False))])
    with CensusMinibatchLoader(inner, ["G1"], vocab, closeables=[handle]) as loader:
        assert len(list(loader)) == 1
    assert handle.closed is True


def test_default_census_version_pinned() -> None:
    assert DEFAULT_CENSUS_VERSION == "2025-01-30"


def test_resolve_obs_columns() -> None:
    from omvqvae.data.census import _resolve_obs_columns

    assert _resolve_obs_columns(None, None) == ["soma_joinid"]
    assert _resolve_obs_columns(None, "dataset_id") == ["soma_joinid", "dataset_id"]
    # An explicitly requested column list is preserved; batch_key not duplicated.
    assert _resolve_obs_columns(["a", "b"], "b") == ["a", "b"]
    assert _resolve_obs_columns(["a"], "b") == ["a", "b"]


# --- Orchestration tests: fake the TileDB-SOMA stack so build_census_dataloader
# --- can be exercised offline (no network / no real Census). ---


class _FakeArrow:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def concat(self) -> "_FakeArrow":
        return self

    def to_pandas(self) -> pd.DataFrame:
        return self._df


class _FakeQuery:
    def __init__(self, var_df: pd.DataFrame) -> None:
        self._var_df = var_df
        self.closed = False

    def var(self) -> _FakeArrow:
        return _FakeArrow(self._var_df)

    def close(self) -> None:
        self.closed = True


class _FakeExperiment:
    def __init__(self, query: _FakeQuery) -> None:
        self._query = query
        self.axis_query_kwargs: dict = {}

    def axis_query(self, **kwargs: object) -> _FakeQuery:
        self.axis_query_kwargs = kwargs
        return self._query


class _FakeCensus:
    def __init__(self, organism: str, experiment: _FakeExperiment) -> None:
        self._data = {organism: experiment}
        self.closed = False

    def __getitem__(self, key: str) -> dict:
        assert key == "census_data"
        return self._data

    def close(self) -> None:
        self.closed = True


def _patch_tiledbsoma_ml(
    monkeypatch: pytest.MonkeyPatch,
    inner: object,
    *,
    raise_on_loader: bool = False,
) -> dict:
    import tiledbsoma_ml

    recorded: dict = {}

    def fake_dataset(query: object, **kwargs: object) -> object:
        recorded["dataset_kwargs"] = kwargs
        return ("dataset", query)

    def fake_dataloader(ds: object, **kwargs: object) -> object:
        recorded["dataloader_kwargs"] = kwargs
        if raise_on_loader:
            raise RuntimeError("boom")
        return inner

    monkeypatch.setattr(tiledbsoma_ml, "ExperimentDataset", fake_dataset)
    monkeypatch.setattr(tiledbsoma_ml, "experiment_dataloader", fake_dataloader)
    return recorded


def test_build_census_dataloader_with_fakes(
    monkeypatch: pytest.MonkeyPatch, human_gene_ids: List[str]
) -> None:
    var_df = pd.DataFrame({"feature_id": human_gene_ids})
    query = _FakeQuery(var_df)
    census = _FakeCensus("homo_sapiens", _FakeExperiment(query))
    inner = _FakeInner([(np.ones((2, len(human_gene_ids)), dtype=np.float32), _obs(2))])
    recorded = _patch_tiledbsoma_ml(monkeypatch, inner)

    loader = build_census_dataloader(
        "homo_sapiens",
        census=census,
        obs_value_filter="tissue_general == 'blood'",
        var_value_filter="feature_biotype == 'gene'",
        batch_key="dataset_id",
        batch_size=2,
        shuffle=False,
    )

    # Vocabulary derived from the query var index.
    assert loader.vocabulary.gene_ids == human_gene_ids
    batch = next(iter(loader))
    assert isinstance(batch, Minibatch)
    assert batch.covariates["batch"] == ["d0", "d1"]
    # raw layer + size/shuffle forwarded to the dataset.
    assert recorded["dataset_kwargs"]["layer_name"] == "raw"
    assert recorded["dataset_kwargs"]["batch_size"] == 2
    assert recorded["dataset_kwargs"]["shuffle"] is False
    # batch_key folded into the requested obs columns.
    assert "dataset_id" in recorded["dataset_kwargs"]["obs_column_names"]

    # A caller-supplied census is not closed by the loader.
    loader.close()
    assert census.closed is False
    assert query.closed is True


def test_build_census_dataloader_explicit_vocabulary(
    monkeypatch: pytest.MonkeyPatch, human_gene_ids: List[str]
) -> None:
    var_df = pd.DataFrame({"feature_id": human_gene_ids})
    census = _FakeCensus("mus_musculus", _FakeExperiment(_FakeQuery(var_df)))
    inner = _FakeInner([])
    _patch_tiledbsoma_ml(monkeypatch, inner)
    explicit = GeneVocabulary("mus_musculus", ["ENSMUSG1", "ENSMUSG2"])

    loader = build_census_dataloader("mus_musculus", census=census, vocabulary=explicit)
    assert loader.vocabulary is explicit


def test_build_census_dataloader_closes_on_error(
    monkeypatch: pytest.MonkeyPatch, human_gene_ids: List[str]
) -> None:
    var_df = pd.DataFrame({"feature_id": human_gene_ids})
    query = _FakeQuery(var_df)
    census = _FakeCensus("homo_sapiens", _FakeExperiment(query))
    _patch_tiledbsoma_ml(monkeypatch, _FakeInner([]), raise_on_loader=True)

    with pytest.raises(RuntimeError, match="boom"):
        build_census_dataloader("homo_sapiens", census=census)
    # The query handle opened during setup is cleaned up; caller census is not.
    assert query.closed is True
    assert census.closed is False


@pytest.mark.skipif(
    not RUN_LIVE_CENSUS,
    reason="Live Census streaming test; set OQAE_RUN_CENSUS_TESTS=1 to run.",
)
@pytest.mark.parametrize("organism", ["homo_sapiens", "mus_musculus"])
def test_live_census_stream_small_slice(organism: str) -> None:  # pragma: no cover
    """Stream a tiny live Census slice through the shared Minibatch API."""
    loader = build_census_dataloader(
        organism,
        obs_value_filter="tissue_general == 'blood' and is_primary_data == True",
        batch_size=4,
        shuffle=False,
        batch_key="dataset_id",
    )
    with loader:
        batch = next(iter(loader))
    assert isinstance(batch, Minibatch)
    assert batch.counts.shape[1] == loader.vocabulary.n_genes
    assert batch.covariates["organism"][0] == organism
