"""Tests for the CELLxGENE Census streaming loader.

The chunk-to-:class:`Minibatch` glue and the :class:`CensusMinibatchLoader`
adapter are tested fully offline with synthetic ``(X, obs)`` fixtures. A single
live-Census streaming test is marked ``network`` and skipped by default.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from omvqvae.data.census import (
    DEFAULT_CENSUS_VERSION,
    CensusMinibatchLoader,
    build_census_dataloader,
    census_chunk_to_minibatch,
    census_experiment_name,
    open_census,
)
from omvqvae.data.dataset import GeneVocabulary, Minibatch


def _chunk(
    gene_ids: List[str], *, n_cells: int = 4, seed: int = 0, batch: bool = True
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Build one synthetic Census-style ``(counts, obs)`` chunk."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=2.0, size=(n_cells, len(gene_ids))).astype(np.float32)
    obs = pd.DataFrame({"soma_joinid": np.arange(n_cells)})
    if batch:
        obs["dataset_id"] = [f"d{i % 2}" for i in range(n_cells)]
    return counts, obs


def test_census_experiment_name_supported() -> None:
    assert census_experiment_name("homo_sapiens") == "homo_sapiens"
    assert census_experiment_name("mus_musculus") == "mus_musculus"


def test_census_experiment_name_rejects_unsupported() -> None:
    with pytest.raises(ValueError, match="Unsupported organism"):
        census_experiment_name("danio_rerio")


def test_chunk_to_minibatch_contract(human_gene_ids: List[str]) -> None:
    counts, obs = _chunk(human_gene_ids, n_cells=4)
    vocab = GeneVocabulary("homo_sapiens", human_gene_ids)
    batch = census_chunk_to_minibatch(
        counts,
        obs,
        human_gene_ids,
        vocab,
        organism="homo_sapiens",
        batch_key="dataset_id",
    )
    assert isinstance(batch, Minibatch)
    assert batch.counts.shape == (4, vocab.n_genes)
    assert batch.size_factors.shape == (4,)
    assert batch.covariates["organism"] == ["homo_sapiens"] * 4
    assert set(batch.covariates["batch"]) <= {"d0", "d1"}
    # Size factor equals the per-cell total of the (here identity-aligned) counts.
    np.testing.assert_allclose(
        batch.size_factors.numpy(), counts.sum(axis=1), rtol=1e-5
    )


def test_chunk_to_minibatch_aligns_and_zero_fills(human_gene_ids: List[str]) -> None:
    # Reference adds a gene absent from the chunk -> zero-filled column.
    reference = human_gene_ids + ["ENSG_EXTRA"]
    counts, obs = _chunk(human_gene_ids, n_cells=3)
    vocab = GeneVocabulary("homo_sapiens", reference)
    batch = census_chunk_to_minibatch(
        counts, obs, human_gene_ids, vocab, organism="homo_sapiens"
    )
    assert batch.counts.shape == (3, len(reference))
    assert float(batch.counts[:, -1].abs().sum()) == 0.0


def test_chunk_to_minibatch_without_batch_key(human_gene_ids: List[str]) -> None:
    counts, obs = _chunk(human_gene_ids, n_cells=2, batch=False)
    vocab = GeneVocabulary("homo_sapiens", human_gene_ids)
    batch = census_chunk_to_minibatch(
        counts, obs, human_gene_ids, vocab, organism="homo_sapiens"
    )
    assert batch.covariates["batch"] == [None, None]


def test_chunk_to_minibatch_missing_batch_key_is_none(
    human_gene_ids: List[str],
) -> None:
    # batch_key requested but not present in obs -> None covariate, no error.
    counts, obs = _chunk(human_gene_ids, n_cells=2, batch=False)
    vocab = GeneVocabulary("homo_sapiens", human_gene_ids)
    batch = census_chunk_to_minibatch(
        counts, obs, human_gene_ids, vocab, organism="homo_sapiens", batch_key="absent"
    )
    assert batch.covariates["batch"] == [None, None]


def test_chunk_to_minibatch_handles_1d_single_cell(human_gene_ids: List[str]) -> None:
    counts = np.array([1.0, 0.0, 3.0, 2.0], dtype=np.float32)
    obs = pd.DataFrame({"soma_joinid": [0]})
    vocab = GeneVocabulary("homo_sapiens", human_gene_ids)
    batch = census_chunk_to_minibatch(
        counts, obs, human_gene_ids, vocab, organism="homo_sapiens"
    )
    assert batch.counts.shape == (1, len(human_gene_ids))


def test_loader_iterates_minibatches(human_gene_ids: List[str]) -> None:
    chunks = [
        _chunk(human_gene_ids, n_cells=3, seed=0),
        _chunk(human_gene_ids, n_cells=2, seed=1),
    ]
    vocab = GeneVocabulary("homo_sapiens", human_gene_ids)
    loader = CensusMinibatchLoader(
        chunks,
        source_gene_ids=human_gene_ids,
        vocabulary=vocab,
        organism="homo_sapiens",
        batch_key="dataset_id",
    )
    assert loader.vocabulary is vocab
    batches = list(loader)
    assert all(isinstance(b, Minibatch) for b in batches)
    assert sum(len(b) for b in batches) == 5
    assert batches[0].counts.shape[1] == vocab.n_genes
    # Iterable is re-iterable (a fresh pass over the source).
    assert sum(len(b) for b in loader) == 5


def test_loader_close_closes_query(human_gene_ids: List[str]) -> None:
    class _FakeQuery:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    query = _FakeQuery()
    vocab = GeneVocabulary("homo_sapiens", human_gene_ids)
    loader = CensusMinibatchLoader(
        [],
        source_gene_ids=human_gene_ids,
        vocabulary=vocab,
        organism="homo_sapiens",
        query=query,
    )
    loader.close()
    assert query.closed is True


def test_loader_close_without_query_is_noop(human_gene_ids: List[str]) -> None:
    vocab = GeneVocabulary("homo_sapiens", human_gene_ids)
    loader = CensusMinibatchLoader(
        [],
        source_gene_ids=human_gene_ids,
        vocabulary=vocab,
        organism="homo_sapiens",
    )
    loader.close()  # should not raise


def test_human_and_mouse_share_same_api(
    human_gene_ids: List[str], mouse_gene_ids: List[str]
) -> None:
    """Human and mouse stream through the same Census loader API."""
    for organism, genes in [
        ("homo_sapiens", human_gene_ids),
        ("mus_musculus", mouse_gene_ids),
    ]:
        chunks = [_chunk(genes, n_cells=3)]
        vocab = GeneVocabulary(organism, genes)
        loader = CensusMinibatchLoader(
            chunks,
            source_gene_ids=genes,
            vocabulary=vocab,
            organism=organism,
        )
        batch = next(iter(loader))
        assert isinstance(batch, Minibatch)
        assert batch.counts.shape[1] == len(genes)
        assert batch.covariates["organism"][0] == organism


def test_default_census_version_is_pinned() -> None:
    # A concrete dated release, not a moving alias, for reproducibility.
    assert DEFAULT_CENSUS_VERSION[:2] == "20"
    assert DEFAULT_CENSUS_VERSION not in {"stable", "latest"}


@pytest.mark.network
@pytest.mark.parametrize("organism", ["homo_sapiens", "mus_musculus"])
def test_live_census_streaming(organism: str) -> None:
    """Stream a tiny live Census slice (human and mouse) end to end.

    Network-gated: run with ``pytest -m network``.
    """
    pytest.importorskip("cellxgene_census")
    obs_filter = "is_primary_data == True and tissue_general == 'blood'"
    with open_census() as census:
        loader = build_census_dataloader(
            census,
            organism,
            obs_value_filter=obs_filter,
            batch_size=8,
            shuffle=False,
            batch_key="dataset_id",
        )
        try:
            batch = next(iter(loader))
        finally:
            loader.close()
    assert isinstance(batch, Minibatch)
    assert batch.counts.shape[0] > 0
    assert batch.counts.shape[1] == loader.vocabulary.n_genes
    assert batch.covariates["organism"][0] == organism
