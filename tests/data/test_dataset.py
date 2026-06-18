"""Tests for the gene vocabulary, alignment, and minibatch contract."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pytest
import torch
from scipy import sparse

from omvqvae.data.dataset import (
    SUPPORTED_ORGANISMS,
    CountsDataset,
    GeneVocabulary,
    Minibatch,
    align_to_reference,
    collate_minibatch,
)


def test_gene_vocabulary_basics() -> None:
    vocab = GeneVocabulary("homo_sapiens", ["A", "B", "C"])
    assert vocab.n_genes == 3
    assert len(vocab) == 3
    assert vocab.gene_ids == ["A", "B", "C"]
    assert vocab.position_of("B") == 1
    assert vocab.position_of("missing") is None
    assert "homo_sapiens" in SUPPORTED_ORGANISMS


def test_gene_vocabulary_rejects_empty_and_duplicates() -> None:
    with pytest.raises(ValueError, match="at least one"):
        GeneVocabulary("homo_sapiens", [])
    with pytest.raises(ValueError, match="Duplicate"):
        GeneVocabulary("homo_sapiens", ["A", "A"])


def test_gene_vocabulary_warns_on_unsupported_organism(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        GeneVocabulary("danio_rerio", ["A"])
    assert any("not in the v1-supported" in r.message for r in caplog.records)


def test_align_to_reference_zero_fill_drop_reorder() -> None:
    vocab = GeneVocabulary("homo_sapiens", ["A", "B", "C", "D"])
    # Source carries C, A, X (X is extra), in a different order.
    matrix = np.array([[10, 20, 99]], dtype=np.float32)
    aligned = align_to_reference(matrix, ["C", "A", "X"], vocab)
    # A->20 at col0, B->0, C->10 at col2, D->0; X dropped.
    np.testing.assert_array_equal(aligned[0], [20.0, 0.0, 10.0, 0.0])
    assert aligned.shape == (1, 4)
    assert aligned.dtype == np.float32


def test_align_to_reference_accepts_sparse() -> None:
    vocab = GeneVocabulary("homo_sapiens", ["A", "B"])
    matrix = sparse.csr_matrix(np.array([[5, 7]], dtype=np.float32))
    aligned = align_to_reference(matrix, ["A", "B"], vocab)
    np.testing.assert_array_equal(aligned[0], [5.0, 7.0])


def test_align_to_reference_low_overlap_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    vocab = GeneVocabulary("homo_sapiens", ["A", "B", "C", "D"])
    matrix = np.array([[1.0]], dtype=np.float32)
    with caplog.at_level(logging.WARNING):
        align_to_reference(matrix, ["A"], vocab, min_overlap=0.9)
    assert any("Low gene overlap" in r.message for r in caplog.records)


def test_align_to_reference_column_mismatch() -> None:
    vocab = GeneVocabulary("homo_sapiens", ["A", "B"])
    with pytest.raises(ValueError, match="columns"):
        align_to_reference(np.ones((1, 3), dtype=np.float32), ["A", "B"], vocab)


def test_counts_dataset_getitem_and_len() -> None:
    counts = np.array([[1, 2, 3], [4, 0, 0]], dtype=np.float32)
    ds = CountsDataset(counts, organism="mus_musculus", batch_ids=["x", "y"])
    assert len(ds) == 2
    sample = ds[0]
    assert isinstance(sample["counts"], torch.Tensor)
    assert sample["counts"].shape == (3,)
    assert pytest.approx(float(sample["size_factor"])) == 6.0
    assert sample["organism"] == "mus_musculus"
    assert sample["batch"] == "x"


def test_counts_dataset_default_batch_ids() -> None:
    ds = CountsDataset(np.ones((3, 2), dtype=np.float32), organism="homo_sapiens")
    assert ds[2]["batch"] is None


def test_counts_dataset_batch_length_mismatch() -> None:
    with pytest.raises(ValueError, match="batch_ids"):
        CountsDataset(
            np.ones((2, 2), dtype=np.float32),
            organism="homo_sapiens",
            batch_ids=["only_one"],
        )


def _samples() -> List[dict]:
    counts = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    ds = CountsDataset(counts, organism="homo_sapiens", batch_ids=["a", "b", "c"])
    return [ds[i] for i in range(len(ds))]


def test_collate_minibatch_shapes_and_covariates() -> None:
    batch = collate_minibatch(_samples())
    assert isinstance(batch, Minibatch)
    assert batch.counts.shape == (3, 2)
    assert batch.size_factors.shape == (3,)
    assert len(batch) == 3
    assert batch.covariates["organism"] == ["homo_sapiens"] * 3
    assert batch.covariates["batch"] == ["a", "b", "c"]
