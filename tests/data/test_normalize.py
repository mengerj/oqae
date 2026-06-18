"""Tests for OQAE internal normalization / size-factor helpers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from omvqvae.data.normalize import (
    compute_size_factors,
    normalize_counts,
    to_dense,
)


def test_to_dense_from_sparse_and_dense() -> None:
    arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
    dense = to_dense(arr)
    assert dense.dtype == np.float32
    np.testing.assert_array_equal(dense, arr)

    spm = sparse.csr_matrix(arr)
    dense_sp = to_dense(spm)
    assert dense_sp.dtype == np.float32
    np.testing.assert_array_equal(dense_sp, arr)


def test_compute_size_factors_total() -> None:
    counts = np.array([[1, 2, 3], [0, 0, 0], [4, 0, 1]], dtype=np.float32)
    factors = compute_size_factors(counts, mode="total")
    # Empty cell gets 1.0 to stay finite.
    np.testing.assert_allclose(factors, [6.0, 1.0, 5.0])
    assert factors.dtype == np.float32


def test_compute_size_factors_ratio_default_median() -> None:
    counts = np.array([[10, 0], [0, 30], [20, 0]], dtype=np.float32)
    factors = compute_size_factors(counts, mode="ratio")
    # Totals are 10, 30, 20; median is 20 -> factors 0.5, 1.5, 1.0.
    np.testing.assert_allclose(factors, [0.5, 1.5, 1.0])


def test_compute_size_factors_ratio_explicit_target() -> None:
    counts = np.array([[5], [10]], dtype=np.float32)
    factors = compute_size_factors(counts, mode="ratio", target_sum=10.0)
    np.testing.assert_allclose(factors, [0.5, 1.0])


def test_compute_size_factors_sparse_matches_dense() -> None:
    counts = np.array([[1, 2, 0], [3, 0, 5]], dtype=np.float32)
    dense = compute_size_factors(counts, mode="total")
    spm = compute_size_factors(sparse.csr_matrix(counts), mode="total")
    np.testing.assert_array_equal(dense, spm)


def test_compute_size_factors_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        compute_size_factors(np.ones((2, 2), dtype=np.float32), mode="bogus")


def test_compute_size_factors_ratio_nonpositive_target() -> None:
    with pytest.raises(ValueError, match="positive"):
        compute_size_factors(
            np.ones((2, 2), dtype=np.float32), mode="ratio", target_sum=0.0
        )


def test_normalize_counts_log1p_and_target_sum() -> None:
    counts = np.array([[1, 1, 2], [0, 0, 0]], dtype=np.float32)
    out = normalize_counts(counts, target_sum=4.0, log1p=True)
    # Row 0 total 4 -> scaled to [1, 1, 2] -> log1p.
    np.testing.assert_allclose(out[0], np.log1p([1.0, 1.0, 2.0]), rtol=1e-6)
    # Empty row stays all-zero (safe division).
    np.testing.assert_array_equal(out[1], np.zeros(3, dtype=np.float32))
    assert out.dtype == np.float32


def test_normalize_counts_no_target_sum_only_log1p() -> None:
    counts = np.array([[3, 0]], dtype=np.float32)
    out = normalize_counts(counts, target_sum=None, log1p=True)
    np.testing.assert_allclose(out[0], np.log1p([3.0, 0.0]))


def test_normalize_counts_with_supplied_size_factors() -> None:
    counts = np.array([[2, 2]], dtype=np.float32)
    out = normalize_counts(
        counts, target_sum=2.0, log1p=False, size_factors=np.array([4.0])
    )
    np.testing.assert_allclose(out[0], [1.0, 1.0])
