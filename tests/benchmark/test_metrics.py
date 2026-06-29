"""Unit tests for the pure benchmark metric functions."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from omvqvae.benchmark.metrics import (
    codebook_usage,
    reconstruction_metrics,
    separability_score,
)
from omvqvae.models.vqvae import OmicsVQVAE


def test_codebook_usage_uniform_is_full_perplexity() -> None:
    """Uniform code use gives perplexity == codebook_size and full utilization."""
    # Two levels, each cycling through all 4 entries uniformly.
    codes = np.tile(np.arange(4), (2, 1)).T  # shape (8, 2)? build explicitly below
    codes = np.array([[i % 4, i % 4] for i in range(8)], dtype=np.int64)
    usage = codebook_usage(codes, codebook_size=4)
    assert usage.codebook_size == 4
    assert usage.perplexity == pytest.approx(4.0)
    assert usage.utilization == pytest.approx(1.0)
    assert usage.perplexities == pytest.approx([4.0, 4.0])
    assert usage.utilizations == pytest.approx([1.0, 1.0])


def test_codebook_usage_collapse_is_minimal() -> None:
    """All cells using one entry gives perplexity 1 and tiny utilization."""
    codes = np.zeros((16, 3), dtype=np.int64)
    usage = codebook_usage(codes, codebook_size=8)
    assert usage.perplexity == pytest.approx(1.0)
    assert usage.utilization == pytest.approx(1.0 / 8.0)


def test_codebook_usage_accepts_torch_tensor() -> None:
    """Codes may be a torch tensor."""
    codes = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=torch.long)
    usage = codebook_usage(codes, codebook_size=2)
    assert usage.perplexity == pytest.approx(2.0)


def test_codebook_usage_empty_codes() -> None:
    """Zero cells yields zero perplexity / utilization, no error."""
    usage = codebook_usage(np.zeros((0, 2), dtype=np.int64), codebook_size=4)
    assert usage.perplexity == 0.0
    assert usage.utilization == 0.0
    assert usage.perplexities == [0.0, 0.0]


def test_codebook_usage_rejects_bad_inputs() -> None:
    """Bad codebook_size, shape, or out-of-range indices raise ValueError."""
    with pytest.raises(ValueError, match="codebook_size"):
        codebook_usage(np.zeros((2, 2), dtype=np.int64), codebook_size=0)
    with pytest.raises(ValueError, match="2-D"):
        codebook_usage(np.zeros(4, dtype=np.int64), codebook_size=4)
    with pytest.raises(ValueError, match="index >= codebook_size"):
        codebook_usage(np.array([[5]], dtype=np.int64), codebook_size=4)


def test_separability_perfect_clusters() -> None:
    """Well-separated label groups score 1.0."""
    latent = np.array(
        [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]], dtype=np.float32
    )
    labels = ["a", "a", "b", "b"]
    assert separability_score(latent, labels) == pytest.approx(1.0)


def test_separability_accepts_torch_and_returns_fraction() -> None:
    """A misassigned point lowers the score below 1; torch input works."""
    latent = torch.tensor([[0.0], [0.2], [5.0], [5.2], [0.3]], dtype=torch.float32)
    # The last point sits in cluster "a" but is labelled "b" -> misassigned.
    labels = ["a", "a", "b", "b", "b"]
    score = separability_score(latent, labels)
    assert score == pytest.approx(0.8)


def test_separability_single_label_is_nan() -> None:
    """Fewer than two labels makes separability undefined (nan)."""
    latent = np.zeros((4, 2), dtype=np.float32)
    assert math.isnan(separability_score(latent, ["a", "a", "a", "a"]))


def test_separability_empty_is_nan() -> None:
    """No cells -> nan."""
    assert math.isnan(separability_score(np.zeros((0, 2), dtype=np.float32), []))


def test_separability_validates_shapes() -> None:
    """Mismatched lengths and non-2D latents raise ValueError."""
    with pytest.raises(ValueError, match="length"):
        separability_score(np.zeros((3, 2), dtype=np.float32), ["a", "b"])
    with pytest.raises(ValueError, match="2-D"):
        separability_score(np.zeros(3, dtype=np.float32), ["a", "b", "c"])


def _toy_counts(seed: int = 0, n_cells: int = 24, n_genes: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float32)


def test_reconstruction_metrics_basic() -> None:
    """Returns finite NLL/MAE and respects eval mode (no codebook drift)."""
    counts = _toy_counts()
    model = OmicsVQVAE(
        n_genes=counts.shape[1], n_latent=4, hidden_dims=(8,), codebook_size=8
    )
    model.train()
    before = model.rvq.quantizers[0].embedding.clone()
    metrics = reconstruction_metrics(model, counts, batch_size=8)
    assert math.isfinite(metrics.nll)
    assert math.isfinite(metrics.mae)
    assert metrics.mae >= 0.0
    # The model is restored to train mode and its codebooks are untouched.
    assert model.training
    assert torch.equal(model.rvq.quantizers[0].embedding, before)


def test_reconstruction_metrics_gaussian_target_is_log1p() -> None:
    """The Gaussian head is evaluated in log1p space without error."""
    counts = _toy_counts(seed=1)
    model = OmicsVQVAE(
        n_genes=counts.shape[1],
        n_latent=4,
        hidden_dims=(8,),
        codebook_size=8,
        likelihood="gaussian",
    )
    metrics = reconstruction_metrics(model, counts)
    assert math.isfinite(metrics.mae)


def test_reconstruction_metrics_explicit_size_factors() -> None:
    """Explicit size factors of the wrong length raise; correct ones work."""
    counts = _toy_counts(seed=2)
    model = OmicsVQVAE(n_genes=counts.shape[1], n_latent=4, hidden_dims=(8,))
    factors = counts.sum(axis=1)
    metrics = reconstruction_metrics(model, counts, size_factors=factors)
    assert math.isfinite(metrics.nll)
    # A torch tensor of size factors is also accepted.
    metrics_t = reconstruction_metrics(
        model, counts, size_factors=torch.from_numpy(factors)
    )
    assert math.isfinite(metrics_t.nll)
    with pytest.raises(ValueError, match="length"):
        reconstruction_metrics(model, counts, size_factors=factors[:-1])


def test_reconstruction_metrics_validates_genes() -> None:
    """A gene-count mismatch raises ValueError."""
    counts = _toy_counts(seed=3, n_genes=6)
    model = OmicsVQVAE(n_genes=5, n_latent=4, hidden_dims=(8,))
    with pytest.raises(ValueError, match="genes"):
        reconstruction_metrics(model, counts)
    with pytest.raises(ValueError, match="2-D"):
        reconstruction_metrics(model, np.zeros(5, dtype=np.float32))


def test_reconstruction_metrics_empty_is_nan() -> None:
    """Zero cells -> nan metrics, no error."""
    model = OmicsVQVAE(n_genes=6, n_latent=4, hidden_dims=(8,))
    metrics = reconstruction_metrics(model, np.zeros((0, 6), dtype=np.float32))
    assert math.isnan(metrics.nll)
    assert math.isnan(metrics.mae)
