"""Tests for :mod:`omvqvae.benchmark.clustering` (NMI / ARI / cell-type ASW).

The clustering metrics require the optional ``scib-metrics`` dependency, so the
substantive tests are guarded with :func:`pytest.importorskip` and skip cleanly
on an offline default install.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from omvqvae.benchmark.clustering import ClusteringMetrics, clustering_metrics

scib_metrics = pytest.importorskip("scib_metrics")


def _two_blobs(sep: float = 6.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated Gaussian blobs with matching labels."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 0.15, size=(40, 8))
    b = rng.normal(sep, 0.15, size=(40, 8))
    latent = np.vstack([a, b]).astype(np.float32)
    labels = np.array(["A"] * 40 + ["B"] * 40, dtype=object)
    return latent, labels


def test_clustering_metrics_recovers_clean_clusters() -> None:
    """Well-separated blocks matching their labels score near 1."""
    latent, labels = _two_blobs()
    metrics = clustering_metrics(latent, labels)
    assert isinstance(metrics, ClusteringMetrics)
    assert metrics.nmi > 0.9
    assert metrics.ari > 0.9
    assert metrics.cell_type_asw > 0.7


def test_clustering_metrics_low_for_shuffled_labels() -> None:
    """Labels unrelated to the geometry score near chance (low NMI/ARI)."""
    latent, labels = _two_blobs()
    rng = np.random.default_rng(1)
    shuffled = labels.copy()
    rng.shuffle(shuffled)
    metrics = clustering_metrics(latent, shuffled)
    assert metrics.nmi < 0.5
    assert metrics.ari < 0.5


def test_clustering_metrics_single_label_is_nan() -> None:
    """Fewer than two distinct labels leaves the metrics undefined (nan)."""
    latent, _ = _two_blobs()
    metrics = clustering_metrics(latent, ["only"] * latent.shape[0])
    assert math.isnan(metrics.nmi)
    assert math.isnan(metrics.ari)
    assert math.isnan(metrics.cell_type_asw)


def test_clustering_metrics_length_mismatch_raises() -> None:
    """A labels/cells length mismatch is a clear ValueError."""
    latent, labels = _two_blobs()
    with pytest.raises(ValueError, match="labels has length"):
        clustering_metrics(latent, labels[:-1])


def test_clustering_metrics_accepts_torch_tensor() -> None:
    """A torch latent is coerced without a hard torch import in the module."""
    torch = pytest.importorskip("torch")
    latent, labels = _two_blobs()
    metrics = clustering_metrics(torch.from_numpy(latent), labels)
    assert metrics.nmi > 0.9
