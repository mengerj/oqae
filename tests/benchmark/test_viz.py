"""Tests for :mod:`omvqvae.benchmark.viz` (UMAP latent visualization).

Guarded with :func:`pytest.importorskip` on ``umap`` / ``matplotlib`` so they
skip cleanly when the optional ``benchmark`` extra is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("umap")
pytest.importorskip("matplotlib")

from omvqvae.benchmark.viz import compute_umap, plot_latent_umap  # noqa: E402


def _latent(n: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 0.2, size=(n // 2, 8))
    b = rng.normal(4.0, 0.2, size=(n // 2, 8))
    latent = np.vstack([a, b]).astype(np.float32)
    labels = np.array(["A"] * (n // 2) + ["B"] * (n // 2), dtype=object)
    return latent, labels


def test_compute_umap_shape() -> None:
    latent, _ = _latent()
    coords = compute_umap(latent, n_neighbors=5, random_state=0)
    assert coords.shape == (latent.shape[0], 2)


def test_plot_latent_umap_single_panel() -> None:
    latent, labels = _latent()
    fig = plot_latent_umap(latent, labels, n_neighbors=5)
    # One row (latent), one column (cell type).
    assert len(fig.axes) == 1
    # Scatter drew both label groups (one PathCollection each).
    assert len(fig.axes[0].collections) == 2


def test_plot_latent_umap_grid_with_batch_and_quantized() -> None:
    """labels + color_by (cols) × latent + quantized (rows) → a 2x2 grid."""
    latent, labels = _latent()
    quantized = latent + 0.01
    batch = np.array(["x", "y"] * (latent.shape[0] // 2), dtype=object)
    fig = plot_latent_umap(
        latent,
        labels,
        color_by=batch,
        quantized=quantized,
        n_neighbors=5,
    )
    assert len(fig.axes) == 4


def test_plot_latent_umap_length_mismatch_raises() -> None:
    latent, labels = _latent()
    with pytest.raises(ValueError, match="labels has length"):
        plot_latent_umap(latent, labels[:-1], n_neighbors=5)
