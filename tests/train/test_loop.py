"""Offline tests for the source-agnostic training loop.

Train a tiny :class:`OmicsVQVAE` on a synthetic in-memory ``CountsDataset`` via
a real ``DataLoader`` and a recording fake tracker; assert the loop reduces the
reconstruction loss, respects ``max_steps``, clips gradients, and logs metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from omvqvae.data.dataset import CountsDataset, Minibatch, collate_minibatch
from omvqvae.models.vqvae import OmicsVQVAE
from omvqvae.train.loop import EpochMetrics, TrainConfig, TrainResult, train
from omvqvae.utils.tracking import ExperimentTracker


class _RecordingTracker(ExperimentTracker):
    """A tracker that records everything passed to it (no I/O)."""

    def __init__(self) -> None:
        self.configs: List[Dict[str, Any]] = []
        self.logs: List[Tuple[Dict[str, float], Optional[int]]] = []
        self.finished = False

    def log_config(self, config: Mapping[str, Any]) -> None:
        self.configs.append(dict(config))

    def log(self, metrics: Mapping[str, float], *, step: Optional[int] = None) -> None:
        self.logs.append((dict(metrics), step))

    def finish(self) -> None:
        self.finished = True


def _loader(n_cells: int = 32, n_genes: int = 12, batch_size: int = 8) -> DataLoader:
    rng = np.random.default_rng(0)
    means = rng.gamma(shape=2.0, scale=1.0, size=(2, n_genes))
    assignments = rng.integers(0, 2, size=n_cells)
    counts = rng.poisson(means[assignments]).astype(np.float32)
    dataset = CountsDataset(counts, organism="homo_sapiens")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_minibatch,
    )


def _model(n_genes: int = 12) -> OmicsVQVAE:
    torch.manual_seed(0)
    return OmicsVQVAE(
        n_genes, n_latent=8, hidden_dims=(16,), codebook_size=16, n_codebooks=2
    )


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #
def test_train_config_validation() -> None:
    with pytest.raises(ValueError):
        TrainConfig(max_epochs=0)
    with pytest.raises(ValueError):
        TrainConfig(max_steps=0)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_train_returns_result_and_logs() -> None:
    model = _model()
    loader = _loader()
    tracker = _RecordingTracker()
    config = TrainConfig(max_epochs=2, lr=1e-2, log_every=1)

    result = train(model, loader, config=config, tracker=tracker)

    assert isinstance(result, TrainResult)
    assert len(result.epochs) == 2
    assert all(isinstance(e, EpochMetrics) for e in result.epochs)
    # 32 cells / batch 8 = 4 steps per epoch, two epochs.
    assert result.epochs[0].steps == 4
    assert result.global_step == 8
    assert result.last_epoch is result.epochs[-1]
    # Per-step train metrics plus a per-epoch summary were logged.
    assert any("train/loss" in m for m, _ in tracker.logs)
    assert any("epoch/loss" in m for m, _ in tracker.logs)


def test_train_reduces_reconstruction_loss() -> None:
    model = _model()
    loader = _loader()
    config = TrainConfig(max_epochs=8, lr=1e-2, log_every=0)

    result = train(model, loader, config=config)

    first = result.epochs[0].reconstruction_loss
    last = result.epochs[-1].reconstruction_loss
    assert last < first


def test_train_defaults_to_console_tracker() -> None:
    # No tracker / config passed: must still run (one epoch, ConsoleTracker).
    result = train(_model(), _loader())
    assert result.global_step == 4
    assert len(result.epochs) == 1


# --------------------------------------------------------------------------- #
# max_steps / grad clipping
# --------------------------------------------------------------------------- #
def test_train_respects_max_steps() -> None:
    config = TrainConfig(max_epochs=5, max_steps=3, log_every=0)
    result = train(_model(), _loader(), config=config)
    assert result.global_step == 3
    # Stopped mid-first-epoch, so only one (partial) epoch recorded.
    assert len(result.epochs) == 1
    assert result.epochs[0].steps == 3


def test_train_grad_clip_runs() -> None:
    config = TrainConfig(max_epochs=1, grad_clip_norm=1.0, log_every=0)
    result = train(_model(), _loader(), config=config)
    assert result.global_step == 4


# --------------------------------------------------------------------------- #
# Injected optimizer
# --------------------------------------------------------------------------- #
def test_train_uses_injected_optimizer() -> None:
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    config = TrainConfig(max_epochs=1, log_every=0)

    result = train(model, _loader(), config=config, optimizer=optimizer)
    assert result.global_step == 4


# --------------------------------------------------------------------------- #
# Device move
# --------------------------------------------------------------------------- #
def test_train_moves_batch_to_device() -> None:
    # CPU device path exercises _move_batch without a GPU.
    model = _model()
    batch_list: List[Minibatch] = list(_loader())
    config = TrainConfig(max_epochs=1, device="cpu", log_every=0)
    result = train(model, batch_list, config=config)
    assert result.global_step == len(batch_list)
