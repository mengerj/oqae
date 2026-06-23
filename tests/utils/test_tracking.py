"""Offline tests for the experiment-tracking wrappers.

Cover the metric flattening, the console backend, the factory dispatch, and the
W&B backend logic via an injected fake run (no real ``wandb`` import).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch

from omvqvae.models.vqvae import OmicsVQVAE
from omvqvae.utils.tracking import (
    ConsoleTracker,
    ExperimentTracker,
    WandbTracker,
    build_tracker,
    vqvae_metrics,
)


def _output() -> Any:
    """A real VQVAEOutput from a tiny synthetic forward pass."""
    torch.manual_seed(0)
    model = OmicsVQVAE(8, n_latent=4, hidden_dims=(8,), codebook_size=8, n_codebooks=2)
    counts = torch.randint(0, 5, (3, 8)).float()
    size_factors = counts.sum(dim=1).clamp(min=1.0)
    return model(counts, size_factors)


# --------------------------------------------------------------------------- #
# vqvae_metrics
# --------------------------------------------------------------------------- #
def test_vqvae_metrics_keys_and_floats() -> None:
    metrics = vqvae_metrics(_output(), prefix="train")
    assert metrics["train/loss"] == pytest.approx(
        metrics["train/reconstruction_loss"] + metrics["train/vq_loss"]
    )
    # Per-codebook series are expanded (n_codebooks == 2).
    assert "train/perplexity/codebook_0" in metrics
    assert "train/perplexity/codebook_1" in metrics
    assert "train/usage/codebook_0" in metrics
    assert all(isinstance(v, float) for v in metrics.values())


def test_vqvae_metrics_empty_prefix() -> None:
    metrics = vqvae_metrics(_output(), prefix="")
    assert "loss" in metrics
    assert not any(k.startswith("/") for k in metrics)


# --------------------------------------------------------------------------- #
# ConsoleTracker
# --------------------------------------------------------------------------- #
def test_console_tracker_logs(caplog: pytest.LogCaptureFixture) -> None:
    tracker = ConsoleTracker(run_name="toy")
    with caplog.at_level(logging.INFO, logger="omvqvae.utils.tracking"):
        tracker.log_config({"lr": 0.01})
        tracker.log({"train/loss": 1.2345}, step=5)
        tracker.log({"train/loss": 1.0})  # no explicit step
    text = caplog.text
    assert "toy" in text
    assert "lr" in text
    assert "step 5" in text
    assert "train/loss=1.234" in text


def test_console_tracker_context_manager_finishes() -> None:
    with ConsoleTracker() as tracker:
        assert isinstance(tracker, ExperimentTracker)
    assert tracker._finished is True


# --------------------------------------------------------------------------- #
# build_tracker
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", ["console", "none", "CONSOLE"])
def test_build_tracker_console_variants(backend: str) -> None:
    tracker = build_tracker(backend, run_name="r")
    assert isinstance(tracker, ConsoleTracker)


def test_build_tracker_logs_config(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="omvqvae.utils.tracking"):
        build_tracker("console", config={"n_latent": 8})
    assert "n_latent" in caplog.text


def test_build_tracker_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown tracking backend"):
        build_tracker("tensorboard")


def test_build_tracker_wandb_uses_init(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    class _FakeRun:
        def __init__(self) -> None:
            self.config = _FakeConfig()
            self.logged: List[Tuple[Dict[str, float], Optional[int]]] = []
            self.finished = False

        def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
            self.logged.append((metrics, step))

        def finish(self) -> None:
            self.finished = True

    class _FakeConfig:
        def __init__(self) -> None:
            self.values: Dict[str, Any] = {}

        def update(self, values: Dict[str, Any], allow_val_change: bool) -> None:
            self.values.update(values)

    fake_run = _FakeRun()

    def _fake_init(**kwargs: Any) -> _FakeRun:
        captured.update(kwargs)
        return fake_run

    monkeypatch.setattr(
        "omvqvae.utils.tracking._init_wandb_run", lambda **kw: _fake_init(**kw)
    )
    tracker = build_tracker(
        "wandb", run_name="exp", project="oqae", config={"lr": 0.1}, offline=True
    )
    assert isinstance(tracker, WandbTracker)
    # Config logged through the run on construction.
    assert fake_run.config.values == {"lr": 0.1}
    assert captured["offline"] is True


# --------------------------------------------------------------------------- #
# WandbTracker (injected fake run)
# --------------------------------------------------------------------------- #
class _FakeRun:
    def __init__(self) -> None:
        self.logged: List[Tuple[Dict[str, float], Optional[int]]] = []
        self.config_updates: List[Dict[str, Any]] = []
        self.finish_calls = 0

    class _Config:
        def __init__(self, outer: "_FakeRun") -> None:
            self._outer = outer

        def update(self, values: Dict[str, Any], allow_val_change: bool) -> None:
            self._outer.config_updates.append(values)

    @property
    def config(self) -> "_FakeRun._Config":
        return _FakeRun._Config(self)

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        self.logged.append((metrics, step))

    def finish(self) -> None:
        self.finish_calls += 1


def test_wandb_tracker_delegates_to_run() -> None:
    run = _FakeRun()
    tracker = WandbTracker(run)
    tracker.log_config({"a": 1})
    tracker.log({"loss": 0.5}, step=3)
    tracker.finish()
    tracker.finish()  # idempotent: finish only fires once

    assert run.config_updates == [{"a": 1}]
    assert run.logged == [({"loss": 0.5}, 3)]
    assert run.finish_calls == 1
