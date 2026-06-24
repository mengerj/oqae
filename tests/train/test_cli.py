"""Offline tests for the config-driven training CLI.

These exercise the config schema/loading, each ``config -> object`` builder, the
full ``run_experiment`` wiring against a synthetic local ``.h5ad``, and the Typer
command via ``CliRunner`` — all without network access or a real ``wandb`` run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData
from typer.testing import CliRunner

from omvqvae.models.vqvae import OmicsVQVAE
from omvqvae.train.cli import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrackingConfig,
    TrainingConfig,
    app,
    build_data,
    build_model,
    build_tracker_from_config,
    build_train_config,
    load_config,
    run_experiment,
)
from omvqvae.utils.tracking import ConsoleTracker, ExperimentTracker

GENE_IDS = ["ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5"]


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


def _make_h5ad(path: Path, *, n_cells: int = 24, seed: int = 0) -> None:
    """Write a tiny synthetic raw-count ``.h5ad`` to ``path``."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=3.0, size=(n_cells, len(GENE_IDS))).astype(np.float32)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    obs["batch"] = [f"b{i % 2}" for i in range(n_cells)]
    var = pd.DataFrame(index=list(GENE_IDS))
    AnnData(X=counts, obs=obs, var=var).write_h5ad(path)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def test_load_config_defaults() -> None:
    cfg = load_config()
    assert isinstance(cfg, ExperimentConfig)
    assert isinstance(cfg.model, ModelConfig)
    assert cfg.model.likelihood == "nb"
    assert cfg.data.source == "anndata"
    assert cfg.training.max_epochs == 1
    assert cfg.tracking.backend == "console"


def test_load_config_yaml_and_overrides(tmp_path: Path) -> None:
    yaml = tmp_path / "cfg.yaml"
    yaml.write_text(
        "model:\n  n_latent: 4\n  hidden_dims: [8]\n" "training:\n  max_epochs: 3\n"
    )
    cfg = load_config(yaml, overrides=["training.lr=0.05", "data.path=foo.h5ad"])
    assert cfg.model.n_latent == 4
    assert cfg.model.hidden_dims == [8]
    assert cfg.training.max_epochs == 3
    assert cfg.training.lr == pytest.approx(0.05)
    assert cfg.data.path == "foo.h5ad"


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_load_config_rejects_unknown_key(tmp_path: Path) -> None:
    yaml = tmp_path / "cfg.yaml"
    yaml.write_text("model:\n  not_a_field: 1\n")
    with pytest.raises(Exception):
        load_config(yaml)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def test_build_model() -> None:
    model = build_model(
        ModelConfig(n_latent=8, hidden_dims=[16], n_codebooks=3, codebook_size=32),
        n_genes=7,
    )
    assert isinstance(model, OmicsVQVAE)
    assert model.n_genes == 7
    assert model.n_latent == 8
    assert model.n_codebooks == 3


def test_build_model_n_genes_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="disagrees"):
        build_model(ModelConfig(n_genes=10), n_genes=7)


def test_build_train_config_maps_fields() -> None:
    tc = build_train_config(
        TrainingConfig(max_epochs=4, lr=0.01, grad_clip_norm=2.0, max_steps=9)
    )
    assert tc.max_epochs == 4
    assert tc.lr == pytest.approx(0.01)
    assert tc.grad_clip_norm == pytest.approx(2.0)
    assert tc.max_steps == 9


def test_build_tracker_console() -> None:
    tracker = build_tracker_from_config(
        TrackingConfig(backend="console", run_name="r"), run_config={"a": 1}
    )
    assert isinstance(tracker, ConsoleTracker)


def test_build_data_anndata(tmp_path: Path) -> None:
    path = tmp_path / "cells.h5ad"
    _make_h5ad(path)
    loader, vocab = build_data(DataConfig(source="anndata", path=str(path)))
    assert vocab.n_genes == len(GENE_IDS)
    assert vocab.organism == "homo_sapiens"
    batch = next(iter(loader))
    assert batch.counts.shape[1] == len(GENE_IDS)


def test_build_data_anndata_requires_path() -> None:
    with pytest.raises(ValueError, match="data.path is required"):
        build_data(DataConfig(source="anndata", path=None))


def test_build_data_unknown_source() -> None:
    with pytest.raises(ValueError, match="Unknown data source"):
        build_data(DataConfig(source="bogus"))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _toy_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        seed=0,
        model=ModelConfig(
            n_latent=8, hidden_dims=[16], codebook_size=16, n_codebooks=2
        ),
        data=DataConfig(source="anndata", path=str(path), batch_size=8, shuffle=False),
        training=TrainingConfig(max_epochs=3, lr=0.01, log_every=1),
        tracking=TrackingConfig(backend="console"),
    )


def test_run_experiment_trains_and_logs(tmp_path: Path) -> None:
    path = tmp_path / "cells.h5ad"
    _make_h5ad(path)
    cfg = _toy_config(path)
    tracker = _RecordingTracker()

    result, model = run_experiment(cfg, tracker=tracker)

    assert isinstance(model, OmicsVQVAE)
    assert result.global_step > 0
    assert len(result.epochs) == 3
    # Injected tracker lifecycle is owned by the caller, so it is not finished.
    assert tracker.finished is False
    assert tracker.logs, "expected per-step metrics to be logged"
    # Reconstruction loss should not increase from the first to the last epoch.
    assert result.epochs[-1].reconstruction_loss <= result.epochs[0].reconstruction_loss


def test_run_experiment_writes_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "cells.h5ad"
    _make_h5ad(path)
    ckpt = tmp_path / "out" / "model.pt"
    cfg = _toy_config(path)
    cfg.training.checkpoint_path = str(ckpt)

    run_experiment(cfg, tracker=_RecordingTracker())

    assert ckpt.exists()
    payload = torch.load(ckpt, weights_only=False)
    assert payload["organism"] == "homo_sapiens"
    assert payload["gene_ids"] == GENE_IDS
    assert "state_dict" in payload


def test_run_experiment_builds_default_tracker(tmp_path: Path) -> None:
    path = tmp_path / "cells.h5ad"
    _make_h5ad(path)
    cfg = _toy_config(path)
    cfg.training.max_steps = 2
    # No tracker injected: run_experiment builds and finishes its own.
    result, _ = run_experiment(cfg)
    assert result.global_step > 0


def test_run_experiment_seed_none_runs(tmp_path: Path) -> None:
    path = tmp_path / "cells.h5ad"
    _make_h5ad(path)
    cfg = _toy_config(path)
    cfg.seed = None
    cfg.training.max_steps = 2
    result, _ = run_experiment(cfg, tracker=_RecordingTracker())
    assert result.global_step > 0


def test_run_experiment_seed_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "cells.h5ad"
    _make_h5ad(path)
    cfg = _toy_config(path)
    cfg.training.max_steps = 3

    _, model_a = run_experiment(cfg, tracker=_RecordingTracker())
    _, model_b = run_experiment(cfg, tracker=_RecordingTracker())

    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(pa, pb)


# --------------------------------------------------------------------------- #
# Typer CLI
# --------------------------------------------------------------------------- #
def test_cli_train_end_to_end(tmp_path: Path) -> None:
    path = tmp_path / "cells.h5ad"
    _make_h5ad(path)
    yaml = tmp_path / "cfg.yaml"
    yaml.write_text(
        "model:\n  n_latent: 8\n  hidden_dims: [16]\n  codebook_size: 16\n"
        f"data:\n  source: anndata\n  path: {path}\n  batch_size: 8\n"
        "training:\n  max_epochs: 2\n  log_every: 1\n"
        "tracking:\n  backend: none\n"
    )
    runner = CliRunner()
    result = runner.invoke(app, [str(yaml)])
    assert result.exit_code == 0, result.output
    assert "Done:" in result.output
    assert "final loss" in result.output


def test_cli_train_with_overrides_and_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "cells.h5ad"
    _make_h5ad(path)
    ckpt = tmp_path / "model.pt"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--set",
            "data.source=anndata",
            "--set",
            f"data.path={path}",
            "--set",
            "data.batch_size=8",
            "--set",
            "model.n_latent=8",
            "--set",
            "model.hidden_dims=[16]",
            "--set",
            "model.codebook_size=16",
            "--set",
            "training.max_steps=2",
            "--set",
            "tracking.backend=none",
            "--set",
            f"training.checkpoint_path={ckpt}",
        ],
    )
    assert result.exit_code == 0, result.output
    assert ckpt.exists()
