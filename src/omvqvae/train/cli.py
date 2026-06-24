"""
Config-driven command-line interface for training / fine-tuning OQAE models.

This module turns a declarative configuration (YAML, parsed with OmegaConf) into
the concrete objects the training loop needs — an
:class:`~omvqvae.models.vqvae.OmicsVQVAE`, a data source yielding
:class:`~omvqvae.data.dataset.Minibatch` batches, and an
:class:`~omvqvae.utils.tracking.ExperimentTracker` — and then runs
:func:`omvqvae.train.loop.train`.

Design follows the project's "pure core + thin I/O shell" convention:

- The **config schema** is a set of plain dataclasses (:class:`ExperimentConfig`
  and friends) so OmegaConf can validate/merge a user YAML against typed
  defaults, and so the wiring is fully typed for mypy.
- The **builders** (:func:`build_model`, :func:`build_train_config`,
  :func:`build_tracker`, :func:`build_data`) are small pure functions mapping a
  config section to one object; they are unit-testable offline.
- The networked piece — streaming from the CELLxGENE Census — is isolated in one
  branch of :func:`build_data` and marked ``# pragma: no cover``; the local
  AnnData path is exercised end-to-end offline (including via Typer's
  ``CliRunner``).

A console entry point ``oqae-train`` is exposed through :func:`main`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, cast

from omegaconf import OmegaConf

from omvqvae.train.loop import TrainConfig
from omvqvae.train.loop import train as run_train
from omvqvae.utils.logging import get_logger
from omvqvae.utils.tracking import ExperimentTracker, build_tracker

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omvqvae.data.dataset import GeneVocabulary, Minibatch
    from omvqvae.models.vqvae import OmicsVQVAE
    from omvqvae.train.loop import TrainResult

logger = get_logger(__name__)

__all__ = [
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "TrackingConfig",
    "ExperimentConfig",
    "load_config",
    "build_model",
    "build_train_config",
    "build_tracker_from_config",
    "build_data",
    "run_experiment",
    "app",
    "main",
]


# --------------------------------------------------------------------------- #
# Config schema
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """:class:`~omvqvae.models.vqvae.OmicsVQVAE` hyper-parameters.

    ``n_genes`` is left unset (``None``) here because the feature-space size is
    determined by the data source's :class:`GeneVocabulary`; set it explicitly
    only to assert/override the expected gene-space size.
    """

    n_genes: Optional[int] = None
    n_latent: int = 16
    hidden_dims: List[int] = field(default_factory=lambda: [128])
    likelihood: str = "nb"
    codebook_size: int = 256
    n_codebooks: int = 2
    commitment_cost: float = 0.25
    ema: bool = True
    ema_decay: float = 0.99
    reset_dead_codes: bool = True
    dropout: float = 0.0


@dataclass
class DataConfig:
    """Data-source configuration.

    ``source`` selects between a local AnnData file (``"anndata"``) and CELLxGENE
    Census streaming (``"census"``). Fields not relevant to the chosen source are
    ignored.
    """

    source: str = "anndata"
    organism: str = "homo_sapiens"
    # Local AnnData (`source == "anndata"`).
    path: Optional[str] = None
    layer: Optional[str] = None
    var_key: Optional[str] = None
    # Census streaming (`source == "census"`).
    census_version: Optional[str] = None
    obs_value_filter: Optional[str] = None
    var_value_filter: Optional[str] = None
    # Shared loader knobs.
    batch_size: int = 128
    shuffle: bool = False
    num_workers: int = 0
    batch_key: Optional[str] = None
    size_factor_mode: str = "total"
    min_overlap: float = 0.0


@dataclass
class TrainingConfig:
    """Training-loop knobs (mirror of :class:`omvqvae.train.loop.TrainConfig`)."""

    max_epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = None
    max_steps: Optional[int] = None
    log_every: int = 10
    device: str = "cpu"
    checkpoint_path: Optional[str] = None


@dataclass
class TrackingConfig:
    """Experiment-tracking configuration."""

    backend: str = "console"
    run_name: Optional[str] = None
    project: Optional[str] = None
    offline: bool = False


@dataclass
class ExperimentConfig:
    """Top-level training configuration."""

    seed: Optional[int] = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def load_config(
    path: Optional[Path] = None,
    *,
    overrides: Optional[List[str]] = None,
) -> ExperimentConfig:
    """Load and validate an :class:`ExperimentConfig`.

    The typed defaults are merged with the user's YAML (if any) and then with
    dot-list overrides (e.g. ``["training.lr=1e-2", "model.n_latent=8"]``), so
    every value is schema-checked and unknown keys are rejected.

    Parameters
    ----------
    path : pathlib.Path, optional
        Path to a YAML config file. When ``None``, only defaults + ``overrides``
        are used.
    overrides : List[str], optional
        OmegaConf dot-list overrides applied last (highest precedence).

    Returns
    -------
    ExperimentConfig
        The fully-resolved, typed configuration.

    Raises
    ------
    FileNotFoundError
        If ``path`` is given but does not exist.
    """
    schema = OmegaConf.structured(ExperimentConfig)
    merged = schema
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file does not exist: {p}")
        merged = OmegaConf.merge(merged, OmegaConf.load(p))
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))
    return cast(ExperimentConfig, OmegaConf.to_object(merged))


# --------------------------------------------------------------------------- #
# Builders (config section -> object)
# --------------------------------------------------------------------------- #
def build_model(config: ModelConfig, *, n_genes: int) -> "OmicsVQVAE":
    """Construct an :class:`~omvqvae.models.vqvae.OmicsVQVAE` from config.

    Parameters
    ----------
    config : ModelConfig
        Model hyper-parameters.
    n_genes : int
        Feature-space size (from the data source's :class:`GeneVocabulary`).

    Returns
    -------
    OmicsVQVAE
        The constructed model.

    Raises
    ------
    ValueError
        If ``config.n_genes`` is set and disagrees with ``n_genes``.
    """
    from omvqvae.models.vqvae import OmicsVQVAE

    if config.n_genes is not None and config.n_genes != n_genes:
        raise ValueError(
            f"model.n_genes={config.n_genes} disagrees with the data source's "
            f"gene-space size ({n_genes})."
        )
    return OmicsVQVAE(
        n_genes,
        n_latent=config.n_latent,
        hidden_dims=tuple(config.hidden_dims),
        likelihood=config.likelihood,
        codebook_size=config.codebook_size,
        n_codebooks=config.n_codebooks,
        commitment_cost=config.commitment_cost,
        ema=config.ema,
        ema_decay=config.ema_decay,
        reset_dead_codes=config.reset_dead_codes,
        dropout=config.dropout,
    )


def build_train_config(config: TrainingConfig) -> TrainConfig:
    """Map a :class:`TrainingConfig` onto the loop's :class:`TrainConfig`."""
    return TrainConfig(
        max_epochs=config.max_epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        grad_clip_norm=config.grad_clip_norm,
        max_steps=config.max_steps,
        log_every=config.log_every,
        device=config.device,
    )


def build_tracker_from_config(
    config: TrackingConfig,
    *,
    run_config: Optional[Dict[str, Any]] = None,
) -> ExperimentTracker:
    """Construct an :class:`ExperimentTracker` from a :class:`TrackingConfig`.

    Parameters
    ----------
    config : TrackingConfig
        Backend selection and run metadata.
    run_config : Dict[str, Any], optional
        Flattened run configuration logged to the tracker for reproducibility.

    Returns
    -------
    ExperimentTracker
        The constructed tracker (already given ``run_config`` to log).
    """
    return build_tracker(
        config.backend,
        run_name=config.run_name,
        project=config.project,
        config=run_config,
        offline=config.offline,
    )


def build_data(config: DataConfig) -> Tuple[Iterable["Minibatch"], "GeneVocabulary"]:
    """Build a data loader and its :class:`GeneVocabulary` from config.

    For the local AnnData source the reference vocabulary is derived from the
    file's own genes; for the Census source it is read from the Census ``var``
    index. The returned loader yields :class:`Minibatch` batches in both cases.

    Parameters
    ----------
    config : DataConfig
        Data-source configuration.

    Returns
    -------
    tuple of (Iterable[Minibatch], GeneVocabulary)
        The minibatch loader and the reference vocabulary (its ``n_genes`` sizes
        the model).

    Raises
    ------
    ValueError
        If ``config.source`` is unknown, or required fields are missing.
    """
    source = config.source.lower()
    if source == "anndata":
        return _build_anndata_data(config)
    if source == "census":  # pragma: no cover - requires live Census/TileDB-SOMA
        return _build_census_data(config)
    raise ValueError(
        f"Unknown data source {config.source!r}; expected 'anndata' or 'census'."
    )


def _build_anndata_data(
    config: DataConfig,
) -> Tuple[Iterable["Minibatch"], "GeneVocabulary"]:
    """Build a local-AnnData loader + vocabulary (reference = the file's genes)."""
    from omvqvae.data.anndata_io import (
        build_anndata_dataloader,
        extract_counts,
        load_anndata,
    )
    from omvqvae.data.dataset import GeneVocabulary

    if config.path is None:
        raise ValueError("data.path is required when data.source == 'anndata'.")
    adata = load_anndata(config.path)
    _, gene_ids = extract_counts(adata, layer=config.layer, var_key=config.var_key)
    vocabulary = GeneVocabulary(config.organism, gene_ids)
    loader = build_anndata_dataloader(
        adata,
        vocabulary,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        layer=config.layer,
        var_key=config.var_key,
        batch_key=config.batch_key,
        min_overlap=config.min_overlap,
        size_factor_mode=config.size_factor_mode,
    )
    return loader, vocabulary


def _build_census_data(  # pragma: no cover - requires live Census/TileDB-SOMA
    config: DataConfig,
) -> Tuple[Iterable["Minibatch"], "GeneVocabulary"]:
    """Build a Census streaming loader + vocabulary (the networked source)."""
    from omvqvae.data.census import (
        DEFAULT_CENSUS_VERSION,
        build_census_dataloader,
        census_gene_vocabulary,
        open_census,
    )

    census = open_census(census_version=config.census_version or DEFAULT_CENSUS_VERSION)
    vocabulary = census_gene_vocabulary(
        census,
        config.organism,
        var_value_filter=config.var_value_filter,
    )
    loader = build_census_dataloader(
        census,
        config.organism,
        vocabulary,
        obs_value_filter=config.obs_value_filter,
        var_value_filter=config.var_value_filter,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        batch_key=config.batch_key,
        size_factor_mode=config.size_factor_mode,
        min_overlap=config.min_overlap,
    )
    return loader, vocabulary


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _set_seed(seed: Optional[int]) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible runs."""
    if seed is None:
        return
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(
    config: ExperimentConfig,
    *,
    tracker: Optional[ExperimentTracker] = None,
    data: Optional[Tuple[Iterable["Minibatch"], "GeneVocabulary"]] = None,
) -> Tuple["TrainResult", "OmicsVQVAE"]:
    """Run a full training experiment from a resolved config.

    Parameters
    ----------
    config : ExperimentConfig
        The resolved configuration.
    tracker : ExperimentTracker, optional
        A pre-built tracker (injected in tests); when ``None`` one is built from
        ``config.tracking`` and closed when training finishes.
    data : tuple of (Iterable[Minibatch], GeneVocabulary), optional
        A pre-built data source (injected in tests); when ``None`` it is built
        from ``config.data``.

    Returns
    -------
    tuple of (TrainResult, OmicsVQVAE)
        The training outcome and the trained model.
    """
    _set_seed(config.seed)
    loader, vocabulary = data if data is not None else build_data(config.data)
    model = build_model(config.model, n_genes=vocabulary.n_genes)
    train_config = build_train_config(config.training)

    owns_tracker = tracker is None
    if tracker is None:
        run_config = cast(
            Dict[str, Any],
            OmegaConf.to_container(OmegaConf.structured(config), resolve=True),
        )
        tracker = build_tracker_from_config(config.tracking, run_config=run_config)
    try:
        result = run_train(model, loader, config=train_config, tracker=tracker)
    finally:
        if owns_tracker:
            tracker.finish()

    if config.training.checkpoint_path is not None:
        _save_checkpoint(model, vocabulary, config, config.training.checkpoint_path)
    return result, model


def _save_checkpoint(
    model: "OmicsVQVAE",
    vocabulary: "GeneVocabulary",
    config: ExperimentConfig,
    path: str,
) -> None:
    """Persist the trained model weights, gene vocabulary, and config."""
    import torch

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "organism": vocabulary.organism,
            "gene_ids": vocabulary.gene_ids,
            "config": OmegaConf.to_container(
                OmegaConf.structured(config), resolve=True
            ),
        },
        out,
    )
    logger.info("Saved checkpoint to %s", out)


# --------------------------------------------------------------------------- #
# Typer CLI
# --------------------------------------------------------------------------- #
def _build_app() -> Any:
    """Construct the Typer application (lazy import keeps ``typer`` optional)."""
    import typer

    app = typer.Typer(
        add_completion=False,
        help="Train or fine-tune an OQAE (Omics Quantized Auto Encoder) model.",
    )

    @app.command("train")
    def train_command(
        config: Optional[Path] = typer.Argument(
            None,
            help="Path to a YAML config file (defaults are used when omitted).",
        ),
        set_: Optional[List[str]] = typer.Option(
            None,
            "--set",
            "-s",
            help="OmegaConf dot-list override, e.g. -s training.lr=1e-2 "
            "(repeatable).",
        ),
    ) -> None:
        """Train a model from a config file (plus optional ``--set`` overrides)."""
        cfg = load_config(config, overrides=set_)
        result, _ = run_experiment(cfg)
        last = result.last_epoch
        if last is not None:
            typer.echo(
                f"Done: {result.global_step} steps, "
                f"final loss={last.loss:.4g} (recon={last.reconstruction_loss:.4g}, "
                f"vq={last.vq_loss:.4g}, ppl={last.perplexity:.4g})."
            )
        else:  # pragma: no cover - defensive; the loop always records an epoch
            typer.echo("Done: no steps were taken (empty data source).")

    return app


#: The Typer application (built at import time).
app = _build_app()


def main() -> None:  # pragma: no cover - thin console-script entry point
    """Console-script entry point (``oqae-train``)."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
