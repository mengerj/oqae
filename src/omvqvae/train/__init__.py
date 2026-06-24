"""Training utilities and CLI interface.

Exposes the source-agnostic training loop (:func:`train`) and its configuration
/ result bundles (:class:`TrainConfig`, :class:`EpochMetrics`,
:class:`TrainResult`), plus the config-driven CLI (:func:`run_experiment`,
:class:`ExperimentConfig`, :func:`load_config`, and the Typer ``app``).
"""

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
    main,
    run_experiment,
)
from omvqvae.train.loop import EpochMetrics, TrainConfig, TrainResult, train

__all__ = [
    # training loop
    "TrainConfig",
    "EpochMetrics",
    "TrainResult",
    "train",
    # CLI / config-driven entry point
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
