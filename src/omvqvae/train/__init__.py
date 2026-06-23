"""Training utilities and CLI interface.

Exposes the source-agnostic training loop (:func:`train`) and its configuration
/ result bundles (:class:`TrainConfig`, :class:`EpochMetrics`,
:class:`TrainResult`).
"""

from omvqvae.train.loop import EpochMetrics, TrainConfig, TrainResult, train

__all__ = [
    "TrainConfig",
    "EpochMetrics",
    "TrainResult",
    "train",
]
