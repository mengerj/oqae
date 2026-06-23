"""
A minimal, source-agnostic training loop for :class:`~omvqvae.models.vqvae.OmicsVQVAE`.

The loop pulls :class:`~omvqvae.data.dataset.Minibatch` objects from *any*
iterable data source (a local-AnnData ``DataLoader`` now, a Census stream
later), runs the model, steps an optimizer, and logs scalar metrics through an
:class:`~omvqvae.utils.tracking.ExperimentTracker`. Following the project's
"inject the data source" convention, the loader and optimizer are passed in, so
the pure loop is testable offline on a synthetic in-memory
:class:`~omvqvae.data.dataset.CountsDataset` with no network, ``wandb``, or CLI.

The heavy/networked I/O (building Census loaders, starting a real W&B run) lives
in the data layer and :func:`~omvqvae.utils.tracking.build_tracker`; this module
only consumes their products.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

import torch
from torch import nn

from omvqvae.utils.logging import get_logger
from omvqvae.utils.tracking import ConsoleTracker, ExperimentTracker, vqvae_metrics

if TYPE_CHECKING:
    from omvqvae.data.dataset import Minibatch
    from omvqvae.models.vqvae import OmicsVQVAE

logger = get_logger(__name__)

__all__ = [
    "TrainConfig",
    "EpochMetrics",
    "TrainResult",
    "train",
]


@dataclass
class TrainConfig:
    """
    Hyper-parameters for a :func:`train` run.

    Attributes
    ----------
    max_epochs : int, default 1
        Number of passes over the data source.
    lr : float, default 1e-3
        Learning rate for the default Adam optimizer (ignored when an optimizer
        is injected).
    weight_decay : float, default 0.0
        Weight decay for the default Adam optimizer.
    grad_clip_norm : float, optional
        If set, clip the global gradient norm to this value before each step.
    max_steps : int, optional
        If set, stop after this many optimizer steps in total (across epochs);
        useful for quick smoke runs.
    log_every : int, default 10
        Log per-step metrics every ``log_every`` optimizer steps. Set ``<= 0``
        to disable per-step logging (epoch summaries are always logged).
    device : str, default "cpu"
        Device the model and batches are moved to.

    Raises
    ------
    ValueError
        If ``max_epochs`` is not positive, or ``max_steps`` is set but not
        positive.
    """

    max_epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = None
    max_steps: Optional[int] = None
    log_every: int = 10
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be a positive integer.")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer when set.")


@dataclass
class EpochMetrics:
    """
    Mean metrics over a single epoch.

    Attributes
    ----------
    epoch : int
        Zero-based epoch index.
    steps : int
        Number of optimizer steps taken in the epoch.
    loss : float
        Mean total loss over the epoch.
    reconstruction_loss : float
        Mean reconstruction loss over the epoch.
    vq_loss : float
        Mean VQ loss over the epoch.
    perplexity : float
        Mean codebook perplexity over the epoch.
    """

    epoch: int
    steps: int
    loss: float
    reconstruction_loss: float
    vq_loss: float
    perplexity: float


@dataclass
class TrainResult:
    """
    Outcome of a :func:`train` run.

    Attributes
    ----------
    epochs : List[EpochMetrics]
        Per-epoch mean metrics, in order.
    global_step : int
        Total number of optimizer steps taken.
    """

    epochs: List[EpochMetrics] = field(default_factory=list)
    global_step: int = 0

    @property
    def last_epoch(self) -> Optional[EpochMetrics]:
        """The final epoch's metrics, or ``None`` if no epoch ran."""
        return self.epochs[-1] if self.epochs else None


def _move_batch(batch: "Minibatch", device: torch.device) -> "Minibatch":
    """Return a shallow copy of ``batch`` with its tensors on ``device``."""
    from omvqvae.data.dataset import Minibatch

    return Minibatch(
        counts=batch.counts.to(device),
        size_factors=batch.size_factors.to(device),
        covariates=batch.covariates,
    )


def train(
    model: "OmicsVQVAE",
    data_source: Iterable["Minibatch"],
    *,
    config: Optional[TrainConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    tracker: Optional[ExperimentTracker] = None,
) -> TrainResult:
    """
    Train an :class:`~omvqvae.models.vqvae.OmicsVQVAE` over a data source.

    Parameters
    ----------
    model : OmicsVQVAE
        The model to train (updated in place).
    data_source : Iterable[Minibatch]
        Any iterable yielding :class:`~omvqvae.data.dataset.Minibatch` objects;
        re-iterated once per epoch, so pass a ``DataLoader`` (or any re-iterable)
        rather than a one-shot iterator when ``max_epochs > 1``.
    config : TrainConfig, optional
        Run hyper-parameters; defaults to :class:`TrainConfig` (one epoch).
    optimizer : torch.optim.Optimizer, optional
        Optimizer to step. Defaults to Adam over ``model.parameters()`` using
        ``config.lr`` / ``config.weight_decay``.
    tracker : ExperimentTracker, optional
        Where to log metrics. Defaults to a :class:`ConsoleTracker`. The tracker
        is *not* closed here; the caller owns its lifecycle.

    Returns
    -------
    TrainResult
        Per-epoch metrics and the total optimizer-step count.
    """
    config = config or TrainConfig()
    tracker = tracker or ConsoleTracker()
    device = torch.device(config.device)
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

    result = TrainResult()
    for epoch in range(config.max_epochs):
        epoch_metrics = _run_epoch(
            model,
            data_source,
            optimizer=optimizer,
            config=config,
            device=device,
            tracker=tracker,
            epoch=epoch,
            global_step=result.global_step,
        )
        result.epochs.append(epoch_metrics)
        result.global_step += epoch_metrics.steps
        tracker.log(
            {
                "epoch": float(epoch),
                "epoch/loss": epoch_metrics.loss,
                "epoch/reconstruction_loss": epoch_metrics.reconstruction_loss,
                "epoch/vq_loss": epoch_metrics.vq_loss,
                "epoch/perplexity": epoch_metrics.perplexity,
            },
            step=result.global_step,
        )
        logger.info(
            "epoch %d done | steps=%d loss=%.4g recon=%.4g vq=%.4g ppl=%.4g",
            epoch,
            epoch_metrics.steps,
            epoch_metrics.loss,
            epoch_metrics.reconstruction_loss,
            epoch_metrics.vq_loss,
            epoch_metrics.perplexity,
        )
        if config.max_steps is not None and result.global_step >= config.max_steps:
            break

    return result


def _run_epoch(
    model: "OmicsVQVAE",
    data_source: Iterable["Minibatch"],
    *,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    device: torch.device,
    tracker: ExperimentTracker,
    epoch: int,
    global_step: int,
) -> EpochMetrics:
    """Run one training epoch, returning its mean metrics."""
    model.train()
    totals: Dict[str, float] = {
        "loss": 0.0,
        "reconstruction_loss": 0.0,
        "vq_loss": 0.0,
        "perplexity": 0.0,
    }
    steps = 0
    step = global_step
    for batch in data_source:
        batch = _move_batch(batch, device)
        optimizer.zero_grad()
        output = model(batch.counts, batch.size_factors)
        output.loss.backward()
        if config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        steps += 1
        step += 1
        totals["loss"] += output.loss.item()
        totals["reconstruction_loss"] += output.reconstruction_loss.item()
        totals["vq_loss"] += output.vq_loss.item()
        totals["perplexity"] += output.perplexity.item()

        if config.log_every > 0 and step % config.log_every == 0:
            tracker.log(vqvae_metrics(output, prefix="train"), step=step)

        if config.max_steps is not None and step >= config.max_steps:
            break

    denom = max(steps, 1)
    return EpochMetrics(
        epoch=epoch,
        steps=steps,
        loss=totals["loss"] / denom,
        reconstruction_loss=totals["reconstruction_loss"] / denom,
        vq_loss=totals["vq_loss"] / denom,
        perplexity=totals["perplexity"] / denom,
    )
