"""
Offline-friendly experiment tracking for OQAE.

Training (:mod:`omvqvae.train.loop`) logs scalar metrics — losses,
reconstruction terms, and codebook usage/perplexity — through a small
:class:`ExperimentTracker` interface so the training loop never imports
``wandb`` directly. Two backends are provided:

- :class:`ConsoleTracker` — a dependency-free default that logs metrics through
  the standard OQAE logger. Used whenever Weights & Biases is disabled or
  unavailable, which keeps the test-suite and offline runs working.
- :class:`WandbTracker` — a thin shell around a live ``wandb`` run. It is
  constructed with an already-initialized run object (dependency-injected) so
  the class logic is testable offline; the lazy ``wandb.init`` call that creates
  a real run lives in :func:`build_tracker` and is the only networked piece.

:func:`vqvae_metrics` flattens a :class:`~omvqvae.models.vqvae.VQVAEOutput` into
a plain ``{name: float}`` dict that any backend can log uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Type

from omvqvae.utils.logging import get_logger

if TYPE_CHECKING:
    from omvqvae.models.vqvae import VQVAEOutput

logger = get_logger(__name__)

__all__ = [
    "ExperimentTracker",
    "ConsoleTracker",
    "WandbTracker",
    "build_tracker",
    "vqvae_metrics",
]


def vqvae_metrics(output: "VQVAEOutput", *, prefix: str = "train") -> Dict[str, float]:
    """
    Flatten a :class:`~omvqvae.models.vqvae.VQVAEOutput` into scalar metrics.

    Per-codebook tensors (``perplexities``, ``usages``) are expanded into one
    entry per level so they can be logged as individual scalar series.

    Parameters
    ----------
    output : VQVAEOutput
        A model forward output.
    prefix : str, default "train"
        Prefix joined to each metric name with ``"/"`` (e.g. ``"train/loss"``).
        An empty prefix yields bare metric names.

    Returns
    -------
    Dict[str, float]
        Mapping of metric name to Python ``float``.
    """
    pre = f"{prefix}/" if prefix else ""
    metrics: Dict[str, float] = {
        f"{pre}loss": output.loss.item(),
        f"{pre}reconstruction_loss": output.reconstruction_loss.item(),
        f"{pre}vq_loss": output.vq_loss.item(),
        f"{pre}commitment_loss": output.commitment_loss.item(),
        f"{pre}codebook_loss": output.codebook_loss.item(),
        f"{pre}perplexity": output.perplexity.item(),
    }
    for level, value in enumerate(output.perplexities.detach().tolist()):
        metrics[f"{pre}perplexity/codebook_{level}"] = float(value)
    for level, value in enumerate(output.usages.detach().tolist()):
        metrics[f"{pre}usage/codebook_{level}"] = float(value)
    return metrics


class ExperimentTracker(ABC):
    """
    Minimal experiment-tracking interface used by the training loop.

    Implementations log scalar metrics and a run configuration, and release any
    resources on :meth:`finish`. Instances are usable as context managers, which
    always call :meth:`finish` on exit.
    """

    @abstractmethod
    def log_config(self, config: Mapping[str, Any]) -> None:
        """Record the run configuration (hyper-parameters, data source, …)."""

    @abstractmethod
    def log(self, metrics: Mapping[str, float], *, step: Optional[int] = None) -> None:
        """Log a mapping of scalar metrics, optionally at an explicit ``step``."""

    @abstractmethod
    def finish(self) -> None:
        """Flush and close the tracker; safe to call more than once."""

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.finish()


class ConsoleTracker(ExperimentTracker):
    """
    Dependency-free tracker that logs metrics through the OQAE logger.

    This is the default backend; it makes training fully offline and importable
    without ``wandb``. Metrics are emitted at ``INFO``; the configuration is
    logged once at ``INFO``.

    Parameters
    ----------
    run_name : str, optional
        Human-readable run name included in log lines.
    """

    def __init__(self, run_name: Optional[str] = None) -> None:
        self.run_name = run_name
        self._finished = False

    def log_config(self, config: Mapping[str, Any]) -> None:
        label = f" [{self.run_name}]" if self.run_name else ""
        logger.info("Run config%s: %s", label, dict(config))

    def log(self, metrics: Mapping[str, float], *, step: Optional[int] = None) -> None:
        rendered = ", ".join(f"{k}={v:.4g}" for k, v in metrics.items())
        prefix = "" if step is None else f"step {step} | "
        logger.info("%s%s", prefix, rendered)

    def finish(self) -> None:
        self._finished = True


class WandbTracker(ExperimentTracker):
    """
    Thin tracker over an already-initialized Weights & Biases run.

    The run object is injected rather than created here so the backend logic is
    testable offline with a fake; :func:`build_tracker` owns the lazy
    ``wandb.init`` that produces a real run.

    Parameters
    ----------
    run : Any
        A live ``wandb`` run (anything exposing ``log``, ``config``, and
        ``finish``).
    """

    def __init__(self, run: Any) -> None:
        self._run = run
        self._finished = False

    def log_config(self, config: Mapping[str, Any]) -> None:
        self._run.config.update(dict(config), allow_val_change=True)

    def log(self, metrics: Mapping[str, float], *, step: Optional[int] = None) -> None:
        self._run.log(dict(metrics), step=step)

    def finish(self) -> None:
        if not self._finished:
            self._run.finish()
            self._finished = True


def build_tracker(
    backend: str = "console",
    *,
    run_name: Optional[str] = None,
    project: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    offline: bool = False,
    **wandb_kwargs: Any,
) -> ExperimentTracker:
    """
    Construct an :class:`ExperimentTracker` for the requested backend.

    Parameters
    ----------
    backend : {"console", "none", "wandb"}, default "console"
        Which tracker to build. ``"console"`` and ``"none"`` both return a
        :class:`ConsoleTracker` (the latter is accepted as an alias for "no
        external tracking"). ``"wandb"`` lazily imports ``wandb`` and starts a
        run.
    run_name : str, optional
        Run name passed to the backend.
    project : str, optional
        W&B project name (ignored by the console backend).
    config : Mapping[str, Any], optional
        Run configuration; logged immediately after the tracker is created.
    offline : bool, default False
        For the ``"wandb"`` backend, start the run in offline mode
        (``mode="offline"``), which writes locally without a network call.
    **wandb_kwargs : Any
        Extra keyword arguments forwarded to ``wandb.init``.

    Returns
    -------
    ExperimentTracker
        The constructed tracker (configuration already logged when ``config``
        is given).

    Raises
    ------
    ValueError
        If ``backend`` is not one of the supported values.
    """
    backend = backend.lower()
    tracker: ExperimentTracker
    if backend in {"console", "none"}:
        tracker = ConsoleTracker(run_name=run_name)
    elif backend == "wandb":
        tracker = WandbTracker(
            _init_wandb_run(
                run_name=run_name,
                project=project,
                config=config,
                offline=offline,
                **wandb_kwargs,
            )
        )
    else:
        raise ValueError(
            f"Unknown tracking backend {backend!r}; expected 'console', "
            "'none', or 'wandb'."
        )
    if config is not None:
        tracker.log_config(config)
    return tracker


def _init_wandb_run(
    *,
    run_name: Optional[str],
    project: Optional[str],
    config: Optional[Mapping[str, Any]],
    offline: bool,
    **wandb_kwargs: Any,
) -> Any:  # pragma: no cover - thin I/O shell around wandb.init
    """Lazily import ``wandb`` and start a run (the only networked piece)."""
    import wandb

    return wandb.init(
        project=project,
        name=run_name,
        config=dict(config) if config is not None else None,
        mode="offline" if offline else None,
        **wandb_kwargs,
    )
