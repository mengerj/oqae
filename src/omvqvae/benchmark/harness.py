"""
A small, offline-runnable benchmarking harness for OQAE.

This is the PR #9 scaffold: train tiny :class:`~omvqvae.models.vqvae.OmicsVQVAE`
models under a few likelihood / codebook configurations on the *same* data and
emit a comparison table of reconstruction quality, codebook utilization, and
downstream separability. It establishes the metrics/reporting shape before the
real Census-scale sweeps are wired (those reuse the same
:class:`BenchmarkConfig` / :class:`BenchmarkResult` contract over a streamed
source).

Following the project's "inject the data source" convention, the training data
is passed in as any re-iterable of
:class:`~omvqvae.data.dataset.Minibatch` (a local-AnnData ``DataLoader`` now, a
Census stream later) and the evaluation set as raw count arrays plus optional
labels, so the whole harness runs offline on a synthetic fixture with no network.

.. code-block:: text

    configs ─► run_suite(train_source, eval_counts, eval_labels) ─► [BenchmarkResult]
                                                                     │
                                                          format_results_table
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

import torch

from omvqvae.benchmark.metrics import (
    CodebookUsage,
    ReconstructionMetrics,
    codebook_usage,
    reconstruction_metrics,
    separability_score,
)
from omvqvae.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omvqvae.benchmark.metrics import SizeFactorsLike
    from omvqvae.data.dataset import Minibatch
    from omvqvae.data.normalize import CountMatrix
    from omvqvae.models.vqvae import OmicsVQVAE

logger = get_logger(__name__)

__all__ = [
    "BenchmarkConfig",
    "EvalMetrics",
    "BenchmarkResult",
    "evaluate_model",
    "run_benchmark",
    "run_suite",
    "results_to_dicts",
    "format_results_table",
]


@dataclass
class BenchmarkConfig:
    """
    A single benchmark cell: a model architecture plus its training knobs.

    The model-architecture fields map onto
    :class:`~omvqvae.models.vqvae.OmicsVQVAE` constructor arguments; the
    remaining fields drive the :func:`omvqvae.train.train` loop. ``n_genes`` is
    supplied separately at run time (derived from the evaluation data), so a
    config is reusable across feature spaces.

    Attributes
    ----------
    name : str
        Human-readable label for the row in the comparison table.
    likelihood : {"nb", "zinb", "gaussian"}, default "nb"
        Reconstruction likelihood to benchmark.
    n_codebooks : int, default 2
        Number of residual quantization levels.
    codebook_size : int, default 64
        Entries per codebook.
    n_latent : int, default 16
        Latent / codebook vector dimensionality.
    hidden_dims : Sequence[int], default (64,)
        Encoder hidden widths (the decoder mirrors them).
    max_epochs : int, default 3
        Training epochs.
    lr : float, default 1e-3
        Adam learning rate.
    grad_clip_norm : float, optional
        Global gradient-norm clip, if set.
    max_steps : int, optional
        Optional cap on total optimizer steps (quick smoke sweeps).
    seed : int, default 0
        Seed applied to ``torch`` before building/training the model so runs are
        reproducible.
    """

    name: str
    likelihood: str = "nb"
    n_codebooks: int = 2
    codebook_size: int = 64
    n_latent: int = 16
    hidden_dims: Sequence[int] = (64,)
    max_epochs: int = 3
    lr: float = 1e-3
    grad_clip_norm: Optional[float] = None
    max_steps: Optional[int] = None
    seed: int = 0

    def model_kwargs(self) -> Dict[str, Any]:
        """Return the :class:`~omvqvae.models.vqvae.OmicsVQVAE` keyword arguments."""
        return {
            "likelihood": self.likelihood,
            "n_codebooks": self.n_codebooks,
            "codebook_size": self.codebook_size,
            "n_latent": self.n_latent,
            "hidden_dims": tuple(self.hidden_dims),
        }


@dataclass
class EvalMetrics:
    """
    Evaluation metrics for one trained model on the held-out set.

    Attributes
    ----------
    reconstruction : ReconstructionMetrics
        Reconstruction NLL / MAE.
    codebook : CodebookUsage
        Codebook perplexity / utilization over the evaluation cells.
    separability : float
        Nearest-centroid separability of the latent given the eval labels, or
        ``nan`` when labels are absent or degenerate.
    """

    reconstruction: ReconstructionMetrics
    codebook: CodebookUsage
    separability: float


@dataclass
class BenchmarkResult:
    """
    The outcome of training and evaluating one :class:`BenchmarkConfig`.

    Attributes
    ----------
    config : BenchmarkConfig
        The configuration that produced this result.
    train_loss : float
        Final-epoch mean training loss.
    eval : EvalMetrics
        Held-out evaluation metrics.
    """

    config: BenchmarkConfig
    train_loss: float
    eval: EvalMetrics

    @property
    def name(self) -> str:
        """The config's display name."""
        return self.config.name


def evaluate_model(
    model: "OmicsVQVAE",
    eval_counts: "CountMatrix",
    *,
    eval_size_factors: Optional["SizeFactorsLike"] = None,
    eval_labels: Optional[Sequence[object]] = None,
    batch_size: int = 512,
) -> EvalMetrics:
    """
    Compute reconstruction, codebook, and separability metrics for a model.

    Parameters
    ----------
    model : OmicsVQVAE
        A trained model whose ``n_genes`` matches ``eval_counts``.
    eval_counts : numpy.ndarray or scipy.sparse.spmatrix
        Held-out raw counts of shape ``(n_cells, n_genes)``.
    eval_size_factors : torch.Tensor or numpy.ndarray, optional
        Per-cell size factors; computed from the counts when omitted.
    eval_labels : Sequence, optional
        Per-cell labels for the separability score (e.g. cell type / program).
        Separability is ``nan`` when omitted.
    batch_size : int, default 512
        Cells per forward pass.

    Returns
    -------
    EvalMetrics
        The bundled held-out metrics.
    """
    # Lazy import keeps `import omvqvae.benchmark` light and avoids a cycle.
    from omvqvae.inference import encode

    recon = reconstruction_metrics(
        model,
        eval_counts,
        size_factors=eval_size_factors,
        batch_size=batch_size,
    )
    encoded = encode(
        model,
        eval_counts,
        size_factors=eval_size_factors,
        batch_size=batch_size,
    )
    usage = codebook_usage(encoded.codes, model.codebook_size)
    if eval_labels is None:
        separability = float("nan")
    else:
        separability = separability_score(encoded.latent, eval_labels)
    return EvalMetrics(reconstruction=recon, codebook=usage, separability=separability)


def run_benchmark(
    config: BenchmarkConfig,
    train_source: Iterable["Minibatch"],
    *,
    n_genes: int,
    eval_counts: "CountMatrix",
    eval_size_factors: Optional["SizeFactorsLike"] = None,
    eval_labels: Optional[Sequence[object]] = None,
    eval_batch_size: int = 512,
) -> BenchmarkResult:
    """
    Train one configuration and evaluate it on the held-out set.

    Parameters
    ----------
    config : BenchmarkConfig
        The architecture / training knobs to run.
    train_source : Iterable[Minibatch]
        Re-iterable training data (a ``DataLoader`` or any re-iterable of
        :class:`~omvqvae.data.dataset.Minibatch`). Re-iterated once per epoch.
    n_genes : int
        Feature-space size; the model is built with this many genes (must match
        ``eval_counts`` and the ``train_source`` minibatches).
    eval_counts : numpy.ndarray or scipy.sparse.spmatrix
        Held-out raw counts ``(n_cells, n_genes)``.
    eval_size_factors : torch.Tensor or numpy.ndarray, optional
        Per-cell eval size factors; computed from the counts when omitted.
    eval_labels : Sequence, optional
        Per-cell eval labels for the separability score.
    eval_batch_size : int, default 512
        Cells per evaluation forward pass.

    Returns
    -------
    BenchmarkResult
        Final training loss plus held-out metrics for ``config``.
    """
    from omvqvae.models.vqvae import OmicsVQVAE
    from omvqvae.train import TrainConfig, train
    from omvqvae.utils.tracking import ConsoleTracker

    torch.manual_seed(config.seed)
    model = OmicsVQVAE(n_genes, **config.model_kwargs())
    train_config = TrainConfig(
        max_epochs=config.max_epochs,
        lr=config.lr,
        grad_clip_norm=config.grad_clip_norm,
        max_steps=config.max_steps,
    )
    logger.info("Benchmark %r: training (%s).", config.name, config.model_kwargs())
    result = train(
        model,
        train_source,
        config=train_config,
        tracker=ConsoleTracker(run_name=config.name),
    )
    last = result.last_epoch
    train_loss = last.loss if last is not None else float("nan")

    eval_metrics = evaluate_model(
        model,
        eval_counts,
        eval_size_factors=eval_size_factors,
        eval_labels=eval_labels,
        batch_size=eval_batch_size,
    )
    return BenchmarkResult(config=config, train_loss=train_loss, eval=eval_metrics)


def run_suite(
    configs: Sequence[BenchmarkConfig],
    train_source: Iterable["Minibatch"],
    *,
    n_genes: int,
    eval_counts: "CountMatrix",
    eval_size_factors: Optional["SizeFactorsLike"] = None,
    eval_labels: Optional[Sequence[object]] = None,
    eval_batch_size: int = 512,
) -> List[BenchmarkResult]:
    """
    Run :func:`run_benchmark` for each config against shared data.

    Parameters
    ----------
    configs : Sequence[BenchmarkConfig]
        Configurations to benchmark, in order.
    train_source : Iterable[Minibatch]
        Shared, re-iterable training data (see :func:`run_benchmark`).
    n_genes : int
        Feature-space size shared by every config.
    eval_counts : numpy.ndarray or scipy.sparse.spmatrix
        Shared held-out counts.
    eval_size_factors : torch.Tensor or numpy.ndarray, optional
        Shared held-out size factors.
    eval_labels : Sequence, optional
        Shared held-out labels for separability.
    eval_batch_size : int, default 512
        Cells per evaluation forward pass.

    Returns
    -------
    List[BenchmarkResult]
        One result per config, in input order.
    """
    results: List[BenchmarkResult] = []
    for config in configs:
        results.append(
            run_benchmark(
                config,
                train_source,
                n_genes=n_genes,
                eval_counts=eval_counts,
                eval_size_factors=eval_size_factors,
                eval_labels=eval_labels,
                eval_batch_size=eval_batch_size,
            )
        )
    return results


def results_to_dicts(results: Sequence[BenchmarkResult]) -> List[Dict[str, Any]]:
    """
    Flatten benchmark results into plain dicts (e.g. for CSV / a DataFrame).

    Parameters
    ----------
    results : Sequence[BenchmarkResult]
        Results to flatten.

    Returns
    -------
    List[Dict[str, Any]]
        One row per result with scalar columns (name, the swept hyper-parameters,
        and every metric).
    """
    rows: List[Dict[str, Any]] = []
    for result in results:
        cfg = result.config
        rows.append(
            {
                "name": cfg.name,
                "likelihood": cfg.likelihood,
                "n_codebooks": cfg.n_codebooks,
                "codebook_size": cfg.codebook_size,
                "n_latent": cfg.n_latent,
                "train_loss": result.train_loss,
                "eval_nll": result.eval.reconstruction.nll,
                "eval_mae": result.eval.reconstruction.mae,
                "perplexity": result.eval.codebook.perplexity,
                "utilization": result.eval.codebook.utilization,
                "separability": result.eval.separability,
            }
        )
    return rows


def format_results_table(results: Sequence[BenchmarkResult]) -> str:
    """
    Render benchmark results as a Markdown comparison table.

    Parameters
    ----------
    results : Sequence[BenchmarkResult]
        Results to tabulate (one row each).

    Returns
    -------
    str
        A GitHub-flavoured Markdown table; ``"(no results)"`` if empty.
    """
    if not results:
        return "(no results)"

    headers = [
        "name",
        "likelihood",
        "codebooks",
        "train_loss",
        "eval_nll",
        "eval_mae",
        "perplexity",
        "utilization",
        "separability",
    ]

    def _fmt(value: float) -> str:
        return "nan" if value != value else f"{value:.4g}"

    rows: List[List[str]] = []
    for result in results:
        cfg = result.config
        rows.append(
            [
                cfg.name,
                cfg.likelihood,
                f"{cfg.n_codebooks}x{cfg.codebook_size}",
                _fmt(result.train_loss),
                _fmt(result.eval.reconstruction.nll),
                _fmt(result.eval.reconstruction.mae),
                _fmt(result.eval.codebook.perplexity),
                _fmt(result.eval.codebook.utilization),
                _fmt(result.eval.separability),
            ]
        )

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    sep = "| " + " | ".join("-" * widths[col] for col in range(len(headers))) + " |"
    header_line = (
        "| "
        + " | ".join(headers[col].ljust(widths[col]) for col in range(len(headers)))
        + " |"
    )
    body = [
        "| "
        + " | ".join(row[col].ljust(widths[col]) for col in range(len(headers)))
        + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep, *body])
