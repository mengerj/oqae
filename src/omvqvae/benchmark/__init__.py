"""
Offline-runnable benchmarking harness for OQAE (PR #9 scaffold).

Train tiny :class:`~omvqvae.models.vqvae.OmicsVQVAE` models under a few
likelihood / codebook configurations on shared data and emit a comparison table
of reconstruction quality, codebook utilization, and downstream separability.

The metric functions (:mod:`omvqvae.benchmark.metrics`) are pure and unit
tested; the harness (:mod:`omvqvae.benchmark.harness`) wires them onto the
training loop. Everything is offline-by-default — inject a re-iterable of
:class:`~omvqvae.data.dataset.Minibatch` for training and pass raw count arrays
(plus optional labels) for evaluation — so the same contract scales to a Census
stream later.
"""

from __future__ import annotations

from omvqvae.benchmark.harness import (
    BenchmarkConfig,
    BenchmarkResult,
    EvalMetrics,
    evaluate_model,
    format_results_table,
    results_to_dicts,
    run_benchmark,
    run_suite,
)
from omvqvae.benchmark.metrics import (
    CodebookUsage,
    ReconstructionMetrics,
    codebook_usage,
    reconstruction_metrics,
    separability_score,
)
from omvqvae.benchmark.report import (
    ReportFixture,
    default_report_configs,
    generate_report,
    make_benchmark_fixture,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "EvalMetrics",
    "evaluate_model",
    "run_benchmark",
    "run_suite",
    "results_to_dicts",
    "format_results_table",
    "CodebookUsage",
    "ReconstructionMetrics",
    "codebook_usage",
    "reconstruction_metrics",
    "separability_score",
    "ReportFixture",
    "make_benchmark_fixture",
    "default_report_configs",
    "generate_report",
]
