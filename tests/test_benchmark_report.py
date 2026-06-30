"""
Offline tests for the benchmark report module (:mod:`omvqvae.benchmark.report`).

The fixture builder, the config grid, the interpretation helper, and the
end-to-end :func:`generate_report` are exercised on a tiny synthetic fixture
with a couple of configs so the suite stays fast and offline.
"""

from __future__ import annotations

from typing import List

import pytest

from omvqvae.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    CodebookUsage,
    EvalMetrics,
    ReconstructionMetrics,
    ReportFixture,
    default_report_configs,
    generate_report,
    make_benchmark_fixture,
)
from omvqvae.benchmark.report import _interpret_results
from omvqvae.data.dataset import Minibatch


def _tiny_fixture() -> ReportFixture:
    return make_benchmark_fixture(
        n_cells=80,
        n_genes=12,
        n_programs=3,
        train_fraction=0.75,
        batch_size=16,
        seed=0,
    )


def _tiny_configs() -> List[BenchmarkConfig]:
    return [
        BenchmarkConfig(
            name="nb-2x16",
            likelihood="nb",
            n_codebooks=2,
            codebook_size=16,
            n_latent=8,
            hidden_dims=(16,),
            max_epochs=2,
        ),
        BenchmarkConfig(
            name="gaussian-2x16",
            likelihood="gaussian",
            n_codebooks=2,
            codebook_size=16,
            n_latent=8,
            hidden_dims=(16,),
            max_epochs=2,
        ),
    ]


def test_make_benchmark_fixture_shapes_and_split() -> None:
    """The fixture splits cells into train/eval and labels the held-out set."""
    fixture = _tiny_fixture()
    assert fixture.n_genes == 12
    assert fixture.n_train == 60  # round(80 * 0.75)
    assert fixture.eval_counts.shape == (20, 12)
    assert len(fixture.eval_labels) == 20
    assert (fixture.eval_counts >= 0).all()
    # Labels are program ids drawn from the configured program count.
    assert set(fixture.eval_labels).issubset({f"program_{p}" for p in range(3)})


def test_fixture_train_source_is_reiterable_minibatches() -> None:
    """The training source yields Minibatches and can be iterated repeatedly."""
    fixture = make_benchmark_fixture(n_cells=40, n_genes=8, batch_size=16, seed=1)

    def total_cells() -> int:
        return sum(len(batch) for batch in fixture.train_source)

    first = list(fixture.train_source)
    assert all(isinstance(b, Minibatch) for b in first)
    assert first[0].counts.shape[1] == 8
    # Re-iterating yields the same number of cells (a one-pass iterator would not).
    assert total_cells() == fixture.n_train
    assert total_cells() == fixture.n_train


def test_make_benchmark_fixture_rejects_degenerate_split() -> None:
    """A train fraction that leaves no eval (or train) cells is rejected."""
    with pytest.raises(ValueError):
        make_benchmark_fixture(n_cells=10, train_fraction=1.5)
    with pytest.raises(ValueError):
        # 3 cells * 0.1 rounds to 0 training cells.
        make_benchmark_fixture(n_cells=3, train_fraction=0.1)


def test_default_report_configs_cover_the_sweeps() -> None:
    """The default grid sweeps likelihoods, codebook sizes, and codebook counts."""
    configs = default_report_configs(max_epochs=4)
    names = {c.name for c in configs}
    assert {"nb-2x64", "zinb-2x64", "gaussian-2x64"} <= names
    assert {"nb-2x16", "nb-2x256"} <= names
    assert {"nb-1x64", "nb-4x64"} <= names
    assert {c.likelihood for c in configs} == {"nb", "zinb", "gaussian"}
    assert all(c.max_epochs == 4 for c in configs)


def test_generate_report_runs_and_renders() -> None:
    """End to end: train the configs and render an interpreted Markdown report."""
    fixture = _tiny_fixture()
    report = generate_report(fixture, _tiny_configs(), title="Test report")

    assert report.startswith("# Test report")
    for section in ("## Setup", "## Results", "## Interpretation", "## Caveats"):
        assert section in report
    # The results table and both config rows are present.
    assert "| name" in report
    assert "nb-2x16" in report
    assert "gaussian-2x16" in report
    # Setup line reflects the fixture split.
    assert "60 train / 20 held-out" in report


def _result(
    name: str,
    likelihood: str,
    *,
    nll: float,
    utilization: float,
    separability: float,
) -> BenchmarkResult:
    config = BenchmarkConfig(name=name, likelihood=likelihood)
    return BenchmarkResult(
        config=config,
        train_loss=1.0,
        eval=EvalMetrics(
            reconstruction=ReconstructionMetrics(nll=nll, mae=0.5),
            codebook=CodebookUsage(
                perplexity=8.0,
                utilization=utilization,
                perplexities=[8.0],
                utilizations=[utilization],
                codebook_size=16,
            ),
            separability=separability,
        ),
    )


def test_interpret_results_empty() -> None:
    """Interpretation degrades gracefully with no results."""
    assert "No results" in _interpret_results([])


def test_interpret_results_flags_collapse_and_compares_likelihoods() -> None:
    """Interpretation names a collapsed codebook and the NB-vs-Gaussian anchor."""
    results = [
        _result("nb-2x64", "nb", nll=10.0, utilization=0.9, separability=0.8),
        _result(
            "gaussian-2x64", "gaussian", nll=2.0, utilization=0.8, separability=0.6
        ),
        _result("nb-2x256", "nb", nll=11.0, utilization=0.2, separability=0.7),
    ]
    prose = _interpret_results(results)
    assert "nb-2x64" in prose  # best NB reconstruction (lowest NLL)
    assert "`nb-2x256`" in prose  # flagged as collapsed (utilization 0.2 < 0.5)
    assert "below" in prose
    assert "raw-count NB model" in prose  # NB separability >= Gaussian
    assert "Batch effects" in prose


def test_interpret_results_no_collapse_and_gaussian_wins() -> None:
    """The no-collapse branch and the Gaussian-wins verdict both render."""
    results = [
        _result("nb-2x64", "nb", nll=10.0, utilization=0.9, separability=0.5),
        _result(
            "gaussian-2x64", "gaussian", nll=2.0, utilization=0.95, separability=0.9
        ),
    ]
    prose = _interpret_results(results)
    assert "no collapse observed" in prose
    assert "log-normalized Gaussian model" in prose


def test_interpret_results_handles_nan_separability() -> None:
    """A nan separability anchor skips the NB-vs-Gaussian comparison cleanly."""
    nan = float("nan")
    results = [
        _result("nb-2x64", "nb", nll=10.0, utilization=0.9, separability=nan),
        _result(
            "gaussian-2x64", "gaussian", nll=2.0, utilization=0.9, separability=nan
        ),
    ]
    prose = _interpret_results(results)
    assert "Raw-count NB vs log-normalized Gaussian" not in prose
    # Separability bullet is omitted when every value is nan.
    assert "Downstream separability" not in prose
