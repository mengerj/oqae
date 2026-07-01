"""Tests for the benchmarking harness (config, run, suite, reporting)."""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest
from torch.utils.data import DataLoader

from omvqvae.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    evaluate_model,
    format_results_table,
    results_to_dicts,
    run_benchmark,
    run_suite,
)
from omvqvae.data.dataset import CountsDataset, collate_minibatch
from omvqvae.models.vqvae import OmicsVQVAE

N_GENES = 8
N_PROGRAMS = 3


def _synthetic_counts(seed: int = 0, n_cells: int = 48) -> tuple[np.ndarray, List[str]]:
    """Tiny raw-count matrix with latent program structure + labels."""
    rng = np.random.default_rng(seed)
    program_rates = rng.gamma(1.5, 1.0, size=(N_PROGRAMS, N_GENES))
    programs = rng.integers(0, N_PROGRAMS, size=n_cells)
    depths = rng.uniform(0.5, 2.0, size=n_cells)
    counts = rng.poisson(program_rates[programs] * depths[:, None]).astype(np.float32)
    labels = [f"program_{p}" for p in programs]
    return counts, labels


def _train_loader(counts: np.ndarray, batch_size: int = 16) -> DataLoader:
    dataset = CountsDataset(counts, organism="homo_sapiens")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_minibatch,
    )


def test_benchmark_config_model_kwargs() -> None:
    """``model_kwargs`` round-trips into a constructable model."""
    cfg = BenchmarkConfig(
        name="cfg", likelihood="zinb", n_codebooks=3, codebook_size=16, n_latent=8
    )
    kwargs = cfg.model_kwargs()
    assert kwargs["likelihood"] == "zinb"
    assert kwargs["n_codebooks"] == 3
    model = OmicsVQVAE(N_GENES, **kwargs)
    assert model.n_codebooks == 3
    assert model.codebook_size == 16


def test_evaluate_model_returns_all_metrics() -> None:
    """``evaluate_model`` bundles reconstruction, codebook, separability."""
    counts, labels = _synthetic_counts()
    model = OmicsVQVAE(N_GENES, n_latent=8, hidden_dims=(16,), codebook_size=16)
    metrics = evaluate_model(model, counts, eval_labels=labels)
    assert math.isfinite(metrics.reconstruction.nll)
    assert 0.0 <= metrics.codebook.utilization <= 1.0
    assert 0.0 <= metrics.separability <= 1.0
    assert 0.0 <= metrics.separability_quantized <= 1.0
    assert math.isclose(
        metrics.separability_gap,
        metrics.separability - metrics.separability_quantized,
    )
    assert metrics.codebook.codebook_size == 16


def test_evaluate_model_without_labels_has_nan_separability() -> None:
    """Separability is nan when no labels are provided."""
    counts, _ = _synthetic_counts(seed=1)
    model = OmicsVQVAE(N_GENES, n_latent=8, hidden_dims=(16,))
    metrics = evaluate_model(model, counts)
    assert math.isnan(metrics.separability)
    assert math.isnan(metrics.separability_quantized)
    assert math.isnan(metrics.separability_gap)


def test_evaluate_model_clustering_off_by_default() -> None:
    """NMI/ARI/ASW stay nan unless clustering is explicitly requested."""
    counts, labels = _synthetic_counts()
    model = OmicsVQVAE(N_GENES, n_latent=8, hidden_dims=(16,), codebook_size=16)
    metrics = evaluate_model(model, counts, eval_labels=labels)
    assert math.isnan(metrics.nmi)
    assert math.isnan(metrics.ari)
    assert math.isnan(metrics.cell_type_asw)


def test_evaluate_model_with_clustering_opt_in() -> None:
    """``compute_clustering=True`` populates the scIB metrics (needs scib-metrics)."""
    pytest.importorskip("scib_metrics")
    counts, labels = _synthetic_counts()
    model = OmicsVQVAE(N_GENES, n_latent=8, hidden_dims=(16,), codebook_size=16)
    metrics = evaluate_model(model, counts, eval_labels=labels, compute_clustering=True)
    assert 0.0 <= metrics.nmi <= 1.0
    assert not math.isnan(metrics.ari)
    assert 0.0 <= metrics.cell_type_asw <= 1.0
    # The clustering columns surface in the table only when computed.
    result = BenchmarkResult(
        config=BenchmarkConfig(name="c"), train_loss=1.0, eval=metrics
    )
    assert "nmi" in format_results_table([result])


def test_run_benchmark_trains_and_evaluates() -> None:
    """A single benchmark trains and returns finite metrics."""
    counts, labels = _synthetic_counts()
    loader = _train_loader(counts)
    cfg = BenchmarkConfig(
        name="nb-2x16",
        likelihood="nb",
        n_codebooks=2,
        codebook_size=16,
        n_latent=8,
        hidden_dims=(16,),
        max_epochs=2,
    )
    result = run_benchmark(
        cfg,
        loader,
        n_genes=N_GENES,
        eval_counts=counts,
        eval_labels=labels,
    )
    assert isinstance(result, BenchmarkResult)
    assert result.name == "nb-2x16"
    assert math.isfinite(result.train_loss)
    assert math.isfinite(result.eval.reconstruction.nll)
    assert result.eval.codebook.codebook_size == 16


def test_run_benchmark_is_reproducible() -> None:
    """The same config + seed gives the same training loss."""
    counts, labels = _synthetic_counts(seed=3)
    cfg = BenchmarkConfig(name="rep", n_latent=8, hidden_dims=(16,), max_epochs=2)
    first = run_benchmark(
        cfg, _train_loader(counts), n_genes=N_GENES, eval_counts=counts
    )
    second = run_benchmark(
        cfg, _train_loader(counts), n_genes=N_GENES, eval_counts=counts
    )
    assert first.train_loss == pytest.approx(second.train_loss)
    assert first.eval.reconstruction.nll == pytest.approx(
        second.eval.reconstruction.nll
    )


def test_run_suite_and_reporting() -> None:
    """A suite of configs produces one result each and a formatted table."""
    counts, labels = _synthetic_counts(seed=4)
    loader = _train_loader(counts)
    configs = [
        BenchmarkConfig(
            name="nb", likelihood="nb", n_latent=8, hidden_dims=(16,), max_epochs=1
        ),
        BenchmarkConfig(
            name="gaussian",
            likelihood="gaussian",
            n_latent=8,
            hidden_dims=(16,),
            max_epochs=1,
        ),
        BenchmarkConfig(
            name="nb-1cb",
            likelihood="nb",
            n_codebooks=1,
            n_latent=8,
            hidden_dims=(16,),
            max_epochs=1,
        ),
    ]
    results = run_suite(
        configs,
        loader,
        n_genes=N_GENES,
        eval_counts=counts,
        eval_labels=labels,
    )
    assert [r.name for r in results] == ["nb", "gaussian", "nb-1cb"]

    rows = results_to_dicts(results)
    assert len(rows) == 3
    assert rows[0]["likelihood"] == "nb"
    assert "perplexity" in rows[0]
    assert "separability_quantized" in rows[0]
    assert "separability_gap" in rows[0]

    table = format_results_table(results)
    assert table.startswith("| name")
    assert "gaussian" in table
    # Clustering was not requested, so those columns are omitted.
    assert "nmi" not in table
    # One header, one separator, three data rows.
    assert len(table.splitlines()) == 5


def test_format_results_table_empty() -> None:
    """Formatting an empty result list is graceful."""
    assert format_results_table([]) == "(no results)"
    assert results_to_dicts([]) == []


def test_run_benchmark_max_steps_caps_training() -> None:
    """``max_steps`` short-circuits training (smoke-sweep knob)."""
    counts, _ = _synthetic_counts(seed=5)
    cfg = BenchmarkConfig(
        name="capped",
        n_latent=8,
        hidden_dims=(16,),
        max_epochs=5,
        max_steps=1,
    )
    result = run_benchmark(
        cfg, _train_loader(counts, batch_size=16), n_genes=N_GENES, eval_counts=counts
    )
    assert math.isfinite(result.train_loss)
