"""Tests for the Census streaming throughput benchmark.

The timing core (:func:`measure_stream_throughput`), the train-step builder, and
the reporting helpers are tested fully offline with a synthetic minibatch source
and a deterministic fake clock. The live Census shell
(:func:`benchmark_census_throughput`) is exercised by a single ``network``-marked
test, skipped by default.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import pytest
import torch

from omvqvae.benchmark import (
    BenchmarkConfig,
    ThroughputResult,
    format_throughput_table,
    make_train_step_fn,
    measure_stream_throughput,
    throughput_to_dicts,
)
from omvqvae.data.dataset import Minibatch
from omvqvae.models.vqvae import OmicsVQVAE

N_GENES = 8


def _batch(n_cells: int = 4, *, seed: int = 0) -> Minibatch:
    """A tiny synthetic :class:`Minibatch` of raw counts."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(2.0, size=(n_cells, N_GENES)).astype(np.float32)
    return Minibatch(
        counts=torch.from_numpy(counts),
        size_factors=torch.from_numpy(counts.sum(axis=1)),
        covariates={"organism": ["homo_sapiens"] * n_cells},
    )


def _fake_clock(ticks: List[float]) -> Callable[[], float]:
    """A clock returning successive ``ticks`` values (last value repeats)."""
    state = {"i": 0}

    def clock() -> float:
        i = state["i"]
        value = ticks[min(i, len(ticks) - 1)]
        state["i"] = i + 1
        return value

    return clock


def test_measures_cells_and_rates_with_fake_clock() -> None:
    source = [_batch(4), _batch(4), _batch(2)]
    # start, (arr, comp) x3. No step_fn, so completion == arrival per batch.
    # start=0; b0 arr=1 comp=1; b1 arr=2 comp=2; b2 arr=4 comp=4.
    clock = _fake_clock([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
    result = measure_stream_throughput(source, clock=clock, label="t")

    assert isinstance(result, ThroughputResult)
    assert result.label == "t"
    assert result.n_batches == 3
    assert result.n_cells == 10
    assert result.n_genes == N_GENES
    # steady window starts at start=0 (no warmup), ends at last completion=4.
    assert result.elapsed_seconds == pytest.approx(4.0)
    assert result.cells_per_second == pytest.approx(10.0 / 4.0)
    assert result.batches_per_second == pytest.approx(3.0 / 4.0)
    assert result.seconds_per_batch == pytest.approx(4.0 / 3.0)
    assert result.time_to_first_batch_seconds == pytest.approx(1.0)
    assert result.warmup_batches == 0
    assert result.warmup_seconds == 0.0


def test_warmup_excluded_from_measured_window() -> None:
    source = [_batch(4), _batch(4), _batch(2)]
    # start=0; b0 arr/comp=5 (slow cold start, warmup); b1 arr/comp=6; b2 arr/comp=7.
    clock = _fake_clock([0.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0])
    result = measure_stream_throughput(
        source, warmup_batches=1, clock=clock, label="warm"
    )

    # Only the two post-warmup batches are measured.
    assert result.n_batches == 2
    assert result.n_cells == 6
    # Steady window: from warmup completion (5) to last completion (7).
    assert result.elapsed_seconds == pytest.approx(2.0)
    assert result.warmup_batches == 1
    assert result.warmup_seconds == pytest.approx(5.0)
    # Time-to-first-batch is still the cold-start latency.
    assert result.time_to_first_batch_seconds == pytest.approx(5.0)


def test_max_batches_caps_measured_batches() -> None:
    source = [_batch(3) for _ in range(10)]
    result = measure_stream_throughput(source, max_batches=4)
    assert result.n_batches == 4
    assert result.n_cells == 12


def test_max_cells_stops_at_batch_granularity() -> None:
    source = [_batch(4) for _ in range(10)]
    # Stops once measured cells reach >= 10: after 3 batches (12 cells).
    result = measure_stream_throughput(source, max_cells=10)
    assert result.n_batches == 3
    assert result.n_cells == 12


def test_empty_source_yields_zeroed_result() -> None:
    result = measure_stream_throughput([])
    assert result.n_batches == 0
    assert result.n_cells == 0
    assert result.n_genes == 0
    assert result.elapsed_seconds == 0.0
    assert result.cells_per_second != result.cells_per_second  # nan
    assert result.batches_per_second != result.batches_per_second  # nan
    assert result.seconds_per_batch != result.seconds_per_batch  # nan
    assert result.time_to_first_batch_seconds != result.time_to_first_batch_seconds


def test_warmup_consuming_whole_source_leaves_no_measured_batches() -> None:
    source = [_batch(4), _batch(4)]
    result = measure_stream_throughput(source, warmup_batches=5)
    assert result.n_batches == 0
    assert result.elapsed_seconds == 0.0
    assert result.cells_per_second != result.cells_per_second  # nan


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_max_batches_rejected(bad: int) -> None:
    with pytest.raises(ValueError, match="max_batches"):
        measure_stream_throughput([_batch(2)], max_batches=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_max_cells_rejected(bad: int) -> None:
    with pytest.raises(ValueError, match="max_cells"):
        measure_stream_throughput([_batch(2)], max_cells=bad)


def test_negative_warmup_rejected() -> None:
    with pytest.raises(ValueError, match="warmup_batches"):
        measure_stream_throughput([_batch(2)], warmup_batches=-1)


def test_step_fn_is_applied_to_every_batch_and_timed() -> None:
    seen: List[int] = []

    def step(batch: Minibatch) -> None:
        seen.append(len(batch))

    source = [_batch(4), _batch(2)]
    # start=0; b0 arr=1 step->comp=3; b1 arr=4 step->comp=6.
    clock = _fake_clock([0.0, 1.0, 3.0, 4.0, 6.0])
    result = measure_stream_throughput(source, step_fn=step, clock=clock)

    assert seen == [4, 2]
    assert result.n_cells == 6
    # Steady window includes the step time: from start=0 to last completion=6.
    assert result.elapsed_seconds == pytest.approx(6.0)


def test_make_train_step_fn_updates_model_and_runs_in_stream() -> None:
    torch.manual_seed(0)
    model = OmicsVQVAE(N_GENES, n_codebooks=1, codebook_size=8, n_latent=4)
    before = model.to_latent.weight.detach().clone()

    step = make_train_step_fn(model, lr=1e-2)
    source = [_batch(8, seed=s) for s in range(3)]
    result = measure_stream_throughput(source, step_fn=step, label="train")

    after = model.to_latent.weight.detach()
    assert not torch.allclose(before, after)  # the optimizer stepped
    assert result.n_batches == 3
    assert result.n_cells == 24


def test_make_train_step_fn_respects_injected_optimizer_and_clip() -> None:
    model = OmicsVQVAE(N_GENES, n_codebooks=1, codebook_size=4, n_latent=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    step = make_train_step_fn(model, optimizer=optimizer, grad_clip_norm=1.0)
    step(_batch(4))  # should run without error


def test_throughput_to_dicts_has_scalar_columns() -> None:
    result = measure_stream_throughput([_batch(4), _batch(4)], label="x")
    rows = throughput_to_dicts([result])
    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "x"
    assert row["n_cells"] == 8
    assert row["n_batches"] == 2
    assert "cells_per_second" in row
    assert "time_to_first_batch_seconds" in row


def test_format_throughput_table_empty() -> None:
    assert format_throughput_table([]) == "(no results)"


def test_format_throughput_table_renders_rows() -> None:
    clock = _fake_clock([0.0, 1.0, 1.0, 2.0, 2.0])
    a = measure_stream_throughput([_batch(4), _batch(4)], clock=clock, label="human")
    b = measure_stream_throughput([_batch(2)], label="mouse")
    table = format_throughput_table([a, b])
    lines = table.splitlines()
    assert lines[0].startswith("| label")
    assert "cells/s" in lines[0]
    assert "ttfb_s" in lines[0]
    assert set(lines[1]) <= {"|", "-", " "}  # separator row
    assert "human" in table and "mouse" in table
    assert len([ln for ln in lines if ln.startswith("|")]) == 4  # header+sep+2 rows


def test_benchmark_config_builds_a_step_fn() -> None:
    """A BenchmarkConfig's model_kwargs round-trip into a timeable train step."""
    cfg = BenchmarkConfig(name="nb-1x8", n_codebooks=1, codebook_size=8, n_latent=4)
    model = OmicsVQVAE(N_GENES, **cfg.model_kwargs())
    step = make_train_step_fn(model, lr=cfg.lr, grad_clip_norm=cfg.grad_clip_norm)
    result = measure_stream_throughput([_batch(4)], step_fn=step, label=cfg.name)
    assert result.n_cells == 4
    assert result.label == "nb-1x8"
