"""
Streaming throughput / scaling benchmark for OQAE (PR #9 slice 3).

Slices 1 and 2 benchmark *model* quality (reconstruction, codebook utilization,
separability) on an offline fixture. This slice benchmarks the *data path*: how
fast cells can be streamed (and, optionally, trained on) — cells/s, batches/s,
and time-to-N-batches — so the CELLxGENE Census streaming stack can be profiled
and the report's throughput exit-criterion closed.

Mirroring the data-layer pattern, the timing logic is a **pure core**
(:func:`measure_stream_throughput`) that consumes any re-iterable of
:class:`~omvqvae.data.dataset.Minibatch` and an injectable clock, so it is unit
tested offline with a synthetic source and a fake clock. The heavy/networked
shell (:func:`benchmark_census_throughput`) only builds a live
``build_census_dataloader`` and feeds it to that core, reusing the same
:class:`~omvqvae.benchmark.harness.BenchmarkConfig` model contract for the
optional train-step timing.

.. code-block:: text

    source (Minibatch stream) ─► measure_stream_throughput ─► ThroughputResult
                                  (optional per-batch step_fn)        │
                                                          format_throughput_table
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
)

from omvqvae.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from omvqvae.benchmark.harness import BenchmarkConfig
    from omvqvae.data.dataset import Minibatch
    from omvqvae.models.vqvae import OmicsVQVAE

logger = get_logger(__name__)

#: A per-batch side-effect callback (e.g. a training step) timed in the stream.
StepFn = Callable[["Minibatch"], None]

__all__ = [
    "ThroughputResult",
    "measure_stream_throughput",
    "make_train_step_fn",
    "throughput_to_dicts",
    "format_throughput_table",
    "benchmark_census_throughput",
]


@dataclass
class ThroughputResult:
    """
    Streaming-throughput measurement over a :class:`Minibatch` source.

    All rates are computed over the *measured* (post-warmup) batches only; the
    warmup batches are timed separately so a cold start (opening TileDB-SOMA
    readers, filling buffers) does not depress the steady-state numbers.

    Attributes
    ----------
    label : str
        Human-readable label for the run (e.g. organism / config name).
    n_batches : int
        Number of measured batches (excludes warmup).
    n_cells : int
        Total cells across the measured batches.
    n_genes : int
        Feature-space size (columns of the streamed counts); ``0`` if no batch
        was measured.
    elapsed_seconds : float
        Wall-clock seconds spent producing (and stepping over) the measured
        batches.
    warmup_batches : int
        Number of leading batches excluded from the measured window.
    warmup_seconds : float
        Wall-clock seconds spent on the warmup batches (``0`` when
        ``warmup_batches`` is 0 or no warmup batch was reached).
    time_to_first_batch_seconds : float
        Latency from the start of iteration to the first batch being yielded
        (the cold-start cost); ``nan`` if the source yielded nothing.
    """

    label: str
    n_batches: int
    n_cells: int
    n_genes: int
    elapsed_seconds: float
    warmup_batches: int
    warmup_seconds: float
    time_to_first_batch_seconds: float

    @property
    def cells_per_second(self) -> float:
        """Mean streamed cells per second over the measured window."""
        return (
            self.n_cells / self.elapsed_seconds
            if self.elapsed_seconds > 0
            else float("nan")
        )

    @property
    def batches_per_second(self) -> float:
        """Mean streamed batches per second over the measured window."""
        return (
            self.n_batches / self.elapsed_seconds
            if self.elapsed_seconds > 0
            else float("nan")
        )

    @property
    def seconds_per_batch(self) -> float:
        """Mean seconds per measured batch."""
        return (
            self.elapsed_seconds / self.n_batches
            if self.n_batches > 0
            else float("nan")
        )


def measure_stream_throughput(
    source: Iterable["Minibatch"],
    *,
    max_batches: Optional[int] = None,
    max_cells: Optional[int] = None,
    warmup_batches: int = 0,
    step_fn: Optional[StepFn] = None,
    clock: Callable[[], float] = perf_counter,
    label: str = "stream",
) -> ThroughputResult:
    """
    Measure streaming throughput over an iterable of minibatches.

    Iterates ``source`` (a ``DataLoader``, a
    :class:`~omvqvae.data.census.CensusMinibatchLoader`, or any iterable of
    :class:`~omvqvae.data.dataset.Minibatch`), optionally applying ``step_fn`` to
    each batch (e.g. a model train step, so the timing includes the forward /
    backward pass), and records cells/s, batches/s, and time-to-first-batch.

    The first ``warmup_batches`` batches are processed but excluded from the
    measured window, and the steady-state clock starts only once they complete —
    so cold-start latency is reported separately
    (:attr:`ThroughputResult.warmup_seconds` /
    :attr:`ThroughputResult.time_to_first_batch_seconds`) instead of biasing the
    rates.

    Parameters
    ----------
    source : Iterable[Minibatch]
        The minibatch stream to time.
    max_batches : int, optional
        Stop after this many *measured* batches. ``None`` consumes the whole
        source.
    max_cells : int, optional
        Stop once this many *measured* cells have been seen (checked at batch
        granularity). ``None`` consumes the whole source.
    warmup_batches : int, default 0
        Number of leading batches to process but exclude from the measured
        window.
    step_fn : Callable[[Minibatch], None], optional
        Per-batch side effect timed inside the stream (e.g.
        :func:`make_train_step_fn`). ``None`` times raw streaming only.
    clock : Callable[[], float], default time.perf_counter
        Monotonic clock returning seconds; injected for deterministic tests.
    label : str, default "stream"
        Label recorded on the result.

    Returns
    -------
    ThroughputResult
        The measured throughput.

    Raises
    ------
    ValueError
        If ``warmup_batches`` is negative, or ``max_batches`` / ``max_cells`` is
        set but not positive.
    """
    if warmup_batches < 0:
        raise ValueError("warmup_batches must be non-negative.")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be a positive integer when set.")
    if max_cells is not None and max_cells < 1:
        raise ValueError("max_cells must be a positive integer when set.")

    start = clock()
    first_arrival: Optional[float] = None
    steady_start: Optional[float] = None
    last_completion = start
    warmup_seconds = 0.0
    seen = 0
    n_batches = 0
    n_cells = 0
    n_genes = 0

    for batch in source:
        arrival = clock()
        if first_arrival is None:
            first_arrival = arrival
        if step_fn is not None:
            step_fn(batch)
        completion = clock()
        seen += 1

        if seen <= warmup_batches:
            last_completion = completion
            if seen == warmup_batches:
                warmup_seconds = completion - start
                steady_start = completion
            continue

        if steady_start is None:
            # No warmup requested: the steady window starts at iteration start.
            steady_start = start
        n_batches += 1
        n_cells += len(batch)
        if batch.counts.ndim == 2:
            n_genes = int(batch.counts.shape[1])
        last_completion = completion

        if max_batches is not None and n_batches >= max_batches:
            break
        if max_cells is not None and n_cells >= max_cells:
            break

    elapsed = (
        (last_completion - steady_start)
        if (steady_start is not None and n_batches > 0)
        else 0.0
    )
    time_to_first = (
        (first_arrival - start) if first_arrival is not None else float("nan")
    )

    logger.info(
        "Throughput %r: %d cells in %d batches over %.4gs (%.4g cells/s).",
        label,
        n_cells,
        n_batches,
        elapsed,
        n_cells / elapsed if elapsed > 0 else float("nan"),
    )
    return ThroughputResult(
        label=label,
        n_batches=n_batches,
        n_cells=n_cells,
        n_genes=n_genes,
        elapsed_seconds=elapsed,
        warmup_batches=warmup_batches,
        warmup_seconds=warmup_seconds,
        time_to_first_batch_seconds=time_to_first,
    )


def make_train_step_fn(
    model: "OmicsVQVAE",
    *,
    optimizer: "Optional[torch.optim.Optimizer]" = None,
    grad_clip_norm: Optional[float] = None,
    lr: float = 1e-3,
    device: str = "cpu",
) -> StepFn:
    """
    Build a single-optimizer-step callback for throughput timing.

    The returned closure runs one training step per :class:`Minibatch` (forward,
    backward, optional grad-clip, optimizer step), mirroring the inner loop of
    :func:`omvqvae.train.train`. Passing it as ``step_fn`` to
    :func:`measure_stream_throughput` measures *end-to-end* throughput — the
    streaming **and** the model update — rather than raw streaming alone.

    Parameters
    ----------
    model : OmicsVQVAE
        The model to step (updated in place); moved to ``device``.
    optimizer : torch.optim.Optimizer, optional
        Optimizer to step. Defaults to Adam over ``model.parameters()`` at
        ``lr``.
    grad_clip_norm : float, optional
        If set, clip the global gradient norm before each step.
    lr : float, default 1e-3
        Learning rate for the default Adam optimizer (ignored when ``optimizer``
        is supplied).
    device : str, default "cpu"
        Device the model and batches are moved to.

    Returns
    -------
    Callable[[Minibatch], None]
        A per-batch training-step callback.
    """
    import torch
    from torch import nn

    dev = torch.device(device)
    model.to(dev)
    opt = optimizer or torch.optim.Adam(model.parameters(), lr=lr)

    def step(batch: "Minibatch") -> None:
        model.train()
        counts = batch.counts.to(dev)
        size_factors = batch.size_factors.to(dev)
        opt.zero_grad()
        output = model(counts, size_factors)
        output.loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        opt.step()

    return step


def throughput_to_dicts(results: Sequence[ThroughputResult]) -> List[Dict[str, Any]]:
    """
    Flatten throughput results into plain dicts (e.g. for CSV / a DataFrame).

    Parameters
    ----------
    results : Sequence[ThroughputResult]
        Results to flatten.

    Returns
    -------
    List[Dict[str, Any]]
        One scalar-valued row per result.
    """
    rows: List[Dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "label": result.label,
                "n_batches": result.n_batches,
                "n_cells": result.n_cells,
                "n_genes": result.n_genes,
                "elapsed_seconds": result.elapsed_seconds,
                "cells_per_second": result.cells_per_second,
                "batches_per_second": result.batches_per_second,
                "seconds_per_batch": result.seconds_per_batch,
                "warmup_seconds": result.warmup_seconds,
                "time_to_first_batch_seconds": result.time_to_first_batch_seconds,
            }
        )
    return rows


def format_throughput_table(results: Sequence[ThroughputResult]) -> str:
    """
    Render throughput results as a Markdown comparison table.

    Parameters
    ----------
    results : Sequence[ThroughputResult]
        Results to tabulate (one row each).

    Returns
    -------
    str
        A GitHub-flavoured Markdown table; ``"(no results)"`` if empty.
    """
    if not results:
        return "(no results)"

    headers = [
        "label",
        "cells",
        "batches",
        "genes",
        "elapsed_s",
        "cells/s",
        "batches/s",
        "ttfb_s",
    ]

    def _fmt(value: float) -> str:
        return "nan" if value != value else f"{value:.4g}"

    rows: List[List[str]] = []
    for result in results:
        rows.append(
            [
                result.label,
                str(result.n_cells),
                str(result.n_batches),
                str(result.n_genes),
                _fmt(result.elapsed_seconds),
                _fmt(result.cells_per_second),
                _fmt(result.batches_per_second),
                _fmt(result.time_to_first_batch_seconds),
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


def benchmark_census_throughput(
    organism: str,
    *,
    config: "Optional[BenchmarkConfig]" = None,
    census_version: Optional[str] = None,
    obs_value_filter: Optional[str] = None,
    var_value_filter: Optional[str] = None,
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: Optional[int] = 0,
    max_batches: Optional[int] = 50,
    warmup_batches: int = 1,
    device: str = "cpu",
    label: Optional[str] = None,
) -> ThroughputResult:  # pragma: no cover - requires live Census/TileDB-SOMA
    """
    Profile CELLxGENE Census streaming throughput end to end.

    Opens a pinned Census, builds a streaming loader for ``organism`` (optionally
    filtered), and times it with :func:`measure_stream_throughput`. When
    ``config`` is given, a model is built from it (reusing the
    :class:`~omvqvae.benchmark.harness.BenchmarkConfig` contract) and a train step
    is timed per batch, so the result measures full streaming-plus-training
    throughput; otherwise raw streaming is measured.

    This is the networked I/O shell — it requires a live Census and is exercised
    by a ``network``-marked test. The timing logic itself lives in the offline
    :func:`measure_stream_throughput`.

    Parameters
    ----------
    organism : str
        Organism identifier selecting the Census experiment.
    config : BenchmarkConfig, optional
        If set, build a model from it and time a per-batch training step (end to
        end). If ``None``, time raw streaming only.
    census_version : str, optional
        Census release to open; defaults to
        :data:`omvqvae.data.census.DEFAULT_CENSUS_VERSION`.
    obs_value_filter : str, optional
        TileDB-SOMA ``obs`` value filter selecting cells.
    var_value_filter : str, optional
        TileDB-SOMA ``var`` value filter selecting genes.
    batch_size : int, default 128
        Cells per streamed minibatch.
    num_workers : int, default 0
        Dataloader worker processes.
    shuffle : bool, default True
        Whether the Census stream shuffles cells.
    seed : int, optional
        Shuffle seed for reproducible streaming.
    max_batches : int, optional, default 50
        Stop after this many measured batches.
    warmup_batches : int, default 1
        Leading batches excluded from the measured window (cold start).
    device : str, default "cpu"
        Device for the optional model train step.
    label : str, optional
        Result label; defaults to the organism (with the config name appended
        when ``config`` is given).

    Returns
    -------
    ThroughputResult
        The measured Census streaming throughput.
    """
    from omvqvae.data.census import (
        DEFAULT_CENSUS_VERSION,
        build_census_dataloader,
    )

    version = census_version or DEFAULT_CENSUS_VERSION
    if label is None:
        label = organism if config is None else f"{organism}/{config.name}"

    from omvqvae.data.census import open_census

    with open_census(census_version=version) as census:
        loader = build_census_dataloader(
            census,
            organism,
            obs_value_filter=obs_value_filter,
            var_value_filter=var_value_filter,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            seed=seed,
        )
        try:
            step_fn: Optional[StepFn] = None
            if config is not None:
                from omvqvae.models.vqvae import OmicsVQVAE

                model = OmicsVQVAE(loader.vocabulary.n_genes, **config.model_kwargs())
                step_fn = make_train_step_fn(
                    model,
                    grad_clip_norm=config.grad_clip_norm,
                    lr=config.lr,
                    device=device,
                )
            return measure_stream_throughput(
                loader,
                max_batches=max_batches,
                warmup_batches=warmup_batches,
                step_fn=step_fn,
                label=label,
            )
        finally:
            loader.close()
