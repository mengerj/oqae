"""
Example 6 — benchmark CELLxGENE Census streaming throughput.

This profiles the *data path* rather than model quality: how fast cells stream
from the Census (cells/s, batches/s, time-to-first-batch), and — with a model in
the loop — the end-to-end streaming-plus-training rate. It reuses the same
:class:`~omvqvae.benchmark.harness.BenchmarkConfig` model contract as the offline
benchmark report (examples 4 and 5), so the throughput numbers line up with the
configurations benchmarked there.

.. note::

   **This example requires network access** to the CELLxGENE Census and is
   therefore *not* run in CI (the offline timing core is unit tested). Run it
   manually::

       python examples/06_census_throughput.py

   Widen the ``obs_value_filter`` / raise ``max_batches`` for a heavier profile.
"""

from __future__ import annotations

from omvqvae.benchmark import (
    BenchmarkConfig,
    benchmark_census_throughput,
    format_throughput_table,
)


def main() -> str:  # pragma: no cover - requires live Census/TileDB-SOMA
    """Profile raw and end-to-end Census streaming throughput for human cells."""
    organism = "homo_sapiens"

    # A narrow slice keeps the example light; widen the filter for a real profile.
    obs_value_filter = (
        "tissue_general == 'blood' and is_primary_data == True "
        "and assay == '10x 3\\' v3'"
    )

    # Raw streaming throughput (no model in the loop).
    raw = benchmark_census_throughput(
        organism,
        obs_value_filter=obs_value_filter,
        batch_size=128,
        max_batches=40,
        warmup_batches=2,
        label="human/raw-stream",
    )

    # End-to-end throughput: the same stream with a model train step per batch.
    config = BenchmarkConfig(name="nb-2x64", likelihood="nb", n_codebooks=2)
    end_to_end = benchmark_census_throughput(
        organism,
        config=config,
        obs_value_filter=obs_value_filter,
        batch_size=128,
        max_batches=40,
        warmup_batches=2,
    )

    table = format_throughput_table([raw, end_to_end])
    print(table)
    return table


if __name__ == "__main__":
    main()
