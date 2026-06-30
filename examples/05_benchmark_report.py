"""
Example 5 — generate the OQAE benchmark report.

This runs the PR #9 *empirical* sweep: it builds a larger offline synthetic
fixture (more cells / genes / programs than example 4), runs the full config
grid from :func:`omvqvae.benchmark.default_report_configs` (raw-count + NB vs
ZINB vs log-normalized + Gaussian, plus codebook sweeps), and writes a Markdown
report interpreting reconstruction quality, codebook utilization, and downstream
separability.

It runs offline (a couple of minutes on CPU) and is the source of truth for the
committed [`docs/benchmark_report.md`](../docs/benchmark_report.md): re-run it to
refresh that report after a change to the harness or model.

Run::

    python examples/05_benchmark_report.py [output.md]
"""

from __future__ import annotations

import sys
from pathlib import Path

from omvqvae.benchmark import (
    default_report_configs,
    generate_report,
    make_benchmark_fixture,
)

#: Where the committed report lives, relative to the repository root.
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "benchmark_report.md"


def main(output_path: "Path | None" = None) -> str:
    """
    Generate the benchmark report and write it to ``output_path``.

    Parameters
    ----------
    output_path : pathlib.Path, optional
        Destination for the Markdown report. Defaults to
        ``docs/benchmark_report.md``.

    Returns
    -------
    str
        The rendered Markdown report (also written to ``output_path``).
    """
    fixture = make_benchmark_fixture()
    configs = default_report_configs()
    report = generate_report(fixture, configs)

    destination = DEFAULT_OUTPUT if output_path is None else output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    print(f"Wrote benchmark report to {destination}")
    return report


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(target)
