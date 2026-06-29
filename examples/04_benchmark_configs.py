"""
Example 4 — benchmark likelihood / codebook configurations.

This is the PR #9 benchmarking harness in miniature: train several tiny
:class:`~omvqvae.models.vqvae.OmicsVQVAE` models — differing in reconstruction
likelihood (NB vs Gaussian) and codebook configuration (``n_codebooks`` /
``codebook_size``) — on the *same* synthetic data, then print a comparison
table of reconstruction quality, codebook utilization, and downstream
separability against the known cell "programs".

It runs offline in seconds; for real sweeps, swap the synthetic loader for a
local-AnnData ``DataLoader`` (example 1) or a Census stream (example 3) — the
:func:`omvqvae.benchmark.run_suite` contract is identical.

Run::

    python examples/04_benchmark_configs.py
"""

from __future__ import annotations

from typing import List

from synthetic_data import make_synthetic_anndata

from omvqvae.benchmark import BenchmarkConfig, format_results_table, run_suite
from omvqvae.data import GeneVocabulary, build_anndata_dataloader, extract_counts


def main() -> str:
    """
    Benchmark a few configs on synthetic data and return the Markdown table.

    Returns
    -------
    str
        The rendered comparison table (also printed to stdout).
    """
    organism = "homo_sapiens"

    # 1. Synthetic raw counts with latent "programs" (the separability target).
    adata = make_synthetic_anndata(
        n_cells=256, n_genes=40, n_programs=4, organism=organism
    )
    counts, gene_ids = extract_counts(adata)
    vocabulary = GeneVocabulary(organism, gene_ids)
    labels = list(adata.obs["program"])

    # 2. One shared, re-iterable training source (same Minibatch contract the
    #    Census loader yields). Every config trains on this.
    loader = build_anndata_dataloader(
        adata, vocabulary, batch_size=64, shuffle=True, batch_key="batch"
    )

    # 3. The sweep: NB vs Gaussian, and a codebook-capacity contrast.
    configs: List[BenchmarkConfig] = [
        BenchmarkConfig(
            name="nb-2x64", likelihood="nb", n_codebooks=2, codebook_size=64
        ),
        BenchmarkConfig(
            name="gaussian-2x64",
            likelihood="gaussian",
            n_codebooks=2,
            codebook_size=64,
        ),
        BenchmarkConfig(
            name="nb-1x16", likelihood="nb", n_codebooks=1, codebook_size=16
        ),
        BenchmarkConfig(
            name="nb-4x64", likelihood="nb", n_codebooks=4, codebook_size=64
        ),
    ]

    # 4. Train + evaluate every config and tabulate the comparison.
    results = run_suite(
        configs,
        loader,
        n_genes=vocabulary.n_genes,
        eval_counts=counts,
        eval_labels=labels,
    )
    table = format_results_table(results)
    print(table)
    return table


if __name__ == "__main__":
    main()
