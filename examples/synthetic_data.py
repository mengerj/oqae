"""
Tiny synthetic single-cell data for the OQAE examples.

The examples are meant to run **offline in seconds** — they illustrate the OQAE
API rather than train a useful model — so they all share this helper instead of
downloading anything. It fabricates a small raw-count
:class:`anndata.AnnData` with a handful of latent "cell programs", which gives
the residual VQ a little structure to quantize while staying tiny.

Run any example directly, e.g.::

    python examples/01_train_local_anndata.py

For the real thing, swap this for a local ``.h5ad`` / ``.zarr`` of raw counts
(example 1) or stream from the CELLxGENE Census (example 3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anndata import AnnData


def synthetic_gene_ids(n_genes: int, organism: str = "homo_sapiens") -> List[str]:
    """Return ``n_genes`` ordered, organism-flavoured gene identifiers."""
    prefix = "ENSMUSG" if organism == "mus_musculus" else "ENSG"
    return [f"{prefix}{i:05d}" for i in range(n_genes)]


def make_synthetic_anndata(
    *,
    n_cells: int = 256,
    n_genes: int = 40,
    n_programs: int = 4,
    organism: str = "homo_sapiens",
    seed: int = 0,
) -> "AnnData":
    """
    Build a small synthetic raw-count AnnData with latent structure.

    Each cell is assigned one of ``n_programs`` "programs"; a program is a random
    per-gene rate vector, and counts are Poisson draws from that rate scaled by a
    per-cell sequencing depth. The program id is stored in ``obs["program"]`` so
    examples can sanity-check that similar cells receive similar discrete codes.

    Parameters
    ----------
    n_cells : int, default 256
        Number of cells (rows).
    n_genes : int, default 40
        Number of genes (columns).
    n_programs : int, default 4
        Number of latent expression programs.
    organism : str, default "homo_sapiens"
        Selects the gene-id prefix (``"homo_sapiens"`` / ``"mus_musculus"``).
    seed : int, default 0
        RNG seed for reproducibility.

    Returns
    -------
    anndata.AnnData
        Cells x genes raw counts in ``X`` with ``obs["program"]`` /
        ``obs["batch"]`` and gene ids as ``var_names``.
    """
    import pandas as pd
    from anndata import AnnData  # lazy import keeps `import` cheap

    rng = np.random.default_rng(seed)
    # One random non-negative rate vector per program.
    program_rates = rng.gamma(shape=1.5, scale=1.0, size=(n_programs, n_genes))
    programs = rng.integers(0, n_programs, size=n_cells)
    depths = rng.uniform(0.5, 2.0, size=n_cells)

    rates = program_rates[programs] * depths[:, None]
    counts = rng.poisson(rates).astype(np.float32)

    gene_ids = synthetic_gene_ids(n_genes, organism)
    obs = pd.DataFrame(
        {
            "program": [f"program_{p}" for p in programs],
            "batch": [f"batch_{i % 2}" for i in range(n_cells)],
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=gene_ids)
    return AnnData(X=counts, obs=obs, var=var)
