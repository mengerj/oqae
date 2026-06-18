"""Shared offline fixtures for OQAE data-layer tests."""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse


def make_anndata(
    *,
    n_cells: int = 6,
    gene_ids: List[str],
    seed: int = 0,
    sparse_x: bool = True,
    add_batch: bool = True,
) -> AnnData:
    """Build a tiny synthetic raw-count AnnData for offline tests."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=2.0, size=(n_cells, len(gene_ids))).astype(np.float32)
    x: object = sparse.csr_matrix(counts) if sparse_x else counts
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    if add_batch:
        obs["batch"] = [f"b{i % 2}" for i in range(n_cells)]
    var = pd.DataFrame(index=list(gene_ids))
    return AnnData(X=x, obs=obs, var=var)


@pytest.fixture
def human_gene_ids() -> List[str]:
    """A small ordered human reference gene set."""
    return ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]


@pytest.fixture
def mouse_gene_ids() -> List[str]:
    """A small ordered mouse reference gene set."""
    return ["ENSMUSG1", "ENSMUSG2", "ENSMUSG3"]
