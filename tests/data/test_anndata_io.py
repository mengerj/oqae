"""Tests for local AnnData loading and the DataLoader factory."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from omvqvae.data.anndata_io import (
    build_anndata_dataloader,
    extract_counts,
    load_anndata,
)
from omvqvae.data.dataset import GeneVocabulary, Minibatch

from .conftest import make_anndata


def test_load_anndata_h5ad_roundtrip(tmp_path: Path, human_gene_ids: List[str]) -> None:
    adata = make_anndata(gene_ids=human_gene_ids)
    path = tmp_path / "data.h5ad"
    adata.write_h5ad(path)
    loaded = load_anndata(path)
    assert list(loaded.var_names) == human_gene_ids
    assert loaded.n_obs == adata.n_obs


def test_load_anndata_zarr_roundtrip(tmp_path: Path, human_gene_ids: List[str]) -> None:
    adata = make_anndata(gene_ids=human_gene_ids, sparse_x=False)
    path = tmp_path / "data.zarr"
    adata.write_zarr(path)
    loaded = load_anndata(path)
    assert list(loaded.var_names) == human_gene_ids


def test_load_anndata_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_anndata(tmp_path / "nope.h5ad")


def test_load_anndata_bad_extension(tmp_path: Path) -> None:
    bad = tmp_path / "data.txt"
    bad.write_text("not anndata")
    with pytest.raises(ValueError, match="Unsupported"):
        load_anndata(bad)


def test_extract_counts_var_names_and_layer(human_gene_ids: List[str]) -> None:
    adata = make_anndata(gene_ids=human_gene_ids, sparse_x=False)
    adata.layers["counts"] = adata.X.copy()
    matrix, gene_ids = extract_counts(adata, layer="counts")
    assert gene_ids == human_gene_ids
    assert matrix.shape == (adata.n_obs, len(human_gene_ids))


def test_extract_counts_with_var_key(human_gene_ids: List[str]) -> None:
    adata = make_anndata(gene_ids=human_gene_ids, sparse_x=False)
    adata.var["gene_symbol"] = [f"SYM_{g}" for g in human_gene_ids]
    _, gene_ids = extract_counts(adata, var_key="gene_symbol")
    assert gene_ids == [f"SYM_{g}" for g in human_gene_ids]


def test_build_dataloader_from_in_memory_anndata(
    human_gene_ids: List[str],
) -> None:
    adata = make_anndata(gene_ids=human_gene_ids, n_cells=6)
    vocab = GeneVocabulary("homo_sapiens", human_gene_ids)
    loader = build_anndata_dataloader(adata, vocab, batch_size=4, batch_key="batch")
    batches = list(loader)
    assert sum(len(b) for b in batches) == 6
    first = batches[0]
    assert isinstance(first, Minibatch)
    assert first.counts.shape[1] == vocab.n_genes
    assert first.covariates["organism"][0] == "homo_sapiens"
    assert set(first.covariates["batch"]) <= {"b0", "b1"}


def test_build_dataloader_aligns_to_reference(human_gene_ids: List[str]) -> None:
    # Reference has an extra gene the source lacks -> zero-filled column.
    reference = human_gene_ids + ["ENSG_EXTRA"]
    adata = make_anndata(gene_ids=human_gene_ids, n_cells=3, sparse_x=False)
    vocab = GeneVocabulary("homo_sapiens", reference)
    loader = build_anndata_dataloader(adata, vocab, batch_size=8)
    batch = next(iter(loader))
    assert batch.counts.shape == (3, len(reference))
    # The extra reference gene (last column) is zero for every cell.
    assert float(batch.counts[:, -1].abs().sum()) == 0.0


def test_build_dataloader_from_path(tmp_path: Path, human_gene_ids: List[str]) -> None:
    adata = make_anndata(gene_ids=human_gene_ids)
    path = tmp_path / "data.h5ad"
    adata.write_h5ad(path)
    vocab = GeneVocabulary("homo_sapiens", human_gene_ids)
    loader = build_anndata_dataloader(path, vocab, batch_size=2)
    assert sum(len(b) for b in loader) == adata.n_obs


def test_human_and_mouse_share_same_api(
    human_gene_ids: List[str], mouse_gene_ids: List[str]
) -> None:
    """The exit criterion: human and mouse iterate through the same API."""
    human = make_anndata(gene_ids=human_gene_ids, n_cells=5)
    mouse = make_anndata(gene_ids=mouse_gene_ids, n_cells=4, seed=1)
    for adata, organism, genes in [
        (human, "homo_sapiens", human_gene_ids),
        (mouse, "mus_musculus", mouse_gene_ids),
    ]:
        vocab = GeneVocabulary(organism, genes)
        loader = build_anndata_dataloader(adata, vocab, batch_size=3)
        batch = next(iter(loader))
        assert isinstance(batch, Minibatch)
        assert batch.counts.shape[1] == len(genes)
        assert batch.covariates["organism"][0] == organism
