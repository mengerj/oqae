"""Tests for :mod:`omvqvae.benchmark.baselines` (the LatentModel comparison).

The OQAE adapter and the comparison helper are exercised fully offline on tiny
synthetic data. The scVI adapter needs the optional ``scvi-tools`` dependency, so
its happy path is guarded with :func:`pytest.importorskip` and marked
``network``/slow; the missing-dependency error path is tested deterministically
by hiding ``scvi`` from the import system.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
import pytest

from omvqvae.benchmark import (
    BenchmarkConfig,
    LatentModel,
    LatentModelReport,
    OmvqvaeLatentModel,
    ScviLatentModel,
    compare_latent_models,
    format_latent_comparison,
    latent_comparison_to_dicts,
)


def _program_counts(
    *, n_cells: int = 48, n_genes: int = 12, n_programs: int = 3, seed: int = 0
) -> tuple[np.ndarray, List[str], List[str]]:
    """A tiny raw-count matrix with latent programs, gene ids, and labels."""
    rng = np.random.default_rng(seed)
    program_rates = rng.gamma(1.5, 1.0, size=(n_programs, n_genes))
    programs = rng.integers(0, n_programs, size=n_cells)
    depths = rng.uniform(0.5, 2.0, size=n_cells)
    counts = rng.poisson(program_rates[programs] * depths[:, None]).astype(np.float32)
    genes = [f"ENSG{i:05d}" for i in range(n_genes)]
    labels = [f"program_{p}" for p in programs]
    return counts, genes, labels


class _FakeLatentModel:
    """A stub :class:`LatentModel` that returns a fixed embedding."""

    def __init__(self, name: str, embedding: np.ndarray) -> None:
        self.name = name
        self._embedding = embedding

    def fit(
        self,
        train_counts: np.ndarray,
        *,
        genes: Sequence[str],
        labels: Optional[Sequence[object]] = None,
    ) -> None:  # pragma: no cover - trivial stub
        pass

    def embed(self, counts: np.ndarray) -> np.ndarray:
        return self._embedding


# --------------------------------------------------------------------------- #
# OmvqvaeLatentModel
# --------------------------------------------------------------------------- #


def test_omvqvae_adapter_requires_exactly_one_source() -> None:
    """Neither / both of config and trained_model is a clear ValueError."""
    with pytest.raises(ValueError, match="exactly one"):
        OmvqvaeLatentModel()
    config = BenchmarkConfig(name="oqae")
    from omvqvae.models.vqvae import OmicsVQVAE

    model = OmicsVQVAE(4)
    with pytest.raises(ValueError, match="exactly one"):
        OmvqvaeLatentModel(config, trained_model=model)


def test_omvqvae_adapter_fits_and_embeds() -> None:
    """A config-driven adapter trains and embeds to (n_cells, n_latent)."""
    counts, genes, labels = _program_counts()
    config = BenchmarkConfig(
        name="oqae-tiny",
        max_epochs=1,
        n_latent=8,
        codebook_size=16,
        n_codebooks=2,
        hidden_dims=(16,),
    )
    adapter = OmvqvaeLatentModel(config, batch_size=16)
    assert adapter.name == "oqae-tiny"
    adapter.fit(counts, genes=genes, labels=labels)
    latent = adapter.embed(counts)
    assert latent.shape == (counts.shape[0], 8)
    assert isinstance(adapter, LatentModel)  # satisfies the protocol


def test_omvqvae_adapter_use_quantized_differs() -> None:
    """The quantized view returns the post-quantization latent."""
    counts, genes, _ = _program_counts()
    config = BenchmarkConfig(
        name="oqae", max_epochs=1, n_latent=8, codebook_size=16, hidden_dims=(16,)
    )
    cont = OmvqvaeLatentModel(config, batch_size=16, use_quantized=False)
    cont.fit(counts, genes=genes)
    quant = OmvqvaeLatentModel(config, name="oqae-q", batch_size=16, use_quantized=True)
    quant.fit(counts, genes=genes)
    assert cont.embed(counts).shape == quant.embed(counts).shape
    # Continuous and quantized latents are not identical (quantization moves z).
    assert not np.allclose(cont.embed(counts), quant.embed(counts))


def test_omvqvae_adapter_embed_before_fit_raises() -> None:
    """Embedding before fit is a clear RuntimeError."""
    adapter = OmvqvaeLatentModel(BenchmarkConfig(name="oqae"))
    counts, _, _ = _program_counts()
    with pytest.raises(RuntimeError, match="before fit"):
        adapter.embed(counts)


def test_omvqvae_adapter_gene_length_mismatch_raises() -> None:
    """A genes/counts width mismatch is caught in fit."""
    counts, genes, _ = _program_counts()
    adapter = OmvqvaeLatentModel(BenchmarkConfig(name="oqae"), batch_size=16)
    with pytest.raises(ValueError, match="genes has length"):
        adapter.fit(counts, genes=genes[:-1])


def test_omvqvae_adapter_rejects_non_2d_counts() -> None:
    """A non-2-D count array is caught before any training."""
    adapter = OmvqvaeLatentModel(BenchmarkConfig(name="oqae"), batch_size=16)
    with pytest.raises(ValueError, match="counts must be 2-D"):
        adapter.fit(np.zeros((10,), dtype=np.float32), genes=["g0"])


def test_omvqvae_adapter_wraps_trained_model() -> None:
    """A pre-trained model can be wrapped; fit only validates the feature size."""
    counts, genes, _ = _program_counts()
    from omvqvae.models.vqvae import OmicsVQVAE

    model = OmicsVQVAE(counts.shape[1], n_latent=8, codebook_size=16)
    adapter = OmvqvaeLatentModel(trained_model=model, name="pretrained")
    assert adapter.name == "pretrained"
    adapter.fit(counts, genes=genes)  # no-op train, just a size check
    assert adapter.embed(counts).shape == (counts.shape[0], 8)


def test_omvqvae_adapter_wrapped_model_size_mismatch_raises() -> None:
    """A wrapped model whose n_genes disagrees with the data is rejected."""
    counts, genes, _ = _program_counts()
    from omvqvae.models.vqvae import OmicsVQVAE

    model = OmicsVQVAE(counts.shape[1] + 1)
    adapter = OmvqvaeLatentModel(trained_model=model)
    with pytest.raises(ValueError, match="wrapped model expects"):
        adapter.fit(counts, genes=genes)


# --------------------------------------------------------------------------- #
# ScviLatentModel — offline error path
# --------------------------------------------------------------------------- #


def test_scvi_adapter_missing_dependency_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without scvi-tools, fit raises a clear, actionable ImportError."""
    import sys

    # Hiding the module makes `import scvi` fail regardless of what's installed.
    monkeypatch.setitem(sys.modules, "scvi", None)
    counts, genes, _ = _program_counts()
    model = ScviLatentModel()
    with pytest.raises(ImportError, match="scvi-tools"):
        model.fit(counts, genes=genes)


def test_scvi_adapter_embed_before_fit_raises() -> None:
    """Embedding before fit is a clear RuntimeError (no scvi needed)."""
    model = ScviLatentModel()
    counts, _, _ = _program_counts()
    with pytest.raises(RuntimeError, match="before fit"):
        model.embed(counts)


# --------------------------------------------------------------------------- #
# compare_latent_models + reporting
# --------------------------------------------------------------------------- #


def _well_separated_embedding(labels: Sequence[object]) -> np.ndarray:
    """Build a latent where each label sits on its own axis (separable)."""
    unique = {value: i for i, value in enumerate(sorted(set(labels)))}
    d = len(unique)
    embedding = np.zeros((len(list(labels)), d), dtype=np.float64)
    for row, value in enumerate(labels):
        embedding[row, unique[value]] = 5.0
    return embedding


def test_compare_latent_models_one_row_per_model() -> None:
    """Each model yields exactly one report; separability is populated."""
    _, _, labels = _program_counts()
    good = _FakeLatentModel("good", _well_separated_embedding(labels))
    rng = np.random.default_rng(0)
    noise = _FakeLatentModel("noise", rng.normal(size=(len(labels), 4)))
    eval_counts = np.zeros((len(labels), 3), dtype=np.float32)

    reports = compare_latent_models(
        [good, noise], eval_counts, labels, compute_clustering=False
    )
    assert [r.name for r in reports] == ["good", "noise"]
    assert all(isinstance(r, LatentModelReport) for r in reports)
    # A perfectly axis-aligned latent separates the labels; noise does not.
    assert reports[0].separability == pytest.approx(1.0)
    assert reports[0].separability > reports[1].separability
    # Clustering columns stay nan when not requested.
    assert math.isnan(reports[0].nmi)


def test_compare_latent_models_batch_key_length_checked() -> None:
    """A batch_key/labels length mismatch is a clear ValueError."""
    _, _, labels = _program_counts()
    model = _FakeLatentModel("m", _well_separated_embedding(labels))
    eval_counts = np.zeros((len(labels), 3), dtype=np.float32)
    with pytest.raises(ValueError, match="batch_key has length"):
        compare_latent_models(
            [model],
            eval_counts,
            labels,
            batch_key=labels[:-1],
            compute_clustering=False,
        )


def test_compare_latent_models_row_count_mismatch_raises() -> None:
    """An embedding with the wrong number of cells is rejected."""
    _, _, labels = _program_counts()
    wrong = _FakeLatentModel("wrong", np.zeros((len(labels) - 1, 4)))
    eval_counts = np.zeros((len(labels), 3), dtype=np.float32)
    with pytest.raises(ValueError, match="embedded"):
        compare_latent_models([wrong], eval_counts, labels, compute_clustering=False)


def test_compare_latent_models_non_2d_embedding_raises() -> None:
    """A non-2-D embedding is a clear ValueError."""
    _, _, labels = _program_counts()
    bad = _FakeLatentModel("bad", np.zeros((len(labels),)))
    eval_counts = np.zeros((len(labels), 3), dtype=np.float32)
    with pytest.raises(ValueError, match="expected 2-D"):
        compare_latent_models([bad], eval_counts, labels, compute_clustering=False)


def test_compare_latent_models_with_clustering() -> None:
    """With the benchmark extra, clustering columns are populated."""
    pytest.importorskip("scib_metrics")
    _, _, labels = _program_counts()
    good = _FakeLatentModel("good", _well_separated_embedding(labels))
    eval_counts = np.zeros((len(labels), 3), dtype=np.float32)
    reports = compare_latent_models(
        [good], eval_counts, labels, compute_clustering=True
    )
    assert not math.isnan(reports[0].nmi)
    assert reports[0].nmi > 0.9


def test_format_and_dicts_empty() -> None:
    """Empty reports render a placeholder and no rows."""
    assert format_latent_comparison([]) == "(no models)"
    assert latent_comparison_to_dicts([]) == []


def test_format_latent_comparison_columns() -> None:
    """The clustering columns show only when some report computed them."""
    core = [LatentModelReport(name="a", n_latent=8, separability=0.9)]
    table = format_latent_comparison(core)
    assert table.startswith("| model")
    assert "nmi" not in table

    with_clustering = [
        LatentModelReport(name="a", n_latent=8, separability=0.9, nmi=0.8, ari=0.7)
    ]
    table2 = format_latent_comparison(with_clustering)
    assert "nmi" in table2
    assert "ct_asw" in table2


def test_latent_comparison_to_dicts_roundtrips_fields() -> None:
    """Flattening preserves every scalar field."""
    reports = [
        LatentModelReport(
            name="m", n_latent=10, separability=0.5, nmi=0.4, ari=0.3, cell_type_asw=0.6
        )
    ]
    rows = latent_comparison_to_dicts(reports)
    assert rows == [
        {
            "name": "m",
            "n_latent": 10,
            "separability": 0.5,
            "nmi": 0.4,
            "ari": 0.3,
            "cell_type_asw": 0.6,
        }
    ]


# --------------------------------------------------------------------------- #
# scVI integration — needs the optional dependency; skipped by default
# --------------------------------------------------------------------------- #


@pytest.mark.network
def test_scvi_adapter_end_to_end() -> None:  # pragma: no cover - needs scvi-tools
    """ScviLatentModel trains and embeds; compare yields a row per model."""
    pytest.importorskip("scvi")
    counts, genes, labels = _program_counts(n_cells=64, n_genes=20)
    scvi_model = ScviLatentModel(name="scVI", n_latent=5, max_epochs=2)
    scvi_model.fit(counts, genes=genes)
    latent = scvi_model.embed(counts)
    assert latent.shape == (counts.shape[0], 5)

    config = BenchmarkConfig(
        name="oqae", max_epochs=1, n_latent=8, codebook_size=16, hidden_dims=(16,)
    )
    oqae = OmvqvaeLatentModel(config, batch_size=16)
    oqae.fit(counts, genes=genes)

    reports = compare_latent_models(
        [oqae, scvi_model], counts, labels, compute_clustering=False
    )
    assert [r.name for r in reports] == ["oqae", "scVI"]
