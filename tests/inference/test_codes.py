"""
Tests for the discrete-code inference API (:mod:`omvqvae.inference.codes`).

Offline and synthetic: the ``encode → decode`` round-trip, the codes/size-factor
formats, alignment of local AnnData, eval-mode side-effect freedom, and the
validation/error paths. Also covers the new code→vector inverse helpers on the
quantizer and model.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
import torch
from anndata import AnnData
from scipy import sparse

from omvqvae.data.dataset import GeneVocabulary
from omvqvae.hf_utils import LoadedModel
from omvqvae.inference import (
    EncodedCells,
    decode,
    decode_to_params,
    encode,
    encode_anndata,
)
from omvqvae.layers.residual_vq import ResidualVQ
from omvqvae.models.vqvae import OmicsVQVAE

GENE_IDS = ["ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5"]


def _model(seed: int = 0, *, likelihood: str = "nb") -> OmicsVQVAE:
    torch.manual_seed(seed)
    return OmicsVQVAE(
        len(GENE_IDS),
        n_latent=6,
        hidden_dims=(16,),
        likelihood=likelihood,
        codebook_size=8,
        n_codebooks=3,
    )


def _counts(n_cells: int = 7, *, seed: int = 1) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=3.0, size=(n_cells, len(GENE_IDS))).astype(np.float32)
    return torch.from_numpy(counts)


def _make_anndata(
    gene_ids: List[str], *, n_cells: int = 5, seed: int = 2, sparse_x: bool = True
) -> AnnData:
    import pandas as pd

    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=2.0, size=(n_cells, len(gene_ids))).astype(np.float32)
    x: object = sparse.csr_matrix(counts) if sparse_x else counts
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=list(gene_ids))
    return AnnData(X=x, obs=obs, var=var)


# --------------------------------------------------------------------------- #
# Layer / model inverse helpers
# --------------------------------------------------------------------------- #
def test_vector_quantizer_lookup_matches_forward() -> None:
    model = _model()
    counts = _counts()
    model.eval()
    latent = model.encode(counts)
    vq = model.quantize(latent)
    # The summed quantized vector reconstructed from indices equals the forward
    # pass' quantized output.
    recon = model.rvq.lookup(vq.indices)
    assert torch.allclose(recon, vq.quantized, atol=1e-6)


def test_residual_vq_lookup_rejects_wrong_levels() -> None:
    rvq = ResidualVQ(codebook_size=4, embedding_dim=3, n_codebooks=2)
    with pytest.raises(ValueError):
        rvq.lookup(torch.zeros((5, 3), dtype=torch.long))


def test_model_decode_codes_matches_expected_counts() -> None:
    model = _model()
    counts = _counts()
    sf = counts.sum(dim=1)
    model.eval()
    codes = model.encode_codes(counts)
    quantized = model.rvq.lookup(codes)
    direct = model.expected_counts(quantized, sf)
    via_codes = model.decode_codes(codes, sf)
    assert torch.allclose(direct, via_codes, atol=1e-6)
    params = model.codes_to_params(codes, sf)
    # px_scale/px_rate are per-cell-per-gene; theta is gene-wise.
    assert params["px_rate"].shape == counts.shape


# --------------------------------------------------------------------------- #
# encode
# --------------------------------------------------------------------------- #
def test_encode_shapes_and_dtypes() -> None:
    model = _model()
    counts = _counts(n_cells=7)
    enc = encode(model, counts)
    assert isinstance(enc, EncodedCells)
    assert enc.codes.shape == (7, model.n_codebooks)
    assert enc.codes.dtype == torch.long
    assert enc.size_factors.shape == (7,)
    assert enc.latent.shape == (7, model.n_latent)
    assert len(enc) == 7
    # Default size factors are the per-cell totals ("total" mode).
    assert torch.allclose(enc.size_factors, counts.sum(dim=1))


def test_encode_matches_model_encode_codes() -> None:
    model = _model()
    counts = _counts()
    model.eval()
    expected = model.encode_codes(counts)
    enc = encode(model, counts)
    assert torch.equal(enc.codes, expected)


def test_encode_batched_equals_unbatched() -> None:
    model = _model()
    counts = _counts(n_cells=10)
    full = encode(model, counts, batch_size=100)
    chunked = encode(model, counts, batch_size=3)
    assert torch.equal(full.codes, chunked.codes)
    assert torch.allclose(full.latent, chunked.latent, atol=1e-6)


def test_encode_accepts_numpy_and_sparse() -> None:
    model = _model()
    counts = _counts()
    from_np = encode(model, counts.numpy())
    from_sparse = encode(model, sparse.csr_matrix(counts.numpy()))
    assert torch.equal(from_np.codes, from_sparse.codes)


def test_encode_explicit_size_factors() -> None:
    model = _model()
    counts = _counts(n_cells=4)
    sf = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    enc = encode(model, counts, size_factors=sf)
    assert torch.allclose(enc.size_factors, torch.from_numpy(sf))


def test_encode_size_factor_tensor() -> None:
    model = _model()
    counts = _counts(n_cells=3)
    sf = torch.tensor([5.0, 6.0, 7.0])
    enc = encode(model, counts, size_factors=sf)
    assert torch.allclose(enc.size_factors, sf)


def test_encode_empty_input() -> None:
    model = _model()
    enc = encode(model, torch.zeros((0, len(GENE_IDS))))
    assert enc.codes.shape == (0, model.n_codebooks)
    assert enc.size_factors.shape == (0,)


def test_encode_gene_mismatch_raises() -> None:
    model = _model()
    with pytest.raises(ValueError, match="genes"):
        encode(model, torch.zeros((2, len(GENE_IDS) + 1)))


def test_encode_non_2d_raises() -> None:
    model = _model()
    with pytest.raises(ValueError, match="2-D"):
        encode(model, torch.zeros((2, 3, 4)))


def test_encode_wrong_size_factor_length_raises() -> None:
    model = _model()
    counts = _counts(n_cells=4)
    with pytest.raises(ValueError, match="length"):
        encode(model, counts, size_factors=np.ones(3, dtype=np.float32))


def test_encode_does_not_mutate_training_mode() -> None:
    model = _model()
    model.train()
    quantizer = model.rvq.quantizers[0]
    codebook_before = quantizer.embedding.clone()  # type: ignore[union-attr]
    encode(model, _counts())
    # Model is left in training mode and EMA codebooks are untouched.
    assert model.training
    assert torch.equal(quantizer.embedding, codebook_before)  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# decode + round-trip
# --------------------------------------------------------------------------- #
def test_round_trip_encode_decode() -> None:
    model = _model()
    counts = _counts(n_cells=6)
    enc = encode(model, counts)
    recon = decode(model, enc)
    assert recon.shape == counts.shape
    assert torch.isfinite(recon).all()
    # NB mean is non-negative.
    assert (recon >= 0).all()


def test_decode_held_out_cells_round_trip() -> None:
    # Held-out cells: encode then decode reproduces the same codes when re-encoded
    # from the decoder's mean is not expected, but decoding the stored codes is
    # deterministic and depth-appropriate.
    model = _model()
    counts = _counts(n_cells=8, seed=99)
    enc = encode(model, counts)
    recon_a = decode(model, enc.codes, enc.size_factors)
    recon_b = decode(model, enc)
    assert torch.allclose(recon_a, recon_b, atol=1e-6)


def test_decode_batched_equals_unbatched() -> None:
    model = _model()
    counts = _counts(n_cells=11)
    enc = encode(model, counts)
    full = decode(model, enc, batch_size=100)
    chunked = decode(model, enc, batch_size=4)
    assert torch.allclose(full, chunked, atol=1e-6)


def test_decode_accepts_numpy_codes() -> None:
    model = _model()
    counts = _counts(n_cells=5)
    enc = encode(model, counts)
    recon = decode(model, enc.codes.numpy(), enc.size_factors.numpy())
    assert recon.shape == counts.shape


def test_decode_size_factor_scales_counts() -> None:
    # Larger size factor → larger expected counts for the same NB code.
    model = _model()
    counts = _counts(n_cells=3)
    enc = encode(model, counts)
    small = decode(model, enc.codes, torch.full((3,), 10.0))
    large = decode(model, enc.codes, torch.full((3,), 100.0))
    assert large.sum() > small.sum()


def test_decode_override_size_factors_on_bundle() -> None:
    model = _model()
    counts = _counts(n_cells=3)
    enc = encode(model, counts)
    override = torch.full((3,), 50.0)
    recon = decode(model, enc, override)
    direct = model.decode_codes(enc.codes, override)
    assert torch.allclose(recon, direct, atol=1e-6)


def test_decode_missing_size_factors_raises() -> None:
    model = _model()
    with pytest.raises(ValueError, match="size_factors is required"):
        decode(model, torch.zeros((2, model.n_codebooks), dtype=torch.long))


def test_decode_non_2d_codes_raises() -> None:
    model = _model()
    with pytest.raises(ValueError, match="2-D"):
        decode(model, torch.zeros((2, 3, model.n_codebooks), dtype=torch.long))


def test_decode_wrong_level_count_raises() -> None:
    model = _model()
    bad = torch.zeros((2, model.n_codebooks + 1), dtype=torch.long)
    with pytest.raises(ValueError, match="levels"):
        decode(model, bad, torch.ones(2))


def test_decode_size_factor_mismatch_raises() -> None:
    model = _model()
    codes = torch.zeros((3, model.n_codebooks), dtype=torch.long)
    with pytest.raises(ValueError, match="length"):
        decode(model, codes, torch.ones(2))


def test_decode_to_params_returns_head_parameters() -> None:
    model = _model(likelihood="zinb")
    counts = _counts(n_cells=4)
    enc = encode(model, counts)
    params = decode_to_params(model, enc)
    assert set(params)
    # The per-cell-per-gene mean is present with the expected shape.
    assert params["px_rate"].shape == (4, len(GENE_IDS))


# --------------------------------------------------------------------------- #
# encode_anndata
# --------------------------------------------------------------------------- #
def _loaded(model: OmicsVQVAE) -> LoadedModel:
    vocab = GeneVocabulary("homo_sapiens", GENE_IDS)
    return LoadedModel(model=model, vocabulary=vocab)


def test_encode_anndata_round_trip() -> None:
    model = _model()
    adata = _make_anndata(GENE_IDS, n_cells=5)
    enc = encode_anndata(_loaded(model), adata)
    assert enc.codes.shape == (5, model.n_codebooks)
    recon = decode(model, enc)
    assert recon.shape == (5, len(GENE_IDS))


def test_encode_anndata_aligns_reordered_and_missing_genes() -> None:
    model = _model()
    # AnnData with a permuted subset + an extra unknown gene; alignment should
    # zero-fill the missing reference gene and drop the extra.
    file_genes = ["ENSG3", "ENSG1", "EXTRA", "ENSG2", "ENSG4"]
    adata = _make_anndata(file_genes, n_cells=4)
    enc = encode_anndata(_loaded(model), adata)
    assert enc.codes.shape == (4, model.n_codebooks)
    # Re-encoding the aligned dense counts directly gives identical codes.
    from omvqvae.data.dataset import align_to_reference

    aligned = align_to_reference(adata.X, file_genes, _loaded(model).vocabulary)
    direct = encode(model, aligned)
    assert torch.equal(enc.codes, direct.codes)
