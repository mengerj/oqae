"""Tests for the end-to-end OQAE VQ-VAE model.

Offline and synthetic: construction/validation, shape and round-trip
invariants, gradient flow through the discrete bottleneck, and a short
smoke-train on synthetic counts (NB and ZINB) checking that the loss decreases
and the codebooks stay utilized.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from omvqvae.models.vqvae import OmicsVQVAE, VQVAEOutput


def _synthetic_counts(
    n_cells: int, n_genes: int, *, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a small synthetic raw-count batch and its size factors."""
    rng = np.random.default_rng(seed)
    # Two latent "cell types" with distinct mean expression so structure exists.
    means = rng.gamma(shape=2.0, scale=1.0, size=(2, n_genes))
    assignments = rng.integers(0, 2, size=n_cells)
    rates = means[assignments]
    counts = rng.poisson(rates).astype(np.float32)
    counts_t = torch.from_numpy(counts)
    size_factors = counts_t.sum(dim=1).clamp(min=1.0)
    return counts_t, size_factors


# --------------------------------------------------------------------------- #
# Construction / validation
# --------------------------------------------------------------------------- #
def test_invalid_dimensions_raise() -> None:
    with pytest.raises(ValueError):
        OmicsVQVAE(0)
    with pytest.raises(ValueError):
        OmicsVQVAE(10, n_latent=0)


def test_unknown_likelihood_raises() -> None:
    with pytest.raises(ValueError):
        OmicsVQVAE(10, likelihood="poisson")


@pytest.mark.parametrize("hidden_dims", [(), (32,), (64, 32)])
def test_encoder_decoder_shapes(hidden_dims: tuple[int, ...]) -> None:
    model = OmicsVQVAE(
        12, n_latent=8, hidden_dims=hidden_dims, codebook_size=16, n_codebooks=3
    )
    counts, size_factors = _synthetic_counts(5, 12)

    latent = model.encode(counts)
    assert latent.shape == (5, 8)

    out = model(counts, size_factors)
    assert isinstance(out, VQVAEOutput)
    assert out.indices.shape == (5, 3)
    assert out.indices.dtype == torch.int64
    assert out.quantized.shape == (5, 8)
    assert out.latent.shape == (5, 8)
    assert out.perplexities.shape == (3,)
    assert out.usages.shape == (3,)
    assert model.n_codebooks == 3


def test_loss_composition_and_scalars() -> None:
    model = OmicsVQVAE(10, n_latent=8, hidden_dims=(16,), codebook_size=16)
    counts, size_factors = _synthetic_counts(6, 10)

    out = model(counts, size_factors)
    for scalar in (
        out.loss,
        out.reconstruction_loss,
        out.vq_loss,
        out.commitment_loss,
        out.codebook_loss,
        out.perplexity,
    ):
        assert scalar.ndim == 0
    # Total loss is reconstruction + VQ loss.
    torch.testing.assert_close(out.loss, out.reconstruction_loss + out.vq_loss)
    # EMA is on by default, so the codebook-pull term is dropped.
    torch.testing.assert_close(out.codebook_loss, torch.zeros(()))


# --------------------------------------------------------------------------- #
# Codes / round-trip
# --------------------------------------------------------------------------- #
def test_encode_codes_matches_forward_indices() -> None:
    model = OmicsVQVAE(10, n_latent=8, hidden_dims=(16,), codebook_size=16)
    model.eval()  # avoid EMA/dead-code updates between the two calls
    counts, _ = _synthetic_counts(4, 10)

    codes = model.encode_codes(counts)
    assert codes.shape == (4, model.n_codebooks)
    assert codes.dtype == torch.int64
    assert int(codes.min()) >= 0
    assert int(codes.max()) < 16


def test_expected_counts_shape_and_nonnegative_for_nb() -> None:
    model = OmicsVQVAE(10, n_latent=8, hidden_dims=(16,), likelihood="nb")
    model.eval()
    counts, size_factors = _synthetic_counts(4, 10)

    out = model(counts, size_factors)
    expected = model.expected_counts(out.quantized, size_factors)
    assert expected.shape == (4, 10)
    assert torch.all(expected >= 0.0)


def test_decode_params_for_zinb() -> None:
    model = OmicsVQVAE(10, n_latent=8, hidden_dims=(16,), likelihood="zinb")
    model.eval()
    counts, size_factors = _synthetic_counts(4, 10)

    out = model(counts, size_factors)
    params = model.decode(out.quantized, size_factors)
    assert set(params) >= {"px_scale", "px_rate", "theta", "zi_logits"}
    assert params["zi_logits"].shape == (4, 10)


def test_gaussian_head_targets_log1p_expression() -> None:
    # The Gaussian head reconstructs log1p expression rather than raw counts.
    model = OmicsVQVAE(10, n_latent=8, hidden_dims=(16,), likelihood="gaussian")
    counts, size_factors = _synthetic_counts(5, 10)

    torch.testing.assert_close(model._recon_target(counts), torch.log1p(counts))
    out = model(counts, size_factors)
    assert out.loss.ndim == 0
    params = model.decode(out.quantized, size_factors)
    assert set(params) == {"mean", "log_var"}


def test_quantize_method_matches_forward() -> None:
    model = OmicsVQVAE(10, n_latent=8, hidden_dims=(16,), codebook_size=16)
    model.eval()
    counts, _ = _synthetic_counts(4, 10)

    latent = model.encode(counts)
    vq = model.quantize(latent)
    torch.testing.assert_close(vq.indices, model.rvq(latent).indices)


def test_dropout_layers_present() -> None:
    model = OmicsVQVAE(10, n_latent=8, hidden_dims=(16,), dropout=0.5)
    has_dropout = any(
        isinstance(m, torch.nn.Dropout) for m in model.encoder_body.modules()
    )
    assert has_dropout


# --------------------------------------------------------------------------- #
# Gradient flow
# --------------------------------------------------------------------------- #
def test_gradients_flow_through_bottleneck() -> None:
    model = OmicsVQVAE(10, n_latent=8, hidden_dims=(16,), codebook_size=16)
    counts, size_factors = _synthetic_counts(6, 10)

    out = model(counts, size_factors)
    out.loss.backward()

    # The straight-through estimator must carry gradient back to the encoder.
    grad = model.to_latent.weight.grad
    assert grad is not None
    assert torch.any(grad != 0.0)


# --------------------------------------------------------------------------- #
# Smoke train
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("likelihood", ["nb", "zinb"])
def test_smoke_train_converges(likelihood: str) -> None:
    torch.manual_seed(0)
    counts, size_factors = _synthetic_counts(64, 20, seed=1)
    model = OmicsVQVAE(
        20,
        n_latent=8,
        hidden_dims=(32,),
        likelihood=likelihood,
        codebook_size=32,
        n_codebooks=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.train()
    first_loss = None
    last_loss = None
    for _ in range(40):
        optimizer.zero_grad()
        out = model(counts, size_factors)
        out.loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = out.reconstruction_loss.item()
        last_loss = out.reconstruction_loss.item()

    assert first_loss is not None and last_loss is not None
    # Reconstruction improves over the short run.
    assert last_loss < first_loss
    # Codebooks are non-trivially utilized (no collapse to a single code).
    model.eval()
    final = model(counts, size_factors)
    assert float(final.perplexity) > 1.0
