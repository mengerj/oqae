"""Offline tests for the residual vector-quantization layer."""

from __future__ import annotations

import pytest
import torch

from omvqvae.layers import (
    QuantizerOutput,
    ResidualVQ,
    ResidualVQOutput,
    VectorQuantizer,
)


def test_vector_quantizer_shapes_and_dtypes() -> None:
    torch.manual_seed(0)
    vq = VectorQuantizer(codebook_size=16, embedding_dim=8)
    inputs = torch.randn(5, 8)
    out = vq(inputs)

    assert isinstance(out, QuantizerOutput)
    assert out.quantized.shape == inputs.shape
    assert out.quantized.dtype == inputs.dtype
    assert out.indices.shape == (5,)
    assert out.indices.dtype == torch.int64
    assert out.indices.min() >= 0 and out.indices.max() < 16
    for scalar in (out.loss, out.commitment_loss, out.codebook_loss, out.perplexity):
        assert scalar.shape == ()


def test_vector_quantizer_preserves_leading_dims() -> None:
    vq = VectorQuantizer(codebook_size=8, embedding_dim=4)
    inputs = torch.randn(3, 7, 4)
    out = vq(inputs)
    assert out.quantized.shape == (3, 7, 4)
    assert out.indices.shape == (3, 7)


def test_straight_through_estimator_passes_gradient() -> None:
    vq = VectorQuantizer(codebook_size=8, embedding_dim=4, ema=True)
    vq.eval()  # disable EMA/reset so only the STE path runs
    inputs = torch.randn(6, 4, requires_grad=True)
    out = vq(inputs)
    out.quantized.sum().backward()
    assert inputs.grad is not None
    # The straight-through estimator copies an identity gradient.
    torch.testing.assert_close(inputs.grad, torch.ones_like(inputs))


def test_non_ema_codebook_is_a_trained_parameter() -> None:
    vq = VectorQuantizer(codebook_size=8, embedding_dim=4, ema=False)
    assert isinstance(vq.embedding, torch.nn.Parameter)
    params = dict(vq.named_parameters())
    assert "embedding" in params

    inputs = torch.randn(6, 4)
    out = vq(inputs)
    # Non-EMA codebook is pulled toward the encoder via a non-zero codebook loss.
    assert out.codebook_loss.item() > 0.0
    out.loss.backward()
    assert vq.embedding.grad is not None


def test_ema_codebook_is_a_buffer_with_zero_codebook_loss() -> None:
    vq = VectorQuantizer(codebook_size=8, embedding_dim=4, ema=True)
    buffers = dict(vq.named_buffers())
    assert "embedding" in buffers
    assert "embedding" not in dict(vq.named_parameters())

    out = vq(torch.randn(6, 4))
    assert out.codebook_loss.item() == 0.0


def test_ema_update_moves_codebook_in_training() -> None:
    torch.manual_seed(0)
    vq = VectorQuantizer(codebook_size=8, embedding_dim=4, ema=True)
    vq.train()
    before = vq.embedding.clone()
    for _ in range(5):
        vq(torch.randn(32, 4))
    assert not torch.allclose(before, vq.embedding)


def test_dead_code_reset_runs_and_updates_codebook() -> None:
    torch.manual_seed(0)
    vq = VectorQuantizer(
        codebook_size=64,
        embedding_dim=4,
        ema=True,
        reset_dead_codes=True,
        dead_code_threshold=1.0,
    )
    vq.train()
    before = vq.embedding.clone()
    # A tiny batch cannot touch all 64 codes, so some are reset.
    vq(torch.randn(4, 4))
    assert not torch.allclose(before, vq.embedding)
    assert torch.all(vq.cluster_size >= 0.0)


def test_dead_code_reset_noop_when_no_dead_codes() -> None:
    vq = VectorQuantizer(
        codebook_size=8,
        embedding_dim=4,
        ema=False,
        reset_dead_codes=True,
        dead_code_threshold=0.0,  # nothing is ever below zero usage
    )
    vq.train()
    before = vq.embedding.clone()
    vq(torch.randn(4, 4))
    # No code qualifies as dead, so the codebook is untouched by the reset path.
    torch.testing.assert_close(vq.embedding, before)


def test_perplexity_is_non_trivial_with_diverse_inputs() -> None:
    torch.manual_seed(0)
    vq = VectorQuantizer(codebook_size=32, embedding_dim=8, ema=False)
    vq.eval()
    # Many distinct inputs should spread across several codes.
    out = vq(torch.randn(256, 8))
    assert out.perplexity.item() > 1.0
    assert 0.0 < out.usage.item() <= 1.0


def test_vector_quantizer_rejects_bad_init_args() -> None:
    with pytest.raises(ValueError, match="codebook_size"):
        VectorQuantizer(codebook_size=0, embedding_dim=4)
    with pytest.raises(ValueError, match="embedding_dim"):
        VectorQuantizer(codebook_size=4, embedding_dim=0)
    with pytest.raises(ValueError, match="ema_decay"):
        VectorQuantizer(codebook_size=4, embedding_dim=4, ema_decay=1.0)


def test_vector_quantizer_rejects_wrong_input_dim() -> None:
    vq = VectorQuantizer(codebook_size=8, embedding_dim=4)
    with pytest.raises(ValueError, match="embedding_dim"):
        vq(torch.randn(3, 5))


def test_residual_vq_shapes_and_aggregation() -> None:
    torch.manual_seed(0)
    rvq = ResidualVQ(codebook_size=16, embedding_dim=8, n_codebooks=3)
    inputs = torch.randn(10, 8)
    out = rvq(inputs)

    assert isinstance(out, ResidualVQOutput)
    assert out.quantized.shape == inputs.shape
    assert out.indices.shape == (10, 3)
    assert out.indices.dtype == torch.int64
    assert out.perplexities.shape == (3,)
    assert out.usages.shape == (3,)
    # Mean perplexity equals the mean of the per-level perplexities.
    torch.testing.assert_close(out.perplexity, out.perplexities.mean())


def test_residual_vq_reduces_residual_norm() -> None:
    torch.manual_seed(0)
    # With more levels the summed quantization should track the input better.
    inputs = torch.randn(64, 8)
    one = ResidualVQ(codebook_size=64, embedding_dim=8, n_codebooks=1, ema=False)
    four = ResidualVQ(codebook_size=64, embedding_dim=8, n_codebooks=4, ema=False)
    err_one = (inputs - one(inputs).quantized).pow(2).mean()
    err_four = (inputs - four(inputs).quantized).pow(2).mean()
    assert err_four < err_one


def test_residual_vq_gradient_flows_to_input() -> None:
    rvq = ResidualVQ(codebook_size=16, embedding_dim=8, n_codebooks=2, ema=True)
    rvq.eval()
    inputs = torch.randn(5, 8, requires_grad=True)
    out = rvq(inputs)
    (out.quantized.sum() + out.loss).backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_residual_vq_rejects_bad_n_codebooks() -> None:
    with pytest.raises(ValueError, match="n_codebooks"):
        ResidualVQ(codebook_size=8, embedding_dim=4, n_codebooks=0)
