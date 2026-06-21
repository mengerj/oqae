"""Tests for OQAE reconstruction likelihoods and decoder heads.

Offline and synthetic: the pure log-probability functions are checked against
SciPy reference implementations, and the decoder heads are exercised for shapes,
gradient flow, and their documented invariants.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import stats

from omvqvae.models.likelihoods import (
    LIKELIHOODS,
    GaussianHead,
    NBHead,
    ReconstructionHead,
    ZINBHead,
    build_reconstruction_head,
    log_gaussian,
    log_nb_positive,
    log_zinb_positive,
)


# --------------------------------------------------------------------------- #
# Pure log-probability functions
# --------------------------------------------------------------------------- #
def test_log_nb_positive_matches_scipy() -> None:
    x = torch.tensor([0.0, 1.0, 5.0, 12.0])
    mu = torch.tensor([2.0, 3.0, 4.0, 10.0])
    theta = torch.tensor([1.5, 2.0, 0.7, 5.0])

    got = log_nb_positive(x, mu, theta).numpy()
    # SciPy nbinom: n = theta (failures), p = theta / (theta + mu).
    p = (theta / (theta + mu)).numpy()
    want = stats.nbinom.logpmf(x.numpy(), n=theta.numpy(), p=p)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_log_zinb_reduces_to_nb_when_no_inflation() -> None:
    x = torch.tensor([0.0, 1.0, 4.0, 9.0])
    mu = torch.tensor([2.0, 3.0, 4.0, 8.0])
    theta = torch.tensor([1.5, 2.0, 0.7, 3.0])
    # Very negative logits => zero-inflation probability ~ 0 => plain NB.
    zi_logits = torch.full_like(x, -30.0)

    zinb = log_zinb_positive(x, mu, theta, zi_logits)
    nb = log_nb_positive(x, mu, theta)
    np.testing.assert_allclose(zinb.numpy(), nb.numpy(), rtol=1e-5, atol=1e-5)


def test_log_zinb_inflates_zero_probability() -> None:
    x_zero = torch.tensor([0.0])
    mu = torch.tensor([3.0])
    theta = torch.tensor([2.0])
    # Logit 0 => 50% zero inflation: P(0) must exceed the plain-NB P(0).
    zi_logits = torch.tensor([0.0])

    p_zero_zinb = float(torch.exp(log_zinb_positive(x_zero, mu, theta, zi_logits)))
    p_zero_nb = float(torch.exp(log_nb_positive(x_zero, mu, theta)))
    assert p_zero_zinb > p_zero_nb


def test_log_gaussian_matches_scipy() -> None:
    x = torch.tensor([0.0, 1.0, -2.0, 3.5])
    mean = torch.tensor([0.5, 1.0, -1.0, 3.0])
    log_var = torch.tensor([0.0, -0.5, 1.0, 0.2])

    got = log_gaussian(x, mean, log_var).numpy()
    sigma = np.exp(0.5 * log_var.numpy())
    want = stats.norm.logpdf(x.numpy(), loc=mean.numpy(), scale=sigma)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
# Decoder heads — shared behaviour
# --------------------------------------------------------------------------- #
@pytest.fixture()
def synthetic_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    n_cells, n_hidden, n_genes = 8, 5, 6
    hidden = torch.randn(n_cells, n_hidden)
    counts = torch.randint(0, 20, (n_cells, n_genes)).float()
    size_factor = counts.sum(dim=1)
    return hidden, counts, size_factor


@pytest.mark.parametrize("name", LIKELIHOODS)
def test_reconstruction_loss_is_scalar_and_differentiable(
    name: str,
    synthetic_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    hidden, counts, size_factor = synthetic_batch
    target = counts if name != "gaussian" else torch.log1p(counts)
    head = build_reconstruction_head(name, n_hidden=5, n_genes=6)

    loss = head.reconstruction_loss(hidden, target, size_factor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    grads = [p.grad for p in head.parameters() if p.requires_grad]
    assert grads, "head should have trainable parameters"
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("name", LIKELIHOODS)
def test_reduction_modes(
    name: str,
    synthetic_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    hidden, counts, size_factor = synthetic_batch
    target = counts if name != "gaussian" else torch.log1p(counts)
    head = build_reconstruction_head(name, n_hidden=5, n_genes=6)

    per_cell = head.reconstruction_loss(hidden, target, size_factor, reduction="none")
    assert per_cell.shape == (hidden.shape[0],)

    mean = head.reconstruction_loss(hidden, target, size_factor, reduction="mean")
    summed = head.reconstruction_loss(hidden, target, size_factor, reduction="sum")
    torch.testing.assert_close(mean, per_cell.mean())
    torch.testing.assert_close(summed, per_cell.sum())


@pytest.mark.parametrize("name", LIKELIHOODS)
def test_expected_counts_shape(
    name: str,
    synthetic_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    hidden, counts, size_factor = synthetic_batch
    head = build_reconstruction_head(name, n_hidden=5, n_genes=6)
    mean = head.expected_counts(hidden, size_factor)
    assert mean.shape == counts.shape


def test_invalid_reduction_raises(
    synthetic_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    hidden, counts, size_factor = synthetic_batch
    head = NBHead(n_hidden=5, n_genes=6)
    with pytest.raises(ValueError, match="Unknown reduction"):
        head.reconstruction_loss(hidden, counts, size_factor, reduction="bogus")


# --------------------------------------------------------------------------- #
# Count-head specifics (NB / ZINB)
# --------------------------------------------------------------------------- #
def test_nb_rate_respects_library_size(
    synthetic_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    hidden, _, size_factor = synthetic_batch
    head = NBHead(n_hidden=5, n_genes=6)
    rate = head.expected_counts(hidden, size_factor)
    # px_scale sums to one per cell, so the NB mean sums to the library size.
    torch.testing.assert_close(rate.sum(dim=-1), size_factor)
    assert (rate >= 0).all()


def test_zinb_mean_below_nb_mean(
    synthetic_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    hidden, _, size_factor = synthetic_batch
    torch.manual_seed(1)
    head = ZINBHead(n_hidden=5, n_genes=6)
    # Force a non-trivial dropout probability.
    with torch.no_grad():
        head.zi_decoder.bias.fill_(2.0)
    nb_rate = head._px_rate(hidden, size_factor)
    zinb_mean = head.expected_counts(hidden, size_factor)
    assert (zinb_mean <= nb_rate + 1e-6).all()
    assert (zinb_mean >= 0).all()


def test_zinb_forward_exposes_zi_logits(
    synthetic_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    hidden, counts, size_factor = synthetic_batch
    head = ZINBHead(n_hidden=5, n_genes=6)
    params = head.forward(hidden, size_factor)
    assert set(params) == {"px_scale", "px_rate", "theta", "zi_logits"}
    assert params["zi_logits"].shape == counts.shape
    assert params["theta"].shape == (6,)


def test_nb_forward_params(
    synthetic_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    hidden, counts, size_factor = synthetic_batch
    head = NBHead(n_hidden=5, n_genes=6)
    params = head.forward(hidden, size_factor)
    assert set(params) == {"px_scale", "px_rate", "theta"}
    torch.testing.assert_close(
        params["px_scale"].sum(dim=-1), torch.ones(hidden.shape[0])
    )


def test_gaussian_forward_params(
    synthetic_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    hidden, counts, size_factor = synthetic_batch
    head = GaussianHead(n_hidden=5, n_genes=6)
    params = head.forward(hidden, size_factor)
    assert set(params) == {"mean", "log_var"}
    assert params["mean"].shape == counts.shape
    assert params["log_var"].shape == counts.shape


# --------------------------------------------------------------------------- #
# Construction / validation
# --------------------------------------------------------------------------- #
def test_build_reconstruction_head_types() -> None:
    assert isinstance(build_reconstruction_head("nb", 4, 3), NBHead)
    assert isinstance(build_reconstruction_head("ZINB", 4, 3), ZINBHead)
    assert isinstance(build_reconstruction_head("Gaussian", 4, 3), GaussianHead)
    assert isinstance(build_reconstruction_head("nb", 4, 3), ReconstructionHead)


def test_build_reconstruction_head_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown likelihood"):
        build_reconstruction_head("poisson", 4, 3)


def test_invalid_dimensions_raise() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        NBHead(n_hidden=0, n_genes=3)


def test_invalid_dispersion_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported dispersion"):
        NBHead(n_hidden=4, n_genes=3, dispersion="gene-cell")


def test_leading_dim_preserved() -> None:
    # Heads should support arbitrary leading dimensions (e.g. extra batch axes).
    head = NBHead(n_hidden=5, n_genes=6)
    hidden = torch.randn(2, 3, 5)
    size_factor = torch.rand(2, 3) * 100 + 1
    rate = head.expected_counts(hidden, size_factor)
    assert rate.shape == (2, 3, 6)
    per_cell = head.reconstruction_loss(
        hidden, torch.zeros(2, 3, 6), size_factor, reduction="none"
    )
    assert per_cell.shape == (2, 3)
