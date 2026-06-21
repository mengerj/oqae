"""
Reconstruction likelihoods and decoder heads for OQAE.

OQAE ingests **raw counts** and reconstructs them with a count likelihood, so
the decoder's final stage is a *reconstruction head* that maps a hidden
representation to the parameters of a per-gene distribution and scores observed
counts under it. This module provides:

- Pure log-probability functions for the three supported likelihoods —
  :func:`log_nb_positive` (Negative Binomial), :func:`log_zinb_positive`
  (Zero-Inflated NB), and :func:`log_gaussian` (Gaussian, for a
  log-normalized/Gaussian alternative). They operate on plain tensors and are
  unit-tested against SciPy.
- :class:`ReconstructionHead` subclasses (:class:`NBHead`, :class:`ZINBHead`,
  :class:`GaussianHead`) — small :class:`torch.nn.Module` heads that decode a
  hidden vector into distribution parameters, compute the per-cell
  reconstruction loss (negative log-likelihood summed over genes), and expose
  the distribution mean for reconstruction/generation.

The count heads follow the scVI parameterization: a softmax over genes gives a
per-cell mean *proportion* (``px_scale``), which is scaled by the observed
library size (the per-cell size factor) to obtain the NB mean ``px_rate``;
dispersion is a learned per-gene parameter. This keeps depth handling inside the
model and the raw count statistics intact (see ``docs/PROJECT_PLAN.md``).

All heads share one interface so the VQ-VAE model (PR #4) and the W&B logging
(PR #5) can swap likelihoods without changing call sites.
"""

from __future__ import annotations

import abc
from typing import Dict

import torch
from torch import Tensor, nn

from omvqvae.utils.logging import get_logger

logger = get_logger(__name__)

#: Numerical-stability floor used inside the count log-likelihoods.
_EPS = 1e-8

#: Registry of available likelihood names (used by ``build_reconstruction_head``).
LIKELIHOODS = ("nb", "zinb", "gaussian")

__all__ = [
    "LIKELIHOODS",
    "log_nb_positive",
    "log_zinb_positive",
    "log_gaussian",
    "ReconstructionHead",
    "NBHead",
    "ZINBHead",
    "GaussianHead",
    "build_reconstruction_head",
]


def log_nb_positive(x: Tensor, mu: Tensor, theta: Tensor, eps: float = _EPS) -> Tensor:
    """
    Element-wise log-probability of a Negative Binomial (mean/dispersion form).

    The distribution is parameterized by its mean ``mu`` and inverse-dispersion
    ``theta`` (the NB ``r``/number-of-failures parameter); the variance is
    ``mu + mu**2 / theta``. This is the scVI ``log_nb_positive`` formulation.

    Parameters
    ----------
    x : torch.Tensor
        Observed counts (non-negative). Broadcastable with ``mu``/``theta``.
    mu : torch.Tensor
        Mean of the distribution (positive).
    theta : torch.Tensor
        Inverse dispersion (positive). Broadcastable with ``mu``.
    eps : float, optional
        Numerical-stability floor added inside logarithms.

    Returns
    -------
    torch.Tensor
        Log-probability with the broadcast shape of the inputs.
    """
    log_theta_mu_eps = torch.log(theta + mu + eps)
    return (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
    )


def log_zinb_positive(
    x: Tensor,
    mu: Tensor,
    theta: Tensor,
    zi_logits: Tensor,
    eps: float = _EPS,
) -> Tensor:
    """
    Element-wise log-probability of a Zero-Inflated Negative Binomial.

    A point mass at zero with probability ``sigmoid(zi_logits)`` is mixed with an
    NB component of mean ``mu`` and inverse-dispersion ``theta`` (scVI's
    ``log_zinb_positive``). This is numerically stable for ``x == 0`` and
    ``x > 0`` via the softplus formulation.

    Parameters
    ----------
    x : torch.Tensor
        Observed counts (non-negative).
    mu : torch.Tensor
        NB-component mean (positive).
    theta : torch.Tensor
        NB-component inverse dispersion (positive).
    zi_logits : torch.Tensor
        Logits of the zero-inflation probability; large negative values recover
        the plain NB.
    eps : float, optional
        Numerical-stability floor added inside logarithms.

    Returns
    -------
    torch.Tensor
        Log-probability with the broadcast shape of the inputs.
    """
    softplus_pi = nn.functional.softplus(-zi_logits)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -zi_logits + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = nn.functional.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type_as(x), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
    )
    mul_case_non_zero = torch.mul((x > eps).type_as(x), case_non_zero)

    return mul_case_zero + mul_case_non_zero


def log_gaussian(x: Tensor, mean: Tensor, log_var: Tensor) -> Tensor:
    """
    Element-wise log-probability of a diagonal Gaussian.

    Used for the log-normalized/Gaussian reconstruction alternative; ``x`` is the
    internally normalized expression (e.g. ``log1p``) rather than raw counts.

    Parameters
    ----------
    x : torch.Tensor
        Observed (normalized) values.
    mean : torch.Tensor
        Predicted mean.
    log_var : torch.Tensor
        Predicted log-variance. Broadcastable with ``mean``.

    Returns
    -------
    torch.Tensor
        Log-probability with the broadcast shape of the inputs.
    """
    return -0.5 * (
        torch.log(torch.tensor(2.0 * torch.pi, dtype=x.dtype, device=x.device))
        + log_var
        + (x - mean) ** 2 / torch.exp(log_var)
    )


def _reduce(values: Tensor, reduction: str) -> Tensor:
    """Reduce a per-cell loss tensor by ``"mean"``, ``"sum"``, or ``"none"``."""
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError(
        f"Unknown reduction {reduction!r}; expected 'mean', 'sum', or 'none'."
    )


class ReconstructionHead(nn.Module, abc.ABC):
    """
    Abstract decoder head mapping a hidden vector to a reconstruction loss.

    Subclasses implement a specific likelihood. The shared contract is:

    - ``forward(hidden, size_factor)`` returns the per-gene distribution
      parameters as a ``dict`` of tensors;
    - ``reconstruction_loss(hidden, target, size_factor)`` returns the negative
      log-likelihood, summed over genes and reduced over cells;
    - ``expected_counts(hidden, size_factor)`` returns the distribution mean.

    Parameters
    ----------
    n_hidden : int
        Dimensionality of the incoming hidden representation.
    n_genes : int
        Number of output genes (the model's feature space size).
    """

    #: Short likelihood identifier (e.g. ``"nb"``), set by each subclass.
    likelihood: str

    def __init__(self, n_hidden: int, n_genes: int) -> None:
        super().__init__()
        if n_hidden < 1 or n_genes < 1:
            raise ValueError("n_hidden and n_genes must be positive.")
        self.n_hidden = n_hidden
        self.n_genes = n_genes

    @abc.abstractmethod
    def forward(self, hidden: Tensor, size_factor: Tensor) -> Dict[str, Tensor]:
        """Return the per-gene distribution parameters for ``hidden``."""
        raise NotImplementedError

    @abc.abstractmethod
    def reconstruction_loss(
        self,
        hidden: Tensor,
        target: Tensor,
        size_factor: Tensor,
        *,
        reduction: str = "mean",
    ) -> Tensor:
        """Return the negative log-likelihood of ``target`` given ``hidden``."""
        raise NotImplementedError

    @abc.abstractmethod
    def expected_counts(self, hidden: Tensor, size_factor: Tensor) -> Tensor:
        """Return the distribution mean (expected counts) for ``hidden``."""
        raise NotImplementedError


class NBHead(ReconstructionHead):
    """
    Negative-Binomial reconstruction head (scVI-style, raw counts).

    A linear layer maps the hidden vector to per-gene logits; a softmax turns
    them into mean proportions ``px_scale`` (summing to one per cell), which are
    scaled by the observed library size (``size_factor``) to obtain the NB mean
    ``px_rate``. Dispersion is a learned per-gene parameter.

    Parameters
    ----------
    n_hidden : int
        Dimensionality of the incoming hidden representation.
    n_genes : int
        Number of output genes.
    dispersion : {"gene"}, default "gene"
        Dispersion parameterization. Only gene-wise dispersion is supported in
        v1 (one ``theta`` per gene, shared across cells).
    """

    likelihood = "nb"

    def __init__(
        self, n_hidden: int, n_genes: int, *, dispersion: str = "gene"
    ) -> None:
        super().__init__(n_hidden, n_genes)
        if dispersion != "gene":
            raise ValueError(
                f"Unsupported dispersion {dispersion!r}; only 'gene' is supported."
            )
        self.dispersion = dispersion
        self.scale_decoder = nn.Linear(n_hidden, n_genes)
        # Learned in log-space for positivity; theta = exp(log_theta).
        self.log_theta = nn.Parameter(torch.zeros(n_genes))

    def _px_scale(self, hidden: Tensor) -> Tensor:
        """Per-cell mean proportions over genes (softmax, sums to one)."""
        return torch.softmax(self.scale_decoder(hidden), dim=-1)

    def _px_rate(self, hidden: Tensor, size_factor: Tensor) -> Tensor:
        """NB mean: mean proportions scaled by the observed library size."""
        return self._px_scale(hidden) * size_factor.unsqueeze(-1)

    def _theta(self) -> Tensor:
        """Per-gene inverse dispersion (positive)."""
        return torch.exp(self.log_theta).clamp(min=_EPS)

    def forward(self, hidden: Tensor, size_factor: Tensor) -> Dict[str, Tensor]:
        px_scale = self._px_scale(hidden)
        px_rate = px_scale * size_factor.unsqueeze(-1)
        return {"px_scale": px_scale, "px_rate": px_rate, "theta": self._theta()}

    def reconstruction_loss(
        self,
        hidden: Tensor,
        target: Tensor,
        size_factor: Tensor,
        *,
        reduction: str = "mean",
    ) -> Tensor:
        px_rate = self._px_rate(hidden, size_factor)
        log_prob = log_nb_positive(target, px_rate, self._theta())
        return _reduce(-log_prob.sum(dim=-1), reduction)

    def expected_counts(self, hidden: Tensor, size_factor: Tensor) -> Tensor:
        return self._px_rate(hidden, size_factor)


class ZINBHead(NBHead):
    """
    Zero-Inflated Negative-Binomial reconstruction head.

    Extends :class:`NBHead` with a second linear layer predicting per-gene
    zero-inflation logits, capturing excess zeros (dropout) on top of the NB
    component. The mean is the NB rate scaled by the non-dropout probability.

    Parameters
    ----------
    n_hidden : int
        Dimensionality of the incoming hidden representation.
    n_genes : int
        Number of output genes.
    dispersion : {"gene"}, default "gene"
        Dispersion parameterization (see :class:`NBHead`).
    """

    likelihood = "zinb"

    def __init__(
        self, n_hidden: int, n_genes: int, *, dispersion: str = "gene"
    ) -> None:
        super().__init__(n_hidden, n_genes, dispersion=dispersion)
        self.zi_decoder = nn.Linear(n_hidden, n_genes)

    def forward(self, hidden: Tensor, size_factor: Tensor) -> Dict[str, Tensor]:
        params = super().forward(hidden, size_factor)
        params["zi_logits"] = self.zi_decoder(hidden)
        return params

    def reconstruction_loss(
        self,
        hidden: Tensor,
        target: Tensor,
        size_factor: Tensor,
        *,
        reduction: str = "mean",
    ) -> Tensor:
        px_rate = self._px_rate(hidden, size_factor)
        zi_logits = self.zi_decoder(hidden)
        log_prob = log_zinb_positive(target, px_rate, self._theta(), zi_logits)
        return _reduce(-log_prob.sum(dim=-1), reduction)

    def expected_counts(self, hidden: Tensor, size_factor: Tensor) -> Tensor:
        # Mixture mean: (1 - dropout_prob) * NB_mean.
        dropout = torch.sigmoid(self.zi_decoder(hidden))
        return (1.0 - dropout) * self._px_rate(hidden, size_factor)


class GaussianHead(ReconstructionHead):
    """
    Gaussian reconstruction head (log-normalized/Gaussian alternative).

    Decodes the hidden vector to a per-gene mean and uses a learned per-gene
    log-variance. The ``target`` is expected to be the internally normalized
    expression (e.g. ``log1p``), and ``size_factor`` is ignored — depth is
    assumed to be handled by the normalization rather than a count mean.

    Parameters
    ----------
    n_hidden : int
        Dimensionality of the incoming hidden representation.
    n_genes : int
        Number of output genes.
    """

    likelihood = "gaussian"

    def __init__(self, n_hidden: int, n_genes: int) -> None:
        super().__init__(n_hidden, n_genes)
        self.mean_decoder = nn.Linear(n_hidden, n_genes)
        self.log_var = nn.Parameter(torch.zeros(n_genes))

    def forward(self, hidden: Tensor, size_factor: Tensor) -> Dict[str, Tensor]:
        del size_factor  # Gaussian reconstruction is depth-agnostic.
        mean = self.mean_decoder(hidden)
        log_var = self.log_var.expand_as(mean)
        return {"mean": mean, "log_var": log_var}

    def reconstruction_loss(
        self,
        hidden: Tensor,
        target: Tensor,
        size_factor: Tensor,
        *,
        reduction: str = "mean",
    ) -> Tensor:
        del size_factor  # Unused; kept for a uniform head interface.
        mean = self.mean_decoder(hidden)
        log_prob = log_gaussian(target, mean, self.log_var)
        return _reduce(-log_prob.sum(dim=-1), reduction)

    def expected_counts(self, hidden: Tensor, size_factor: Tensor) -> Tensor:
        del size_factor
        mean: Tensor = self.mean_decoder(hidden)
        return mean


def build_reconstruction_head(
    likelihood: str, n_hidden: int, n_genes: int, **kwargs: object
) -> ReconstructionHead:
    """
    Construct a reconstruction head by likelihood name.

    Parameters
    ----------
    likelihood : {"nb", "zinb", "gaussian"}
        Which reconstruction head to build.
    n_hidden : int
        Dimensionality of the incoming hidden representation.
    n_genes : int
        Number of output genes.
    **kwargs
        Forwarded to the head constructor (e.g. ``dispersion`` for the count
        heads).

    Returns
    -------
    ReconstructionHead
        The constructed head.

    Raises
    ------
    ValueError
        If ``likelihood`` is not one of :data:`LIKELIHOODS`.
    """
    name = likelihood.lower()
    if name == "nb":
        return NBHead(n_hidden, n_genes, **kwargs)  # type: ignore[arg-type]
    if name == "zinb":
        return ZINBHead(n_hidden, n_genes, **kwargs)  # type: ignore[arg-type]
    if name == "gaussian":
        return GaussianHead(n_hidden, n_genes, **kwargs)
    raise ValueError(
        f"Unknown likelihood {likelihood!r}; expected one of {LIKELIHOODS}."
    )
