"""
Residual vector quantization for OQAE.

This module implements the discrete bottleneck of the OQAE VQ-VAE: a stack of
vector-quantization codebooks applied to *residuals*, so each encoder output is
represented as a small set of codebook indices — the discrete, universal code
for a cell.

The building blocks are:

- :class:`VectorQuantizer` — a single codebook with a straight-through
  estimator, commitment/codebook losses, optional EMA codebook updates, and
  dead-code reset. It also reports per-forward monitoring metrics (codebook
  **perplexity** and **utilization**).
- :class:`ResidualVQ` — stacks ``n_codebooks`` quantizers; codebook ``i`` + 1
  quantizes the residual left by the previous codebooks. The sum of the
  per-level quantized vectors approximates the input; the per-level indices are
  the cell's discrete code.

Both return small dataclasses (:class:`QuantizerOutput` / :class:`ResidualVQOutput`)
bundling the quantized vectors, indices, losses, and metrics so the model and
W&B-logging PRs can consume them uniformly.

Design notes
------------
- Gradients flow through the discrete argmin via the straight-through estimator
  (``quantized = x + (quantized - x).detach()``), so the encoder trains end to
  end.
- With ``ema=True`` the codebook is updated by an exponential moving average of
  assigned encoder outputs (scVI/VQ-VAE-2 style) instead of by gradient descent,
  which is typically more stable; the codebook-pull loss term is then dropped
  and only the commitment loss is kept.
- Dead codes (codebook entries that go unused) are periodically reset to random
  encoder outputs from the current batch to guard against codebook collapse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, cast

import torch
from torch import Tensor, nn

from omvqvae.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "QuantizerOutput",
    "ResidualVQOutput",
    "VectorQuantizer",
    "ResidualVQ",
]


@dataclass
class QuantizerOutput:
    """
    Output of a single :class:`VectorQuantizer` forward pass.

    Attributes
    ----------
    quantized : torch.Tensor
        Quantized vectors of the same shape as the input, carrying the
        straight-through gradient back to the encoder.
    indices : torch.Tensor
        Selected codebook indices of shape ``input.shape[:-1]`` (``int64``).
    loss : torch.Tensor
        Scalar VQ loss for this codebook (codebook + weighted commitment term;
        codebook term is zero under EMA).
    commitment_loss : torch.Tensor
        Scalar commitment loss pulling encoder outputs toward the codebook.
    codebook_loss : torch.Tensor
        Scalar codebook loss pulling the codebook toward encoder outputs
        (always zero under EMA updates).
    perplexity : torch.Tensor
        Scalar codebook perplexity for this batch (``exp`` of the assignment
        entropy); higher means more codes are used evenly.
    usage : torch.Tensor
        Scalar fraction of codebook entries assigned at least one input in this
        batch (in ``[0, 1]``).
    """

    quantized: Tensor
    indices: Tensor
    loss: Tensor
    commitment_loss: Tensor
    codebook_loss: Tensor
    perplexity: Tensor
    usage: Tensor


@dataclass
class ResidualVQOutput:
    """
    Output of a :class:`ResidualVQ` forward pass.

    Attributes
    ----------
    quantized : torch.Tensor
        Sum of the per-level quantized vectors (same shape as the input),
        carrying the straight-through gradient back to the encoder.
    indices : torch.Tensor
        Per-level codebook indices of shape ``input.shape[:-1] + (n_codebooks,)``
        (``int64``); this is the cell's discrete code.
    loss : torch.Tensor
        Scalar total VQ loss summed over codebooks.
    commitment_loss : torch.Tensor
        Scalar commitment loss summed over codebooks.
    codebook_loss : torch.Tensor
        Scalar codebook loss summed over codebooks (zero under EMA updates).
    perplexity : torch.Tensor
        Scalar mean codebook perplexity across levels.
    perplexities : torch.Tensor
        Per-level perplexities of shape ``(n_codebooks,)``.
    usages : torch.Tensor
        Per-level codebook utilization of shape ``(n_codebooks,)``.
    """

    quantized: Tensor
    indices: Tensor
    loss: Tensor
    commitment_loss: Tensor
    codebook_loss: Tensor
    perplexity: Tensor
    perplexities: Tensor
    usages: Tensor


class VectorQuantizer(nn.Module):
    """
    A single vector-quantization codebook with a straight-through estimator.

    Parameters
    ----------
    codebook_size : int
        Number of codebook entries (vocabulary size for this codebook).
    embedding_dim : int
        Dimensionality of each codebook vector (must match the encoder output).
    commitment_cost : float, default 0.25
        Weight ``beta`` on the commitment loss in the total VQ loss.
    ema : bool, default True
        If ``True``, update the codebook with an exponential moving average of
        assigned encoder outputs instead of by gradient descent; the codebook
        loss term is then dropped.
    ema_decay : float, default 0.99
        EMA decay used when ``ema=True``.
    ema_epsilon : float, default 1e-5
        Laplace-smoothing constant for EMA cluster sizes.
    reset_dead_codes : bool, default True
        If ``True``, reset codes whose usage EMA falls below
        ``dead_code_threshold`` to random encoder outputs from the batch
        (collapse guard); only active in training mode.
    dead_code_threshold : float, default 1.0
        Usage-EMA threshold below which a code is considered dead.

    Raises
    ------
    ValueError
        If ``codebook_size`` or ``embedding_dim`` is not positive, or if
        ``ema_decay`` is not in ``[0, 1)``.
    """

    # Registered as buffers/parameter in ``__init__``; declared here so the
    # static type checker treats them as tensors rather than ``Tensor | Module``
    # (the return type of ``nn.Module.__getattr__``).
    embedding: Tensor
    cluster_size: Tensor
    embed_avg: Tensor

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        *,
        commitment_cost: float = 0.25,
        ema: bool = True,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
        reset_dead_codes: bool = True,
        dead_code_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        if codebook_size <= 0:
            raise ValueError("codebook_size must be a positive integer.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer.")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1).")

        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema = ema
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        self.reset_dead_codes = reset_dead_codes
        self.dead_code_threshold = dead_code_threshold

        embedding = torch.randn(codebook_size, embedding_dim)
        # Under EMA the codebook is a buffer updated in-place (no autograd);
        # otherwise it is a learnable parameter optimized by gradient descent.
        if ema:
            self.register_buffer("embedding", embedding)
        else:
            self.embedding = nn.Parameter(embedding)

        # EMA statistics / usage tracker (kept regardless of update rule so the
        # dead-code reset has a usage signal to act on).
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embedding.clone())

    def _codebook(self) -> Tensor:
        """Return the codebook tensor (buffer under EMA, parameter otherwise)."""
        embedding: Tensor = self.embedding
        return embedding

    def lookup(self, indices: Tensor) -> Tensor:
        """
        Map codebook indices back to their codebook vectors.

        This is the inverse of the assignment step in :meth:`forward`: it embeds
        discrete indices without computing distances or losses, so a stored code
        can be turned back into its quantized vector for decoding/generation.

        Parameters
        ----------
        indices : torch.Tensor
            Codebook indices of arbitrary shape (``int64``); each value must lie
            in ``[0, codebook_size)``.

        Returns
        -------
        torch.Tensor
            Codebook vectors of shape ``indices.shape + (embedding_dim,)``.
        """
        return self._codebook()[indices]

    def forward(self, inputs: Tensor) -> QuantizerOutput:
        """
        Quantize ``inputs`` to the nearest codebook entries.

        Parameters
        ----------
        inputs : torch.Tensor
            Encoder outputs of shape ``(..., embedding_dim)``.

        Returns
        -------
        QuantizerOutput
            Quantized vectors, indices, losses, and monitoring metrics.

        Raises
        ------
        ValueError
            If the last dimension of ``inputs`` is not ``embedding_dim``.
        """
        if inputs.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"inputs last dim {inputs.shape[-1]} != embedding_dim "
                f"{self.embedding_dim}."
            )

        leading_shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.embedding_dim)
        codebook = self._codebook()

        # Squared Euclidean distances to every codebook entry: (N, codebook_size).
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        encodings = torch.zeros(
            flat.shape[0], self.codebook_size, device=flat.device, dtype=flat.dtype
        )
        encodings.scatter_(1, indices.unsqueeze(1), 1.0)

        quantized_flat = codebook[indices]

        # VQ losses. The commitment term pulls the encoder toward the codebook;
        # the codebook term pulls the codebook toward the encoder (skipped under
        # EMA, where the codebook is updated by moving average instead).
        commitment_loss = torch.mean((flat - quantized_flat.detach()) ** 2)
        if self.ema:
            codebook_loss = torch.zeros((), device=flat.device, dtype=flat.dtype)
        else:
            codebook_loss = torch.mean((flat.detach() - quantized_flat) ** 2)
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: copy gradients from quantized to inputs.
        quantized_flat = flat + (quantized_flat - flat).detach()

        if self.training:
            self._update_usage(encodings)
            if self.ema:
                self._ema_update(flat, encodings)
            if self.reset_dead_codes:
                self._reset_dead_codes(flat)

        perplexity, usage = self._metrics(encodings)

        quantized = quantized_flat.reshape(*leading_shape, self.embedding_dim)
        indices_out = indices.reshape(leading_shape)
        return QuantizerOutput(
            quantized=quantized,
            indices=indices_out,
            loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            perplexity=perplexity,
            usage=usage,
        )

    @torch.no_grad()
    def _update_usage(self, encodings: Tensor) -> None:
        """Update the EMA usage tracker from this batch's assignments."""
        counts = encodings.sum(dim=0)
        self.cluster_size.mul_(self.ema_decay).add_(counts, alpha=1.0 - self.ema_decay)

    @torch.no_grad()
    def _ema_update(self, flat: Tensor, encodings: Tensor) -> None:
        """Update the codebook by an EMA of assigned encoder outputs."""
        embed_sum = encodings.t() @ flat
        self.embed_avg.mul_(self.ema_decay).add_(embed_sum, alpha=1.0 - self.ema_decay)
        # Laplace smoothing keeps unused codes from collapsing to zero.
        n = self.cluster_size.sum()
        smoothed = (
            (self.cluster_size + self.ema_epsilon)
            / (n + self.codebook_size * self.ema_epsilon)
            * n
        )
        self._codebook().copy_(self.embed_avg / smoothed.unsqueeze(1))

    @torch.no_grad()
    def _reset_dead_codes(self, flat: Tensor) -> None:
        """Reset under-used codebook entries to random encoder outputs."""
        dead = self.cluster_size < self.dead_code_threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return
        # Sample replacement vectors (with replacement) from the current batch.
        sample_idx = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
        replacements = flat[sample_idx]
        codebook = self._codebook()
        codebook[dead] = replacements
        self.embed_avg[dead] = replacements
        self.cluster_size[dead] = 1.0
        logger.debug("Reset %d dead codebook entries.", n_dead)

    def _metrics(self, encodings: Tensor) -> tuple[Tensor, Tensor]:
        """Compute batch perplexity and codebook utilization."""
        avg_probs = encodings.mean(dim=0)
        # exp(entropy); add eps inside log for numerical stability.
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        usage = (avg_probs > 0).to(avg_probs.dtype).mean()
        return perplexity, usage


class ResidualVQ(nn.Module):
    """
    A stack of vector quantizers applied to successive residuals.

    Each cell's encoder output is quantized by the first codebook; the residual
    (input minus its quantization) is quantized by the second codebook, and so
    on. The per-level indices form the cell's discrete code and the per-level
    quantized vectors sum to an approximation of the input.

    Parameters
    ----------
    n_codebooks : int, default 2
        Number of residual quantization levels.
    codebook_size : int
        Number of entries in each codebook.
    embedding_dim : int
        Dimensionality of the encoder output / codebook vectors.
    commitment_cost : float, default 0.25
        Commitment-loss weight passed to every :class:`VectorQuantizer`.
    ema : bool, default True
        Whether each codebook is updated by EMA (see :class:`VectorQuantizer`).
    ema_decay : float, default 0.99
        EMA decay for the codebooks.
    reset_dead_codes : bool, default True
        Whether each codebook resets dead entries (collapse guard).

    Raises
    ------
    ValueError
        If ``n_codebooks`` is not a positive integer.
    """

    def __init__(
        self,
        *,
        codebook_size: int,
        embedding_dim: int,
        n_codebooks: int = 2,
        commitment_cost: float = 0.25,
        ema: bool = True,
        ema_decay: float = 0.99,
        reset_dead_codes: bool = True,
    ) -> None:
        super().__init__()
        if n_codebooks <= 0:
            raise ValueError("n_codebooks must be a positive integer.")

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.quantizers = nn.ModuleList(
            VectorQuantizer(
                codebook_size,
                embedding_dim,
                commitment_cost=commitment_cost,
                ema=ema,
                ema_decay=ema_decay,
                reset_dead_codes=reset_dead_codes,
            )
            for _ in range(n_codebooks)
        )

    def lookup(self, indices: Tensor) -> Tensor:
        """
        Reconstruct the summed quantized vector from per-level codebook indices.

        This inverts the index half of :meth:`forward`: it maps a cell's discrete
        code (one index per residual level) back to the summed quantized latent
        that the decoder consumes, without recomputing the encoder/residuals.

        Parameters
        ----------
        indices : torch.Tensor
            Per-level codebook indices of shape ``(..., n_codebooks)``
            (``int64``), as returned in :attr:`ResidualVQOutput.indices`.

        Returns
        -------
        torch.Tensor
            Summed quantized vectors of shape
            ``indices.shape[:-1] + (embedding_dim,)``.

        Raises
        ------
        ValueError
            If the last dimension of ``indices`` is not ``n_codebooks``.
        """
        if indices.shape[-1] != self.n_codebooks:
            raise ValueError(
                f"indices last dim {indices.shape[-1]} != n_codebooks "
                f"{self.n_codebooks}."
            )
        quantized: Optional[Tensor] = None
        for level in range(self.n_codebooks):
            quantizer = cast(VectorQuantizer, self.quantizers[level])
            vectors = quantizer.lookup(indices[..., level])
            quantized = vectors if quantized is None else quantized + vectors
        # ``n_codebooks >= 1`` is enforced in ``__init__`` so the loop always runs.
        assert quantized is not None
        return quantized

    def forward(self, inputs: Tensor) -> ResidualVQOutput:
        """
        Quantize ``inputs`` through the residual codebook stack.

        Parameters
        ----------
        inputs : torch.Tensor
            Encoder outputs of shape ``(..., embedding_dim)``.

        Returns
        -------
        ResidualVQOutput
            Summed quantized vectors, per-level indices, summed losses, and
            per-level / mean monitoring metrics.
        """
        residual = inputs
        quantized_sum = torch.zeros_like(inputs)
        total_loss = inputs.new_zeros(())
        total_commitment = inputs.new_zeros(())
        total_codebook = inputs.new_zeros(())

        indices_list: List[Tensor] = []
        perplexities: List[Tensor] = []
        usages: List[Tensor] = []

        for quantizer in self.quantizers:
            out: QuantizerOutput = quantizer(residual)
            # Subtract the (straight-through) quantization to form the next
            # residual; the running sum reconstructs the input.
            residual = residual - out.quantized
            quantized_sum = quantized_sum + out.quantized

            total_loss = total_loss + out.loss
            total_commitment = total_commitment + out.commitment_loss
            total_codebook = total_codebook + out.codebook_loss
            indices_list.append(out.indices)
            perplexities.append(out.perplexity)
            usages.append(out.usage)

        indices = torch.stack(indices_list, dim=-1)
        perplexities_t = torch.stack(perplexities)
        usages_t = torch.stack(usages)
        return ResidualVQOutput(
            quantized=quantized_sum,
            indices=indices,
            loss=total_loss,
            commitment_loss=total_commitment,
            codebook_loss=total_codebook,
            perplexity=perplexities_t.mean(),
            perplexities=perplexities_t,
            usages=usages_t,
        )
