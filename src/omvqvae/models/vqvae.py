"""
End-to-end VQ-VAE model for OQAE.

This module wires the two halves of the discrete bottleneck — the residual
vector quantizer (:mod:`omvqvae.layers.residual_vq`) and the reconstruction
heads (:mod:`omvqvae.models.likelihoods`) — into a single
:class:`torch.nn.Module`:

.. code-block:: text

    raw counts ─► (internal log1p) ─► Encoder ─► ResidualVQ ─► Decoder ─► NB/ZINB
                                                  (discrete codes)         params

The model ingests **raw counts** and an observed per-cell **size factor**. An
internal ``log1p`` transform is applied to the encoder input for numerical
stability (the raw counts themselves are reconstructed by a count likelihood).
The encoder maps each cell to a continuous latent vector; the
:class:`~omvqvae.layers.residual_vq.ResidualVQ` quantizes it to a small set of
codebook indices — the cell's discrete, universal code — and the decoder maps
the summed quantized vectors (plus the size factor) to the parameters of the
reconstruction likelihood.

The forward pass returns a :class:`VQVAEOutput` bundling the composed loss
(reconstruction + VQ), the per-level codes, and the codebook monitoring metrics
(perplexity / utilization) so the training and W&B-logging PRs can consume them
uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn

from omvqvae.layers.residual_vq import ResidualVQ, ResidualVQOutput
from omvqvae.models.likelihoods import ReconstructionHead, build_reconstruction_head
from omvqvae.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "VQVAEOutput",
    "OmicsVQVAE",
]


@dataclass
class VQVAEOutput:
    """
    Output of an :class:`OmicsVQVAE` forward pass.

    Attributes
    ----------
    loss : torch.Tensor
        Scalar total loss: reconstruction loss + VQ loss.
    reconstruction_loss : torch.Tensor
        Scalar reconstruction loss (negative log-likelihood, summed over genes
        and reduced over cells).
    vq_loss : torch.Tensor
        Scalar VQ loss summed over codebooks (commitment + codebook terms).
    commitment_loss : torch.Tensor
        Scalar commitment loss summed over codebooks.
    codebook_loss : torch.Tensor
        Scalar codebook loss summed over codebooks (zero under EMA updates).
    indices : torch.Tensor
        Per-level codebook indices of shape ``(batch, n_codebooks)``
        (``int64``); this is each cell's discrete code.
    latent : torch.Tensor
        Continuous encoder output of shape ``(batch, n_latent)`` before
        quantization.
    quantized : torch.Tensor
        Summed quantized latent of shape ``(batch, n_latent)``.
    perplexity : torch.Tensor
        Scalar mean codebook perplexity across levels.
    perplexities : torch.Tensor
        Per-level perplexities of shape ``(n_codebooks,)``.
    usages : torch.Tensor
        Per-level codebook utilization of shape ``(n_codebooks,)``.
    """

    loss: Tensor
    reconstruction_loss: Tensor
    vq_loss: Tensor
    commitment_loss: Tensor
    codebook_loss: Tensor
    indices: Tensor
    latent: Tensor
    quantized: Tensor
    perplexity: Tensor
    perplexities: Tensor
    usages: Tensor


def _build_mlp(
    in_dim: int, dims: Sequence[int], dropout: float
) -> Tuple[nn.Sequential, int]:
    """
    Build a ``Linear -> ReLU (-> Dropout)`` MLP body.

    Parameters
    ----------
    in_dim : int
        Input dimensionality.
    dims : Sequence[int]
        Hidden-layer widths, in order. May be empty (yields an identity body).
    dropout : float
        Dropout probability applied after each ReLU; skipped when ``<= 0``.

    Returns
    -------
    tuple of (torch.nn.Sequential, int)
        The MLP body and its output dimensionality (``in_dim`` when ``dims`` is
        empty).
    """
    layers: List[nn.Module] = []
    prev = in_dim
    for width in dims:
        layers.append(nn.Linear(prev, width))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev = width
    return nn.Sequential(*layers), prev


class OmicsVQVAE(nn.Module):
    """
    Encoder / residual-VQ / decoder VQ-VAE over raw single-cell counts.

    The model encodes raw counts to a continuous latent, quantizes it to a set
    of discrete codes via a :class:`~omvqvae.layers.residual_vq.ResidualVQ`, and
    decodes the codes back to the parameters of a count (or Gaussian)
    reconstruction likelihood. The encoder input is internally ``log1p``-
    transformed for numerical stability; the reconstruction *target* is the raw
    counts for the NB/ZINB heads and the ``log1p`` expression for the Gaussian
    head.

    Parameters
    ----------
    n_genes : int
        Number of input/output genes (the organism's feature-space size).
    n_latent : int, default 16
        Dimensionality of the latent / codebook vectors.
    hidden_dims : Sequence[int], default (128,)
        Encoder hidden-layer widths (``n_genes -> ... -> n_latent``). The decoder
        mirrors these in reverse. May be empty for a linear encoder/decoder.
    likelihood : {"nb", "zinb", "gaussian"}, default "nb"
        Reconstruction likelihood (see
        :func:`omvqvae.models.likelihoods.build_reconstruction_head`).
    codebook_size : int, default 256
        Number of entries in each codebook.
    n_codebooks : int, default 2
        Number of residual quantization levels.
    commitment_cost : float, default 0.25
        Commitment-loss weight passed to the quantizer.
    ema : bool, default True
        Whether codebooks update by EMA (see
        :class:`~omvqvae.layers.residual_vq.VectorQuantizer`).
    ema_decay : float, default 0.99
        EMA decay for the codebooks.
    reset_dead_codes : bool, default True
        Whether each codebook resets dead entries (collapse guard).
    dropout : float, default 0.0
        Dropout probability in the encoder/decoder MLPs.

    Raises
    ------
    ValueError
        If ``n_genes`` or ``n_latent`` is not positive.
    """

    def __init__(
        self,
        n_genes: int,
        *,
        n_latent: int = 16,
        hidden_dims: Sequence[int] = (128,),
        likelihood: str = "nb",
        codebook_size: int = 256,
        n_codebooks: int = 2,
        commitment_cost: float = 0.25,
        ema: bool = True,
        ema_decay: float = 0.99,
        reset_dead_codes: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_genes < 1:
            raise ValueError("n_genes must be a positive integer.")
        if n_latent < 1:
            raise ValueError("n_latent must be a positive integer.")

        self.n_genes = n_genes
        self.n_latent = n_latent
        self.hidden_dims = tuple(hidden_dims)
        self.likelihood = likelihood.lower()
        # Remaining constructor hyper-parameters are stored verbatim so the model
        # is self-describing (see ``get_config`` / ``from_config``), which is what
        # the HuggingFace-Hub serialization round-trips.
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.ema = ema
        self.ema_decay = ema_decay
        self.reset_dead_codes = reset_dead_codes
        self.dropout = dropout

        # Encoder: counts (log1p) -> hidden MLP -> latent.
        enc_body, enc_out = _build_mlp(n_genes, self.hidden_dims, dropout)
        self.encoder_body = enc_body
        self.to_latent = nn.Linear(enc_out, n_latent)

        # Discrete bottleneck.
        self.rvq = ResidualVQ(
            codebook_size=codebook_size,
            embedding_dim=n_latent,
            n_codebooks=n_codebooks,
            commitment_cost=commitment_cost,
            ema=ema,
            ema_decay=ema_decay,
            reset_dead_codes=reset_dead_codes,
        )

        # Decoder: latent -> hidden MLP (mirror of the encoder) -> head.
        dec_body, dec_out = _build_mlp(
            n_latent, tuple(reversed(self.hidden_dims)), dropout
        )
        self.decoder_body = dec_body
        self.head: ReconstructionHead = build_reconstruction_head(
            self.likelihood, dec_out, n_genes
        )

    #: Constructor keyword arguments restored by :meth:`from_config`.
    _CONFIG_KEYS: Tuple[str, ...] = (
        "n_latent",
        "hidden_dims",
        "likelihood",
        "codebook_size",
        "n_codebooks",
        "commitment_cost",
        "ema",
        "ema_decay",
        "reset_dead_codes",
        "dropout",
    )

    @property
    def n_codebooks(self) -> int:
        """Number of residual quantization levels."""
        return self.rvq.n_codebooks

    def get_config(self) -> Dict[str, Any]:
        """
        Return the hyper-parameters needed to rebuild this model.

        The mapping is JSON-serializable and is the inverse of
        :meth:`from_config`: ``OmicsVQVAE.from_config(model.get_config())``
        reconstructs an architecturally-identical (untrained) model.

        Returns
        -------
        Dict[str, Any]
            ``n_genes`` plus every constructor keyword argument.
        """
        return {
            "n_genes": self.n_genes,
            "n_latent": self.n_latent,
            "hidden_dims": list(self.hidden_dims),
            "likelihood": self.likelihood,
            "codebook_size": self.codebook_size,
            "n_codebooks": self.n_codebooks,
            "commitment_cost": self.commitment_cost,
            "ema": self.ema,
            "ema_decay": self.ema_decay,
            "reset_dead_codes": self.reset_dead_codes,
            "dropout": self.dropout,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "OmicsVQVAE":
        """
        Construct an :class:`OmicsVQVAE` from a :meth:`get_config` mapping.

        Unknown keys are ignored so the format can grow without breaking older
        checkpoints; ``n_genes`` is required.

        Parameters
        ----------
        config : Mapping[str, Any]
            A hyper-parameter mapping as produced by :meth:`get_config`.

        Returns
        -------
        OmicsVQVAE
            A freshly-constructed (untrained) model.

        Raises
        ------
        KeyError
            If ``n_genes`` is absent from ``config``.
        """
        if "n_genes" not in config:
            raise KeyError("Model config is missing the required 'n_genes' key.")
        kwargs = {key: config[key] for key in cls._CONFIG_KEYS if key in config}
        return cls(int(config["n_genes"]), **kwargs)

    def _encoder_input(self, counts: Tensor) -> Tensor:
        """Internal ``log1p`` transform applied to the encoder input."""
        return torch.log1p(counts)

    def _recon_target(self, counts: Tensor) -> Tensor:
        """
        Reconstruction target for the configured likelihood.

        Raw counts for the count heads (NB/ZINB); ``log1p`` expression for the
        Gaussian head, which models normalized values rather than counts.
        """
        if self.likelihood == "gaussian":
            return torch.log1p(counts)
        return counts

    def encode(self, counts: Tensor) -> Tensor:
        """
        Encode raw counts to the continuous (pre-quantization) latent.

        Parameters
        ----------
        counts : torch.Tensor
            Raw counts of shape ``(batch, n_genes)``.

        Returns
        -------
        torch.Tensor
            Latent vectors of shape ``(batch, n_latent)``.
        """
        hidden = self.encoder_body(self._encoder_input(counts))
        latent: Tensor = self.to_latent(hidden)
        return latent

    def quantize(self, latent: Tensor) -> ResidualVQOutput:
        """Quantize a continuous latent through the residual codebook stack."""
        out: ResidualVQOutput = self.rvq(latent)
        return out

    def encode_codes(self, counts: Tensor) -> Tensor:
        """
        Encode raw counts to their discrete codes (per-level codebook indices).

        Parameters
        ----------
        counts : torch.Tensor
            Raw counts of shape ``(batch, n_genes)``.

        Returns
        -------
        torch.Tensor
            Codebook indices of shape ``(batch, n_codebooks)`` (``int64``).

        Notes
        -----
        In training mode this still updates the EMA codebooks / dead-code
        statistics; call on a model in ``eval()`` mode for pure inference.
        """
        out: ResidualVQOutput = self.rvq(self.encode(counts))
        return out.indices

    def decode(self, quantized: Tensor, size_factors: Tensor) -> Dict[str, Tensor]:
        """
        Decode a quantized latent to the reconstruction-likelihood parameters.

        Parameters
        ----------
        quantized : torch.Tensor
            Quantized latent of shape ``(batch, n_latent)``.
        size_factors : torch.Tensor
            Per-cell size factors of shape ``(batch,)``.

        Returns
        -------
        Dict[str, torch.Tensor]
            The per-gene distribution parameters (see the configured head).
        """
        params: Dict[str, Tensor] = self.head(
            self.decoder_body(quantized), size_factors
        )
        return params

    def expected_counts(self, quantized: Tensor, size_factors: Tensor) -> Tensor:
        """Return the reconstruction mean for a quantized latent."""
        return self.head.expected_counts(self.decoder_body(quantized), size_factors)

    def codes_to_params(self, codes: Tensor, size_factors: Tensor) -> Dict[str, Tensor]:
        """
        Decode discrete codes to the reconstruction-likelihood parameters.

        Inverts the index half of :meth:`encode_codes`: the per-level codebook
        indices are mapped back to the summed quantized latent
        (:meth:`~omvqvae.layers.residual_vq.ResidualVQ.lookup`) and decoded to the
        per-gene distribution parameters.

        Parameters
        ----------
        codes : torch.Tensor
            Per-level codebook indices of shape ``(batch, n_codebooks)``
            (``int64``).
        size_factors : torch.Tensor
            Per-cell size factors of shape ``(batch,)``.

        Returns
        -------
        Dict[str, torch.Tensor]
            The per-gene distribution parameters (see the configured head).
        """
        return self.decode(self.rvq.lookup(codes), size_factors)

    def decode_codes(self, codes: Tensor, size_factors: Tensor) -> Tensor:
        """
        Reconstruct expected counts directly from discrete codes.

        Inverse of :meth:`encode_codes`: maps a cell's per-level codebook indices
        back to the summed quantized latent and decodes it to the reconstruction
        mean (depth-appropriate expected counts for NB/ZINB; ``log1p`` expression
        for the Gaussian head).

        Parameters
        ----------
        codes : torch.Tensor
            Per-level codebook indices of shape ``(batch, n_codebooks)``
            (``int64``).
        size_factors : torch.Tensor
            Per-cell size factors of shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Reconstruction mean of shape ``(batch, n_genes)``.
        """
        return self.expected_counts(self.rvq.lookup(codes), size_factors)

    def forward(self, counts: Tensor, size_factors: Tensor) -> VQVAEOutput:
        """
        Encode, quantize, decode, and compose the loss for a batch.

        Parameters
        ----------
        counts : torch.Tensor
            Raw counts of shape ``(batch, n_genes)``.
        size_factors : torch.Tensor
            Per-cell size factors of shape ``(batch,)``.

        Returns
        -------
        VQVAEOutput
            Composed loss, discrete codes, and codebook monitoring metrics.
        """
        latent = self.encode(counts)
        vq: ResidualVQOutput = self.rvq(latent)
        hidden = self.decoder_body(vq.quantized)
        recon_loss = self.head.reconstruction_loss(
            hidden, self._recon_target(counts), size_factors, reduction="mean"
        )
        total_loss = recon_loss + vq.loss
        return VQVAEOutput(
            loss=total_loss,
            reconstruction_loss=recon_loss,
            vq_loss=vq.loss,
            commitment_loss=vq.commitment_loss,
            codebook_loss=vq.codebook_loss,
            indices=vq.indices,
            latent=latent,
            quantized=vq.quantized,
            perplexity=vq.perplexity,
            perplexities=vq.perplexities,
            usages=vq.usages,
        )
