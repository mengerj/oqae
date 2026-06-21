"""Neural network layers for VQ-VAE models."""

from omvqvae.layers.residual_vq import (
    QuantizerOutput,
    ResidualVQ,
    ResidualVQOutput,
    VectorQuantizer,
)

__all__ = [
    "QuantizerOutput",
    "ResidualVQOutput",
    "VectorQuantizer",
    "ResidualVQ",
]
