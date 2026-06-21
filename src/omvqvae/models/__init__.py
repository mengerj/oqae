"""VQ-VAE model implementations.

Currently exposes the raw-count reconstruction likelihoods and decoder heads
(NB / ZINB / Gaussian); the encoder/decoder VQ-VAE itself lands with PR #4.
"""

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
