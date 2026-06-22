"""VQ-VAE model implementations.

Exposes the raw-count reconstruction likelihoods and decoder heads
(NB / ZINB / Gaussian) and the end-to-end encoder/residual-VQ/decoder
:class:`OmicsVQVAE`.
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
from omvqvae.models.vqvae import OmicsVQVAE, VQVAEOutput

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
    "OmicsVQVAE",
    "VQVAEOutput",
]
