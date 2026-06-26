"""
Inference API for OQAE.

The user-facing latent interface on top of a trained
:class:`~omvqvae.models.vqvae.OmicsVQVAE`: encode expression into the model's
discrete universal codes and decode codes back into expression (see
:mod:`omvqvae.inference.codes`).
"""

from omvqvae.inference.codes import (
    EncodedCells,
    decode,
    decode_to_params,
    encode,
    encode_anndata,
)

__all__ = [
    "EncodedCells",
    "encode",
    "encode_anndata",
    "decode",
    "decode_to_params",
]
