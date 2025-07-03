"""
OQAE: Omics Quantized Auto Encoder.

A lightweight VQ-VAE library for large-scale omics data analysis with
memory-efficient processing and HuggingFace Hub integration.
"""

from omvqvae.utils.logging import get_logger

__version__ = "0.1.0"
__author__ = "mengerj"
__email__ = "mengerj@example.com"

# Initialize logging for the package
logger = get_logger(__name__)
logger.info(f"OQAE v{__version__} initialized")

# Public API will be imported here as modules are implemented
__all__ = [
    "__version__",
    "__author__",
    "__email__",
]
