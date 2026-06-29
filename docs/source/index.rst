OQAE — Omics Quantized Auto Encoder
===================================

**OQAE** (Omics Quantized Auto Encoder) learns a **discrete, universal latent
space for single-cell omics** with a residual VQ-VAE. Each scRNA-seq cell is
encoded as a small set of discrete codes (indices into learned codebooks); a
generative decoder turns those codes back into expression. The codes are a
compact, composable "vocabulary" of expression patterns, enabling
representation learning, compression, integration, and in-silico generation
from a shared latent space.

The Python package is :mod:`omvqvae` (distribution name ``oqae``).

Highlights
----------

- **Discrete universal latent space** — every cell becomes a set of discrete
  codes drawn from shared codebooks.
- **Train at scale by streaming** — stream millions of cells directly from the
  CZ CELLxGENE Census via TileDB-SOMA, or train on a local ``.h5ad`` / ``.zarr``
  AnnData with the same interface.
- **Raw counts in, counts out** — model raw counts directly with a Negative
  Binomial / Zero-Inflated NB likelihood; library size is handled internally.
- **Organism-aware** — human and mouse, each with its own gene space (one model
  per organism in v1).
- **HuggingFace Hub + W&B** — share models and codebooks; track runs offline or
  with Weights & Biases.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
