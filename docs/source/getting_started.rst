Getting started
===============

This page walks through installing OQAE and running the end-to-end workflow:
train a model on raw counts, encode cells into discrete codes, and decode codes
back into expression. Every step is exercised by the runnable scripts under
`examples/ <https://github.com/mengerj/oqae/tree/main/examples>`_.

Installation
------------

OQAE uses `uv <https://docs.astral.sh/uv/>`_ for environment management:

.. code-block:: bash

   make setup-env          # uv sync --all-extras  (creates .venv)
   source .venv/bin/activate

To build these docs locally, install the ``docs`` extra and run the
``docs`` target:

.. code-block:: bash

   uv sync --extra docs
   make docs               # sphinx-build -> docs/_build/html

Core concepts
-------------

- **Raw counts in.** OQAE ingests **unnormalized counts**; ``log1p`` /
  depth normalization happen *inside* the model. Do not pre-normalize.
- **Size factors.** Decoding needs a per-cell *size factor* (observed depth) so
  the decoder reconstructs depth-appropriate counts. ``encode`` returns it
  alongside the codes.
- **Discrete codes.** A cell's representation is an ``int64``
  ``(n_cells, n_codebooks)`` array — column ``j`` is the index chosen from
  residual codebook ``j``. This *is* the universal latent code.
- **Organism-aware.** A model carries a fixed :class:`~omvqvae.data.GeneVocabulary`
  (organism + ordered gene ids); v1 trains one model per organism.

Train on a local AnnData
------------------------

Derive a :class:`~omvqvae.data.GeneVocabulary` from your file's genes, build a
:class:`~omvqvae.models.OmicsVQVAE`, and run the source-agnostic
:func:`~omvqvae.train.train` loop. See
`examples/01_train_local_anndata.py
<https://github.com/mengerj/oqae/blob/main/examples/01_train_local_anndata.py>`_.

The same run is a one-liner via the CLI (installed as the ``oqae-train``
console script):

.. code-block:: bash

   oqae-train configs/train_toy.yaml -s data.path=your_cells.h5ad

Train at scale by streaming the Census
--------------------------------------

:func:`~omvqvae.data.build_census_dataloader` streams raw counts from the CZ
CELLxGENE Census via TileDB-SOMA behind the same ``Minibatch`` contract as the
local loader. See `examples/03_census_streaming.py
<https://github.com/mengerj/oqae/blob/main/examples/03_census_streaming.py>`_
(network access required).

Encode and decode discrete codes
--------------------------------

Once a model is trained (or loaded with
:func:`~omvqvae.hf_utils.load_pretrained`), the :mod:`omvqvae.inference` API
turns expression into codes and back:

.. code-block:: python

   from omvqvae.inference import encode, decode

   encoded = encode(model, counts)          # -> EncodedCells (codes + size factors)
   codes = encoded.codes                    # (n_cells, n_codebooks) int64
   counts_hat = decode(model, encoded)      # codes -> expected counts

:func:`~omvqvae.inference.encode_anndata` aligns a local AnnData to the model's
gene vocabulary first; :func:`~omvqvae.inference.decode_to_params` exposes the
full decoder distribution for sampling/generation. The full walk-through —
inspecting codebook usage and generating a novel profile from edited codes — is
in `examples/02_inspect_and_generate_codes.py
<https://github.com/mengerj/oqae/blob/main/examples/02_inspect_and_generate_codes.py>`_.

Save and share a model
----------------------

:func:`~omvqvae.hf_utils.save_pretrained` writes a HuggingFace-style directory
(``config.json`` + ``pytorch_model.bin``) capturing the model, its codebooks,
and the gene vocabulary; :func:`~omvqvae.hf_utils.load_pretrained` rebuilds the
exact model and feature space. :func:`~omvqvae.hf_utils.push_to_hub` /
:func:`~omvqvae.hf_utils.from_pretrained` are thin Hub wrappers over the same
serialization.

Next steps
----------

- Browse the full :doc:`api` reference.
- Run the `example scripts <https://github.com/mengerj/oqae/tree/main/examples>`_
  end to end on tiny synthetic data.
