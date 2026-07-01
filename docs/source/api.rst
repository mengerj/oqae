API reference
=============

Auto-generated reference for the public :mod:`omvqvae` API. Each section
documents the implementation modules behind a subpackage; the subpackage
``__init__`` re-exports these names (e.g. ``from omvqvae.data import
GeneVocabulary``).

Data layer (:mod:`omvqvae.data`)
--------------------------------

Organism-aware loading of raw single-cell counts onto a shared ``Minibatch``
contract — local AnnData and CELLxGENE Census streaming, plus gene-space
alignment and normalization.

.. automodule:: omvqvae.data.dataset
   :members:

.. automodule:: omvqvae.data.normalize
   :members:

.. automodule:: omvqvae.data.anndata_io
   :members:

.. automodule:: omvqvae.data.census
   :members:

Layers (:mod:`omvqvae.layers`)
------------------------------

The residual vector-quantization bottleneck.

.. automodule:: omvqvae.layers.residual_vq
   :members:

Models (:mod:`omvqvae.models`)
------------------------------

Reconstruction likelihoods / decoder heads and the end-to-end VQ-VAE.

.. automodule:: omvqvae.models.likelihoods
   :members:

.. automodule:: omvqvae.models.vqvae
   :members:

Training (:mod:`omvqvae.train`)
-------------------------------

The source-agnostic training loop and the config-driven CLI.

.. automodule:: omvqvae.train.loop
   :members:

.. automodule:: omvqvae.train.cli
   :members:

Inference (:mod:`omvqvae.inference`)
------------------------------------

The discrete-code latent API: encode expression into codes and decode codes
back into expression.

.. automodule:: omvqvae.inference.codes
   :members:

Benchmarking (:mod:`omvqvae.benchmark`)
---------------------------------------

Offline-runnable harness to compare likelihood / codebook configurations:
reconstruction quality, codebook utilization, and downstream separability, with
a comparison-table report.

.. automodule:: omvqvae.benchmark.metrics
   :members:

.. automodule:: omvqvae.benchmark.harness
   :members:

.. automodule:: omvqvae.benchmark.report
   :members:

.. automodule:: omvqvae.benchmark.throughput
   :members:

.. automodule:: omvqvae.benchmark.baselines
   :members:

Serialization (:mod:`omvqvae.hf_utils`)
---------------------------------------

HuggingFace Hub-style model serialization.

.. automodule:: omvqvae.hf_utils
   :members:

Experiment tracking (:mod:`omvqvae.utils.tracking`)
---------------------------------------------------

Offline-friendly experiment tracking (console / Weights & Biases).

.. automodule:: omvqvae.utils.tracking
   :members:
