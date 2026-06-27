"""
Example 3 — train OQAE by streaming from the CZ CELLxGENE Census.

This is the at-scale path: instead of holding data in memory, stream raw counts
straight from the Census via TileDB-SOMA. An ``obs_value_filter`` selects the
slice of cells (here a small tissue/assay subset so the example stays light), the
per-organism :class:`~omvqvae.data.dataset.GeneVocabulary` is read from the
Census ``var`` index, and the resulting loader yields the *same*
:class:`~omvqvae.data.dataset.Minibatch` contract as the local-AnnData loader —
so the model and training loop are identical to example 1.

.. note::

   **This example requires network access** to the CELLxGENE Census and is
   therefore *not* run in CI (the offline examples 1 and 2 cover the API). It is
   written to be runnable manually::

       python examples/03_census_streaming.py

   For a heavier run, drop the ``obs_value_filter`` / raise ``max_steps`` and add
   a W&B tracker. The Census version is pinned for reproducibility.
"""

from __future__ import annotations

from omvqvae.data import (
    DEFAULT_CENSUS_VERSION,
    build_census_dataloader,
    census_gene_vocabulary,
    open_census,
)
from omvqvae.models import OmicsVQVAE
from omvqvae.train import TrainConfig, train


def main() -> None:  # pragma: no cover - requires live Census/TileDB-SOMA
    """Stream a small human Census slice and train a few steps on it."""
    organism = "homo_sapiens"

    # A narrow slice keeps the example light; widen the filter for real training.
    obs_value_filter = (
        "tissue_general == 'blood' and is_primary_data == True "
        "and assay == '10x 3\\' v3'"
    )

    with open_census(census_version=DEFAULT_CENSUS_VERSION) as census:
        # The reference gene space is the Census `var` index for the organism.
        vocabulary = census_gene_vocabulary(census, organism)
        print(f"Census {DEFAULT_CENSUS_VERSION}: {vocabulary.n_genes} {organism} genes")

        # Streaming loader of `Minibatch`es — identical contract to example 1.
        loader = build_census_dataloader(
            census,
            organism,
            vocabulary,
            obs_value_filter=obs_value_filter,
            batch_size=128,
            shuffle=True,
            seed=0,
        )

        model = OmicsVQVAE(
            n_genes=vocabulary.n_genes,
            n_latent=32,
            hidden_dims=(256, 128),
            likelihood="nb",
            codebook_size=512,
            n_codebooks=2,
        )

        # `max_steps` caps the smoke run; the loader stays open for the duration.
        result = train(
            model,
            loader,
            config=TrainConfig(max_epochs=1, max_steps=20, lr=1e-3),
        )
        last = result.last_epoch
        assert last is not None
        print(
            f"Streamed {result.global_step} steps | "
            f"loss={last.loss:.4g} perplexity={last.perplexity:.4g}"
        )


if __name__ == "__main__":
    main()
