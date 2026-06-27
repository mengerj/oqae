"""
Example 1 — train (or fine-tune) OQAE on a local AnnData of raw counts.

This is the "bring-your-own-data" path: point the loader at a local
``.h5ad`` / ``.zarr`` of **raw counts**, derive the organism's
:class:`~omvqvae.data.dataset.GeneVocabulary` from the file's own genes, train an
:class:`~omvqvae.models.vqvae.OmicsVQVAE` with the source-agnostic
:func:`omvqvae.train.train` loop, and save a Hub-ready model directory with
:func:`omvqvae.hf_utils.save_pretrained`.

The dataset here is tiny synthetic counts (see ``synthetic_data.py``) so the
example runs offline in seconds; replace :func:`make_synthetic_anndata` with
``omvqvae.data.load_anndata("your_cells.h5ad")`` for real data. The same run is
also expressible as a one-liner with the ``oqae-train`` CLI::

    oqae-train configs/train_toy.yaml -s data.path=your_cells.h5ad

Run::

    python examples/01_train_local_anndata.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from synthetic_data import make_synthetic_anndata

from omvqvae.data import GeneVocabulary, build_anndata_dataloader, extract_counts
from omvqvae.hf_utils import save_pretrained
from omvqvae.models import OmicsVQVAE
from omvqvae.train import TrainConfig, train


def main(save_directory: Optional[Path] = None) -> Path:
    """
    Train a small OQAE model on synthetic counts and save it.

    Parameters
    ----------
    save_directory : pathlib.Path, optional
        Where to write the Hub-ready model directory. A temporary directory is
        used when omitted.

    Returns
    -------
    pathlib.Path
        The directory the trained model was saved to (``config.json`` +
        ``pytorch_model.bin``).
    """
    organism = "homo_sapiens"

    # 1. Get raw counts. Swap this for `omvqvae.data.load_anndata(path)`.
    adata = make_synthetic_anndata(n_cells=256, n_genes=40, organism=organism)

    # 2. The reference gene space is the organism's vocabulary. For a local file
    #    we derive it from the file's own genes; a Census run would read it from
    #    the Census `var` index instead.
    _, gene_ids = extract_counts(adata)
    vocabulary = GeneVocabulary(organism, gene_ids)
    print(f"Vocabulary: {organism} with {vocabulary.n_genes} genes")

    # 3. Build a DataLoader of `Minibatch`es (raw counts + size factors). This is
    #    the *same* contract the Census streaming loader yields.
    loader = build_anndata_dataloader(
        adata,
        vocabulary,
        batch_size=64,
        shuffle=True,
        batch_key="batch",
    )

    # 4. Model: raw counts -> encoder -> residual VQ -> decoder -> NB likelihood.
    model = OmicsVQVAE(
        n_genes=vocabulary.n_genes,
        n_latent=16,
        hidden_dims=(64,),
        likelihood="nb",
        codebook_size=64,
        n_codebooks=2,
    )

    # 5. Train. The loop is source-agnostic and logs through a console tracker by
    #    default (set up a W&B tracker for real runs).
    result = train(model, loader, config=TrainConfig(max_epochs=8, lr=1e-3))
    last = result.last_epoch
    assert last is not None
    print(
        f"Trained {result.global_step} steps | "
        f"final loss={last.loss:.4g} recon={last.reconstruction_loss:.4g} "
        f"vq={last.vq_loss:.4g} perplexity={last.perplexity:.4g}"
    )

    # 6. Save a Hub-ready directory (weights + architecture + gene vocabulary).
    #    `omvqvae.hf_utils.push_to_hub(model, vocabulary, "user/oqae-human")`
    #    would upload the same artifacts to the HuggingFace Hub.
    if save_directory is None:
        save_directory = Path(tempfile.mkdtemp(prefix="oqae_example_")) / "model"
    out = save_pretrained(model, vocabulary, save_directory)
    print(f"Saved model to {out}")
    return out


if __name__ == "__main__":
    main()
