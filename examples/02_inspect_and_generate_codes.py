"""
Example 2 — encode cells to discrete codes, inspect them, decode / generate.

This walks the user-facing latent API (:mod:`omvqvae.inference`) on a trained
model:

- **encode**: raw counts -> an :class:`~omvqvae.inference.EncodedCells` bundle
  (per-cell ``codes`` ``(n_cells, n_codebooks)`` + ``size_factors`` + the
  continuous ``latent``). ``encode_anndata`` first aligns a local AnnData to the
  model's gene vocabulary.
- **inspect**: each cell is now a tiny integer code; cells sharing a latent
  program should share codes, and the codebook should be well utilized.
- **decode**: codes (+ size factor) -> expected counts, i.e. a generative
  reconstruction. ``decode_to_params`` exposes the full likelihood parameters.
- **generate**: feed *hand-edited* codes back through the decoder to synthesize a
  novel expression profile from the shared discrete vocabulary.

Runs offline in seconds on synthetic data. Run::

    python examples/02_inspect_and_generate_codes.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from synthetic_data import make_synthetic_anndata

from omvqvae.data import GeneVocabulary, build_anndata_dataloader, extract_counts
from omvqvae.hf_utils import load_pretrained, save_pretrained
from omvqvae.inference import decode, decode_to_params, encode_anndata
from omvqvae.models import OmicsVQVAE
from omvqvae.train import TrainConfig, train


def _train_tiny_model(adata: object, vocabulary: GeneVocabulary) -> OmicsVQVAE:
    """Train a small model so the example has codes to inspect."""
    loader = build_anndata_dataloader(adata, vocabulary, batch_size=64, shuffle=True)
    model = OmicsVQVAE(
        n_genes=vocabulary.n_genes,
        n_latent=16,
        hidden_dims=(64,),
        likelihood="nb",
        codebook_size=64,
        n_codebooks=2,
    )
    train(model, loader, config=TrainConfig(max_epochs=10, lr=1e-3))
    return model


def main(workdir: Optional[Path] = None) -> None:
    """Train a tiny model, then encode / inspect / decode / generate."""
    organism = "homo_sapiens"
    adata = make_synthetic_anndata(n_cells=256, n_genes=40, organism=organism)
    _, gene_ids = extract_counts(adata)
    vocabulary = GeneVocabulary(organism, gene_ids)

    # Train then round-trip through a saved directory so we get a `LoadedModel`
    # (model + vocabulary) — exactly what `from_pretrained` returns from the Hub.
    model = _train_tiny_model(adata, vocabulary)
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="oqae_example_"))
    save_pretrained(model, vocabulary, workdir / "model")
    loaded = load_pretrained(workdir / "model")

    # --- Encode: cells -> discrete codes -------------------------------------
    # `encode_anndata` aligns the AnnData to the model's gene vocabulary first
    # (missing genes zero-filled, extra genes dropped), so it is robust to gene
    # ordering. For an already-aligned matrix use `encode(model, counts)`.
    encoded = encode_anndata(loaded, adata)
    print(f"Encoded {len(encoded)} cells")
    print(f"  codes shape       : {tuple(encoded.codes.shape)}  (n_cells, n_codebooks)")
    print(f"  size_factors shape: {tuple(encoded.size_factors.shape)}")
    print(f"  example codes     : {encoded.codes[:3].tolist()}")

    # --- Inspect: utilization + program structure ----------------------------
    n_unique = torch.unique(encoded.codes, dim=0).shape[0]
    used_level0 = int(torch.unique(encoded.codes[:, 0]).numel())
    print(
        f"  {n_unique} distinct codes across {len(encoded)} cells; "
        f"{used_level0} codebook entries used at level 0"
    )
    programs = np.asarray(adata.obs["program"])
    for program in np.unique(programs):
        rows = encoded.codes[programs == program]
        majority = torch.mode(rows[:, 0]).values.item()
        share = float((rows[:, 0] == majority).float().mean())
        print(f"  {program}: {share:.0%} share the same level-0 code ({majority})")

    # --- Decode: codes -> expected counts (generative reconstruction) --------
    reconstructed = decode(loaded.model, encoded)
    observed = torch.from_numpy(adata.X.astype(np.float32))
    corr = float(
        torch.corrcoef(torch.stack([observed.flatten(), reconstructed.flatten()]))[0, 1]
    )
    print(f"Reconstruction: counts shape {tuple(reconstructed.shape)}, corr={corr:.3f}")

    # `decode_to_params` exposes the full NB distribution (mean rate, dispersion)
    # for sampling or inspection rather than just the mean.
    params = decode_to_params(loaded.model, encoded)
    print(f"Likelihood params: {{{', '.join(params)}}}")

    # --- Generate: edit codes, decode a novel profile ------------------------
    # The codes are a composable vocabulary: change a level and decode to
    # synthesize a profile that need not correspond to any observed cell.
    novel = encoded.codes[:1].clone()
    novel[0, 0] = (novel[0, 0] + 1) % loaded.model.codebook_size
    generated = decode(loaded.model, novel, size_factors=encoded.size_factors[:1])
    print(f"Generated 1 synthetic profile from edited codes: {tuple(generated.shape)}")


if __name__ == "__main__":
    main()
