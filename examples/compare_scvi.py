"""
Compare OQAE against an scVI baseline on the same cells, genes, and metrics.

OQAE's own benchmark ranks OQAE configurations but cannot tell you whether an
absolute separability of ``~0.7`` is good — that number is entirely
dataset/metric dependent. Running scVI (a well-tested external VAE) on the
**same cells, same genes, same metric** turns "0.70 in a vacuum" into a claim:
matched vs. a real gap.

This example wires both models behind the model-agnostic
:class:`~omvqvae.benchmark.baselines.LatentModel` protocol and scores them with
the shared latent metrics:

- :class:`~omvqvae.benchmark.baselines.OmvqvaeLatentModel` trains an
  ``OmicsVQVAE`` and embeds the continuous latent;
- :class:`~omvqvae.benchmark.baselines.ScviLatentModel` trains
  ``scvi.model.SCVI`` and embeds ``get_latent_representation()``;
- :func:`~omvqvae.benchmark.baselines.compare_latent_models` embeds the shared
  held-out cells through both and tabulates separability + scIB NMI/ARI/ASW.

Fairness (see ``benchmark/baselines.py``): both models see the same gene panel,
the same cells, and the same train/eval split; scVI trains **unconditionally**
(no ``batch_key``) to match v1 OQAE; and no reconstruction NLL is compared
across models (incomparable units).

This script runs on tiny **synthetic** data so it is self-contained and fast.
For the real comparison, swap :func:`make_synthetic_anndata` for a local
``.h5ad`` of raw counts (e.g. Tabula Sapiens bone marrow) aligned to a curated
gene panel — see the commented block in :func:`main`.

The scVI pieces need the optional ``baselines`` extra
(``uv sync --extra baselines``); the clustering / UMAP pieces need the
``benchmark`` extra. Run::

    python examples/compare_scvi.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from omvqvae.benchmark import (
    BenchmarkConfig,
    OmvqvaeLatentModel,
    ScviLatentModel,
    compare_latent_models,
    format_latent_comparison,
    plot_latent_umap,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure


def _synthetic_split() -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """A tiny raw-count train/eval split with program labels and gene ids."""
    from synthetic_data import make_synthetic_anndata  # sibling example helper

    adata = make_synthetic_anndata(n_cells=400, n_genes=32, n_programs=5)
    counts = np.asarray(adata.X, dtype=np.float32)
    genes = list(adata.var_names)
    labels = list(adata.obs["program"])

    rng = np.random.default_rng(0)
    perm = rng.permutation(counts.shape[0])
    n_train = int(round(0.8 * counts.shape[0]))
    train_idx, eval_idx = perm[:n_train], perm[n_train:]
    train_counts = counts[train_idx]
    eval_counts = counts[eval_idx]
    eval_labels = [labels[i] for i in eval_idx]
    return train_counts, eval_counts, genes, eval_labels, labels


#: Real-data default: a local Tabula Sapiens bone-marrow ``.h5ad`` aligned to the
#: committed 2k gene panel. Falls back to synthetic data when the file is absent
#: (e.g. in CI), so the example stays runnable everywhere.
DEFAULT_DATA_PATH = Path("/Volumes/sandisk_2tb/dev/oqae/data/tab-sap-marrow.h5ad")


def _real_split(
    data_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """Load a local ``.h5ad``, align to the 2k panel, and split train/eval."""
    from omvqvae.data import GeneVocabulary, extract_counts, load_anndata
    from omvqvae.data.anndata_io import align_to_reference

    panel_path = (
        Path(__file__).resolve().parents[1] / "resources" / "gene_selection_2k.txt"
    )
    adata = load_anndata(data_path)
    counts, gene_ids = extract_counts(adata)
    panel = [ln.strip() for ln in panel_path.read_text().splitlines() if ln.strip()]
    vocab = GeneVocabulary("homo_sapiens", panel)
    aligned = align_to_reference(counts, gene_ids, vocab, min_overlap=500)
    labels = list(adata.obs["cell_type"])

    rng = np.random.default_rng(0)
    perm = rng.permutation(aligned.shape[0])
    n_train = int(round(0.8 * aligned.shape[0]))
    train_idx, eval_idx = perm[:n_train], perm[n_train:]
    eval_labels = [labels[i] for i in eval_idx]
    return (
        aligned[train_idx],
        aligned[eval_idx],
        list(vocab.gene_ids),
        eval_labels,
        labels,
    )


def main(
    *,
    data_path: Optional[Path] = None,
    use_scvi: bool = True,
    compute_clustering: bool = True,
    make_umap: bool = True,
) -> Tuple[str, Optional["Figure"], Optional["Figure"]]:
    """
    Train OQAE (+ scVI), tabulate the shared latent metrics, and plot UMAPs.

    Parameters
    ----------
    data_path : pathlib.Path, optional
        Local ``.h5ad`` of raw counts to compare on (defaults to
        :data:`DEFAULT_DATA_PATH`). When the file is missing, the example falls
        back to the tiny synthetic split so it still runs anywhere.
    use_scvi : bool, default True
        Include the scVI baseline (needs the ``baselines`` extra). Set False to
        run the OQAE-only path on a core install.
    compute_clustering : bool, default True
        Also compute NMI / ARI / cell-type ASW (needs the ``benchmark`` extra).
    make_umap : bool, default True
        Render and return per-model latent UMAP figures.

    Returns
    -------
    (str, Figure or None, Figure or None)
        The comparison table and the OQAE / scVI UMAP figures (``None`` when a
        model is absent or ``make_umap`` is False).
    """
    path = DEFAULT_DATA_PATH if data_path is None else data_path
    if path.exists():
        train_counts, eval_counts, genes, eval_labels, _ = _real_split(path)
    else:
        print(f"{path} not found — falling back to synthetic data.")
        train_counts, eval_counts, genes, eval_labels, _ = _synthetic_split()

    # OQAE behind the protocol: a realistic-ish config trained on the raw counts.
    oqae_config = BenchmarkConfig(
        name="OQAE",
        likelihood="nb",
        n_codebooks=2,
        codebook_size=64,
        n_latent=16,
        hidden_dims=(64,),
        max_epochs=8,
    )
    oqae = OmvqvaeLatentModel(oqae_config, batch_size=128)
    oqae.fit(train_counts, genes=genes)

    models: List = [oqae]
    if use_scvi:
        # Leave max_epochs=None in real runs for scVI's own budget / early
        # stopping; a small cap keeps this synthetic demo fast.
        scvi_model = ScviLatentModel(name="scVI", n_latent=10, max_epochs=20)
        scvi_model.fit(train_counts, genes=genes)
        models.append(scvi_model)

    reports = compare_latent_models(
        models, eval_counts, eval_labels, compute_clustering=compute_clustering
    )
    table = format_latent_comparison(reports)
    print(table)

    if not make_umap:
        return table, None, None

    oqae_fig = plot_latent_umap(oqae.embed(eval_counts), eval_labels, n_neighbors=15)
    scvi_fig = None
    if use_scvi:
        scvi_fig = plot_latent_umap(
            models[1].embed(eval_counts), eval_labels, n_neighbors=15
        )
    return table, oqae_fig, scvi_fig


if __name__ == "__main__":
    _, oqae_figure, scvi_figure = main()
    if oqae_figure is not None:
        oqae_figure.savefig("oqae_umap.png", dpi=150, bbox_inches="tight")
        print("wrote oqae_umap.png")
    if scvi_figure is not None:
        scvi_figure.savefig("scvi_umap.png", dpi=150, bbox_inches="tight")
        print("wrote scvi_umap.png")
