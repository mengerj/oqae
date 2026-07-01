"""
Example 7 — latent-quality evaluation (quantization gap, scIB metrics, UMAP).

Extends the config sweep of example 4 with the offline latent-quality metrics
from issue #36, so you can judge a model beyond nearest-centroid separability:

- the **quantization-cost** view — ``separability`` (continuous latent) vs
  ``sep_quant`` (post-quantization latent) and their ``sep_gap``;
- **scIB clustering** metrics (NMI / ARI / cell-type ASW) via
  ``compute_clustering=True``;
- a **UMAP** of the trained latent (continuous vs post-quantization), coloured by
  the known "program" and by "batch".

The clustering + UMAP pieces need the optional ``benchmark`` extra
(``uv sync --extra benchmark``); pass ``compute_clustering=False`` /
``make_umap=False`` to skip them on a core install.

Run::

    python examples/07_latent_quality.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from synthetic_data import make_synthetic_anndata

from omvqvae.benchmark import (
    BenchmarkConfig,
    format_results_table,
    plot_latent_umap,
    run_suite,
)
from omvqvae.data import GeneVocabulary, build_anndata_dataloader, extract_counts
from omvqvae.inference import encode
from omvqvae.models.vqvae import OmicsVQVAE
from omvqvae.train import TrainConfig, train

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure


def main(
    *,
    compute_clustering: bool = True,
    make_umap: bool = True,
) -> Tuple[str, Optional["Figure"]]:
    """
    Sweep a couple of configs, tabulate the quality metrics, and (optionally) plot.

    Parameters
    ----------
    compute_clustering : bool, default True
        Also compute NMI / ARI / cell-type ASW (needs the ``benchmark`` extra).
    make_umap : bool, default True
        Train the best config once more and render a latent UMAP figure.

    Returns
    -------
    (str, matplotlib.figure.Figure or None)
        The rendered comparison table and the UMAP figure (``None`` when
        ``make_umap`` is False).
    """
    organism = "homo_sapiens"
    adata = make_synthetic_anndata(
        n_cells=256, n_genes=40, n_programs=4, organism=organism
    )
    counts, gene_ids = extract_counts(adata)
    vocabulary = GeneVocabulary(organism, gene_ids)
    labels = list(adata.obs["program"])
    batches = list(adata.obs["batch"])
    loader = build_anndata_dataloader(
        adata, vocabulary, batch_size=64, shuffle=True, batch_key="batch"
    )

    configs: List[BenchmarkConfig] = [
        BenchmarkConfig(
            name="nb-2x64", likelihood="nb", n_codebooks=2, codebook_size=64
        ),
        BenchmarkConfig(
            name="nb-1x16", likelihood="nb", n_codebooks=1, codebook_size=16
        ),
    ]

    # The quantization gap + scIB metrics land in the table (sep_quant / sep_gap;
    # nmi / ari / ct_asw when clustering is requested).
    results = run_suite(
        configs,
        loader,
        n_genes=vocabulary.n_genes,
        eval_counts=counts,
        eval_labels=labels,
        compute_clustering=compute_clustering,
    )
    table = format_results_table(results)
    print(table)

    if not make_umap:
        return table, None

    # run_suite does not return the trained models, so re-fit the lowest-NLL
    # config to grab its latent for the figure.
    best = min(results, key=lambda r: r.eval.reconstruction.nll).config
    model = OmicsVQVAE(vocabulary.n_genes, **best.model_kwargs())
    train(model, loader, config=TrainConfig(max_epochs=best.max_epochs))
    encoded = encode(model, counts)
    fig = plot_latent_umap(
        encoded.latent,
        labels,
        color_by=batches,
        quantized=encoded.quantized,
        n_neighbors=15,
    )
    return table, fig


if __name__ == "__main__":
    _, figure = main()
    if figure is not None:
        figure.savefig("latent_umap.png", dpi=150, bbox_inches="tight")
        print("wrote latent_umap.png")
