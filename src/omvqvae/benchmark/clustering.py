"""
Clustering-based biology metrics for the OQAE benchmarking harness.

The harness's built-in :func:`omvqvae.benchmark.metrics.separability_score` is a
fast, dependency-free *nearest-centroid* proxy: it only rewards convex,
roughly-spherical label clusters and under-reports quality on continuum tissues
(e.g. bone-marrow progenitors, whose class means sit close together). This module
adds the stronger, community-standard measures used by scIB — NMI / ARI between
labels and an unsupervised clustering of the latent, plus the cell-type average
silhouette width — which capture non-convex structure.

These metrics require the optional :mod:`scib_metrics` dependency (installed via
the ``benchmark`` extra: ``uv sync --extra benchmark`` /
``pip install 'oqae[benchmark]'``). It is a heavy dependency, so it is
**lazy-imported** inside :func:`clustering_metrics`; ``import omvqvae`` and the
offline default test suite never touch it.

Everything here operates on a plain ``(n_cells, n_latent)`` embedding, so it runs
identically over the OQAE continuous latent, the OQAE post-quantization latent
(see :attr:`omvqvae.inference.EncodedCells.quantized`), and any external
baseline's latent — the shared metric that makes those comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from omvqvae.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omvqvae.benchmark.metrics import LatentLike

logger = get_logger(__name__)

__all__ = [
    "ClusteringMetrics",
    "clustering_metrics",
]

_MISSING_DEP_MSG = (
    "clustering_metrics requires the optional 'scib-metrics' dependency. "
    "Install the benchmark extra, e.g. `uv sync --extra benchmark` or "
    "`pip install 'oqae[benchmark]'`."
)


@dataclass
class ClusteringMetrics:
    """
    Clustering-agreement biology metrics for a latent space given labels.

    Attributes
    ----------
    nmi : float
        Normalized mutual information between the labels and a KMeans clustering
        of the latent, in ``[0, 1]`` (higher = the clustering recovers the label
        structure).
    ari : float
        Adjusted Rand index between the labels and the same clustering, in
        ``[-0.5, 1]`` (higher = better; ``0`` ≈ chance).
    cell_type_asw : float
        Cell-type average silhouette width, rescaled to ``[0, 1]`` (higher =
        labels form tighter, better-separated clusters in the latent).
    """

    nmi: float
    ari: float
    cell_type_asw: float


def clustering_metrics(
    latent: "LatentLike",
    labels: Sequence[object],
) -> ClusteringMetrics:
    """
    Compute NMI / ARI / cell-type ASW for a latent space against known labels.

    Parameters
    ----------
    latent : numpy.ndarray or torch.Tensor
        Latent vectors of shape ``(n_cells, n_latent)`` — e.g. the OQAE
        continuous latent (:attr:`omvqvae.inference.EncodedCells.latent`), the
        post-quantization latent (``.quantized``), or a baseline's embedding.
    labels : Sequence
        Per-cell labels of length ``n_cells`` (e.g. cell type).

    Returns
    -------
    ClusteringMetrics
        NMI, ARI (both from a KMeans clustering of the latent), and the rescaled
        cell-type silhouette width. All three are ``nan`` when there are fewer
        than two distinct labels (undefined).

    Raises
    ------
    ImportError
        If the optional ``scib-metrics`` dependency is not installed.
    ValueError
        If ``latent`` is not 2-D or ``labels`` length disagrees with the number
        of cells.

    Notes
    -----
    The clustering is KMeans-based (scIB's ``*_kmeans`` variant), which needs no
    graph-clustering backend (igraph/leiden) and so keeps the extra light and
    deterministic. Metrics are computed over all cells (no held-out split); use
    them for relative comparison across configurations / baselines.
    """
    try:
        from scib_metrics import (
            nmi_ari_cluster_labels_kmeans,
            silhouette_label,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(_MISSING_DEP_MSG) from exc

    array = np.asarray(_to_numpy(latent), dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"latent must be 2-D (n_cells, n_latent); got {array.ndim}-D.")
    labels_arr = np.asarray(list(labels), dtype=object)
    if labels_arr.shape[0] != array.shape[0]:
        raise ValueError(
            f"labels has length {labels_arr.shape[0]} but there are "
            f"{array.shape[0]} cells."
        )
    # Encode labels to contiguous integer codes (scib_metrics expects a numeric
    # label vector); NMI/ARI/ASW are invariant to the encoding.
    _, label_codes = np.unique(labels_arr, return_inverse=True)
    if np.unique(label_codes).size < 2:
        return ClusteringMetrics(
            nmi=float("nan"), ari=float("nan"), cell_type_asw=float("nan")
        )

    nmi_ari = nmi_ari_cluster_labels_kmeans(array, label_codes)
    asw = float(silhouette_label(array, label_codes))
    return ClusteringMetrics(
        nmi=float(nmi_ari["nmi"]),
        ari=float(nmi_ari["ari"]),
        cell_type_asw=asw,
    )


def _to_numpy(latent: "LatentLike") -> np.ndarray:
    """Coerce a latent array-like to a NumPy array without importing torch."""
    if hasattr(latent, "detach"):  # torch.Tensor without importing torch here
        return latent.detach().cpu().numpy()
    return np.asarray(latent)
