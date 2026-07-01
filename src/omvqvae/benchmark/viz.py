"""
UMAP visualization of an OQAE latent space.

A qualitative companion to the quantitative benchmark metrics: project a latent
onto 2-D with UMAP and colour the cells by a categorical annotation (cell type,
donor, assay, …) to eyeball whether biologically distinct cells separate and
whether obvious batch structure dominates.

The discrete VQ bottleneck is **not** an obstacle: OQAE exposes two continuous
representations that UMAP handles directly — the pre-quantization latent
(:attr:`omvqvae.inference.EncodedCells.latent`) and the post-quantization latent
(``.quantized``, the codes embedded back into latent space). Plotting both side
by side (``quantized=`` below) visualizes the quantization cost measured by the
separability gap. Do **not** UMAP the raw integer codes: Euclidean distance
between codebook *indices* is meaningless (index 5 is not "closer" to 6 than to
500); one-hot + Hamming distance would be required and is out of scope here.

Requires the optional ``benchmark`` extra (``umap-learn`` + ``matplotlib``);
both are **lazy-imported** so ``import omvqvae`` stays light.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np

from omvqvae.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from omvqvae.benchmark.metrics import LatentLike

logger = get_logger(__name__)

__all__ = ["compute_umap", "plot_latent_umap"]

_MISSING_DEP_MSG = (
    "plot_latent_umap requires the optional 'umap-learn' and 'matplotlib' "
    "dependencies. Install the benchmark extra, e.g. `uv sync --extra benchmark` "
    "or `pip install 'oqae[benchmark]'`."
)


def compute_umap(
    latent: "LatentLike",
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
) -> np.ndarray:
    """
    Project a latent space to 2-D with UMAP.

    Parameters
    ----------
    latent : numpy.ndarray or torch.Tensor
        Latent vectors of shape ``(n_cells, n_latent)``.
    n_neighbors : int, default 15
        UMAP neighbourhood size.
    min_dist : float, default 0.1
        UMAP minimum-distance packing parameter.
    random_state : int, default 0
        Seed for a deterministic embedding.

    Returns
    -------
    numpy.ndarray
        2-D coordinates of shape ``(n_cells, 2)``.

    Raises
    ------
    ImportError
        If ``umap-learn`` is not installed.
    ValueError
        If ``latent`` is not 2-D.
    """
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(_MISSING_DEP_MSG) from exc

    array = np.asarray(_to_numpy(latent), dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"latent must be 2-D (n_cells, n_latent); got {array.ndim}-D.")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state
    )
    coords: np.ndarray = reducer.fit_transform(array)
    return coords


def plot_latent_umap(
    latent: "LatentLike",
    labels: Optional[Sequence[object]] = None,
    *,
    color_by: Optional[Sequence[object]] = None,
    color_by_name: str = "batch",
    quantized: Optional["LatentLike"] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
    point_size: float = 6.0,
) -> "Figure":
    """
    UMAP-embed a latent and scatter it, coloured by categorical annotations.

    One column is drawn per colouring (``labels`` and, if given, ``color_by``);
    one row per representation (the continuous ``latent`` and, if given,
    ``quantized``). The same UMAP coordinates back every colouring within a row.

    Parameters
    ----------
    latent : numpy.ndarray or torch.Tensor
        Continuous latent of shape ``(n_cells, n_latent)`` (the primary panel).
    labels : Sequence, optional
        Per-cell categorical annotation for the first colouring (e.g. cell type).
        A single unlabelled panel is drawn when omitted.
    color_by : Sequence, optional
        A second per-cell annotation (e.g. donor / assay) drawn as an additional
        column — the "batch view".
    color_by_name : str, default "batch"
        Title/legend name for the ``color_by`` column.
    quantized : numpy.ndarray or torch.Tensor, optional
        Post-quantization latent (:attr:`omvqvae.inference.EncodedCells.quantized`)
        of shape ``(n_cells, n_latent)``; drawn as a second row to visualize the
        quantization cost.
    n_neighbors, min_dist, random_state : UMAP parameters
        Forwarded to :func:`compute_umap`.
    point_size : float, default 6.0
        Scatter marker size.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure (not shown or saved; the caller decides).

    Raises
    ------
    ImportError
        If ``umap-learn`` / ``matplotlib`` are not installed.
    ValueError
        If any provided annotation length disagrees with the number of cells.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(_MISSING_DEP_MSG) from exc

    n_cells = np.asarray(_to_numpy(latent)).shape[0]
    colorings: List[Tuple[str, Sequence[object]]] = []
    if labels is not None:
        _check_length(labels, n_cells, "labels")
        colorings.append(("cell type", labels))
    if color_by is not None:
        _check_length(color_by, n_cells, "color_by")
        colorings.append((color_by_name, color_by))
    if not colorings:
        colorings.append(("cells", [0] * n_cells))

    rows: List[Tuple[str, "LatentLike"]] = [("latent", latent)]
    if quantized is not None:
        rows.append(("quantized", quantized))

    n_rows, n_cols = len(rows), len(colorings)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.0 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )
    for r, (row_name, rep) in enumerate(rows):
        coords = compute_umap(
            rep,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        for c, (col_name, groups) in enumerate(colorings):
            ax = axes[r][c]
            _scatter_by_group(ax, coords, groups, point_size=point_size)
            ax.set_title(f"{row_name} — {col_name}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    return fig


def _scatter_by_group(
    ax: "Axes",
    coords: np.ndarray,
    groups: Sequence[object],
    *,
    point_size: float,
) -> None:
    """Scatter ``coords`` with one colour per distinct value in ``groups``."""
    groups_arr = np.asarray(list(groups), dtype=object)
    unique = np.unique(groups_arr)
    for value in unique:
        mask = groups_arr == value
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            label=str(value),
            linewidths=0.0,
        )
    # Only show a legend when it stays readable.
    if 1 < unique.size <= 20:
        ax.legend(markerscale=2.0, fontsize="x-small", loc="best", frameon=False)


def _check_length(values: Sequence[object], n_cells: int, name: str) -> None:
    """Raise if ``values`` does not have exactly ``n_cells`` entries."""
    length = len(list(values))
    if length != n_cells:
        raise ValueError(f"{name} has length {length} but there are {n_cells} cells.")


def _to_numpy(latent: "LatentLike") -> np.ndarray:
    """Coerce a latent array-like to a NumPy array without importing torch."""
    if hasattr(latent, "detach"):  # torch.Tensor without importing torch here
        return latent.detach().cpu().numpy()
    return np.asarray(latent)
