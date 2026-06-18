"""
Local AnnData (``.h5ad`` / ``.zarr``) loaders for OQAE.

This module bridges on-disk AnnData files to OQAE's shared minibatch contract:
it reads raw counts, aligns them to an organism's :class:`GeneVocabulary`, and
builds a :class:`torch.utils.data.DataLoader` that yields :class:`Minibatch`
batches — the *same* API the Census streaming loader will expose.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from torch.utils.data import DataLoader

from omvqvae.data.dataset import (
    CountsDataset,
    GeneVocabulary,
    align_to_reference,
    collate_minibatch,
)
from omvqvae.data.normalize import CountMatrix
from omvqvae.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anndata import AnnData

logger = get_logger(__name__)

__all__ = [
    "load_anndata",
    "extract_counts",
    "build_anndata_dataloader",
]


def load_anndata(
    path: Union[str, Path],
    *,
    backed: Optional[str] = None,
) -> "AnnData":
    """
    Load a local AnnData file from ``.h5ad`` or ``.zarr``.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.h5ad`` file or a ``.zarr`` store / directory.
    backed : str, optional
        Backed mode for ``.h5ad`` (e.g. ``"r"``) to read larger-than-memory
        files without loading ``X`` fully. Ignored for ``.zarr`` (which is
        chunked on disk by construction).

    Returns
    -------
    anndata.AnnData
        The loaded AnnData object.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file extension is not ``.h5ad`` or ``.zarr``.
    """
    import anndata as ad

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"AnnData path does not exist: {p}")

    suffix = p.suffix.lower()
    if suffix == ".h5ad":
        return ad.read_h5ad(p, backed=backed)
    if suffix == ".zarr":
        return ad.read_zarr(p)
    raise ValueError(
        f"Unsupported AnnData extension {suffix!r}; expected .h5ad or .zarr."
    )


def extract_counts(
    adata: "AnnData",
    *,
    layer: Optional[str] = None,
    var_key: Optional[str] = None,
) -> Tuple[CountMatrix, List[str]]:
    """
    Extract a raw-count matrix and its gene identifiers from an AnnData.

    Parameters
    ----------
    adata : anndata.AnnData
        Source object holding raw counts.
    layer : str, optional
        Name of the layer to read counts from. Defaults to ``adata.X``.
    var_key : str, optional
        Column in ``adata.var`` providing gene identifiers. Defaults to
        ``adata.var_names``.

    Returns
    -------
    tuple
        ``(matrix, gene_ids)`` where ``matrix`` is the cell x gene counts and
        ``gene_ids`` is the list of column gene identifiers.
    """
    matrix: CountMatrix = adata.layers[layer] if layer is not None else adata.X
    if var_key is not None:
        gene_ids = [str(g) for g in adata.var[var_key].tolist()]
    else:
        gene_ids = [str(g) for g in adata.var_names.tolist()]
    return matrix, gene_ids


def build_anndata_dataloader(
    source: Union[str, Path, "AnnData"],
    vocabulary: GeneVocabulary,
    *,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 0,
    layer: Optional[str] = None,
    var_key: Optional[str] = None,
    batch_key: Optional[str] = None,
    min_overlap: float = 0.0,
    size_factor_mode: str = "total",
    backed: Optional[str] = None,
) -> DataLoader:
    """
    Build a :class:`~torch.utils.data.DataLoader` over a local AnnData source.

    The returned loader yields :class:`omvqvae.data.dataset.Minibatch` objects,
    matching the contract shared with the (forthcoming) Census streaming loader.

    Parameters
    ----------
    source : str, pathlib.Path, or anndata.AnnData
        A path to a ``.h5ad`` / ``.zarr`` file, or an in-memory AnnData.
    vocabulary : GeneVocabulary
        Organism reference defining the output gene ordering.
    batch_size : int, default 128
        Minibatch size.
    shuffle : bool, default False
        Whether to shuffle cells each epoch.
    num_workers : int, default 0
        Number of DataLoader workers.
    layer : str, optional
        AnnData layer to read counts from (default ``adata.X``).
    var_key : str, optional
        ``adata.var`` column providing gene ids (default ``var_names``).
    batch_key : str, optional
        ``adata.obs`` column carried as the per-cell ``"batch"`` covariate.
    min_overlap : float, default 0.0
        Warn-below threshold for reference-gene coverage during alignment.
    size_factor_mode : str, default "total"
        Size-factor mode (see
        :func:`omvqvae.data.normalize.compute_size_factors`).
    backed : str, optional
        Backed mode forwarded to :func:`load_anndata` for ``.h5ad`` sources.

    Returns
    -------
    torch.utils.data.DataLoader
        Loader yielding :class:`Minibatch` batches aligned to ``vocabulary``.
    """
    if isinstance(source, (str, Path)):
        adata = load_anndata(source, backed=backed)
    else:
        adata = source

    matrix, gene_ids = extract_counts(adata, layer=layer, var_key=var_key)
    aligned = align_to_reference(matrix, gene_ids, vocabulary, min_overlap=min_overlap)

    batch_ids = (
        [obj for obj in adata.obs[batch_key].tolist()]
        if batch_key is not None
        else None
    )

    dataset = CountsDataset(
        aligned,
        organism=vocabulary.organism,
        batch_ids=batch_ids,
        size_factor_mode=size_factor_mode,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_minibatch,
    )
