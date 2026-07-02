"""
Model-agnostic latent baselines for the OQAE benchmarking harness.

OQAE's benchmark tells us the *relatively* best OQAE configuration, but a bare
separability of ``0.70`` is uninterpretable on its own — the number is entirely
dataset/metric dependent. Running a well-tested external VAE on the **same
cells, same genes, same metric** turns that into a claim: OQAE ``0.70`` vs scVI
``0.72`` means "essentially matched, the dataset is just hard", while ``0.70``
vs ``0.85`` means "a real gap to close".

scVI does not fit :class:`~omvqvae.models.vqvae.OmicsVQVAE` /
:class:`~omvqvae.benchmark.harness.BenchmarkConfig`, and we should not force it
to. Every metric that transfers across models needs only a **latent embedding**,
so this module defines a thin :class:`LatentModel` protocol and adapts both
models to it:

- :class:`OmvqvaeLatentModel` wraps a trained ``OmicsVQVAE``; ``embed`` returns
  the continuous pre-quantization latent (or, opt-in, the post-quantization
  ``.quantized`` variant for the quantization-gap view).
- :class:`ScviLatentModel` wraps ``scvi.model.SCVI``; ``embed`` returns
  ``get_latent_representation()``. ``scvi-tools`` is **lazy-imported** (project
  idiom) and only pulled in via the optional ``baselines`` extra.

:func:`compare_latent_models` then runs the **same** shared latent metrics
(:func:`~omvqvae.benchmark.metrics.separability_score`,
:func:`~omvqvae.benchmark.clustering.clustering_metrics`) over every model and
emits a tidy one-row-per-model comparison table.

Fairness constraints (read before comparing):

- **Do not compare reconstruction NLL across models.** Different
  likelihood/normalization conventions make the units incomparable; only
  latent-based metrics (separability, NMI/ARI, UMAP) transfer. This module
  deliberately reports no reconstruction metric.
- **Match covariate treatment.** v1 OQAE is unconditional, so the scVI adapter
  trains with **no ``batch_key``** by default. Feed both models the same gene
  panel, the same cells, and the same train/eval split.
- **Give scVI a realistic training budget.** Leave ``max_epochs=None`` so scVI
  uses its own heuristic / early stopping, rather than the tiny epoch counts
  used for OQAE smoke sweeps — otherwise the baseline is strawmanned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np

from omvqvae.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omvqvae.benchmark.harness import BenchmarkConfig
    from omvqvae.data.normalize import CountMatrix
    from omvqvae.models.vqvae import OmicsVQVAE

logger = get_logger(__name__)

__all__ = [
    "LatentModel",
    "OmvqvaeLatentModel",
    "ScviLatentModel",
    "LatentModelReport",
    "compare_latent_models",
    "latent_comparison_to_dicts",
    "format_latent_comparison",
]

_SCVI_MISSING_MSG = (
    "ScviLatentModel requires the optional 'scvi-tools' dependency. Install the "
    "baselines extra, e.g. `uv sync --extra baselines` or "
    "`pip install 'oqae[baselines]'`."
)


@runtime_checkable
class LatentModel(Protocol):
    """
    Minimal interface every latent baseline exposes for the comparison.

    A ``LatentModel`` is any object that can be **fit** on a raw-count training
    matrix and then **embed** raw counts into a continuous latent space. That is
    all the shared latent metrics need, which is what lets OQAE and an external
    VAE such as scVI be compared on equal footing.

    Attributes
    ----------
    name : str
        Human-readable label for the model's row in the comparison table.
    """

    name: str

    def fit(
        self,
        train_counts: "CountMatrix",
        *,
        genes: Sequence[str],
        labels: Optional[Sequence[object]] = None,
    ) -> None:
        """Train the model on raw counts ``(n_cells, n_genes)``."""
        ...

    def embed(self, counts: "CountMatrix") -> np.ndarray:
        """Embed raw counts into a ``(n_cells, d)`` latent array."""
        ...


def _counts_to_array(counts: "CountMatrix") -> np.ndarray:
    """Coerce a dense/sparse/tensor count matrix to a 2-D float32 NumPy array."""
    from omvqvae.data.normalize import to_dense

    dense = to_dense(counts)
    array = np.asarray(dense, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"counts must be 2-D (n_cells, n_genes); got {array.ndim}-D.")
    return array


class OmvqvaeLatentModel:
    """
    OQAE adapter: train an :class:`~omvqvae.models.vqvae.OmicsVQVAE` and embed.

    Two construction paths:

    - Pass a :class:`~omvqvae.benchmark.harness.BenchmarkConfig` and call
      :meth:`fit` — the adapter builds and trains a fresh model on the training
      counts (the usual "train OQAE and scVI on the same data" comparison).
    - Pass an already-``trained_model`` — :meth:`fit` becomes a no-op (beyond a
      feature-size check), so a model trained elsewhere can be dropped straight
      into the comparison.

    Parameters
    ----------
    config : BenchmarkConfig, optional
        Architecture / training knobs used to build and train the model in
        :meth:`fit`. Mutually exclusive with ``trained_model``.
    trained_model : OmicsVQVAE, optional
        A pre-trained model to wrap directly. Mutually exclusive with ``config``.
    name : str, optional
        Row label; defaults to ``config.name`` (or ``"oqae"`` for a wrapped
        model).
    organism : str, default "homo_sapiens"
        Organism id recorded on the training minibatches.
    batch_size : int, default 128
        Minibatch size for the training ``DataLoader`` built in :meth:`fit`.
    use_quantized : bool, default False
        When True, :meth:`embed` returns the **post-quantization** latent (the
        codes embedded back into latent space) instead of the continuous
        pre-quantization latent — the "what the discrete bottleneck keeps" view.

    Raises
    ------
    ValueError
        If neither or both of ``config`` / ``trained_model`` are given.
    """

    def __init__(
        self,
        config: Optional["BenchmarkConfig"] = None,
        *,
        trained_model: Optional["OmicsVQVAE"] = None,
        name: Optional[str] = None,
        organism: str = "homo_sapiens",
        batch_size: int = 128,
        use_quantized: bool = False,
    ) -> None:
        if (config is None) == (trained_model is None):
            raise ValueError("Provide exactly one of `config` or `trained_model`.")
        self.config = config
        self.organism = organism
        self.batch_size = batch_size
        self.use_quantized = use_quantized
        self._model: Optional["OmicsVQVAE"] = trained_model
        if name is not None:
            self.name = name
        elif config is not None:
            self.name = config.name
        else:
            self.name = "oqae"

    def fit(
        self,
        train_counts: "CountMatrix",
        *,
        genes: Sequence[str],
        labels: Optional[Sequence[object]] = None,
    ) -> None:
        """
        Train an ``OmicsVQVAE`` on ``train_counts`` (no-op if wrapping a model).

        ``labels`` are ignored (OQAE is unsupervised). ``genes`` only fixes the
        feature-space size; a wrapped pre-trained model is validated against it.
        """
        counts = _counts_to_array(train_counts)
        n_genes = counts.shape[1]
        if len(genes) != n_genes:
            raise ValueError(
                f"genes has length {len(genes)} but train_counts has {n_genes} "
                "columns."
            )
        if self.config is None:
            # Wrapping an already-trained model: only validate the feature space.
            # __init__ guarantees a model is set whenever config is None.
            model = self._model
            if model is not None and model.n_genes != n_genes:
                raise ValueError(
                    f"wrapped model expects {model.n_genes} genes but "
                    f"train_counts has {n_genes}."
                )
            return

        import torch
        from torch.utils.data import DataLoader

        from omvqvae.data.dataset import CountsDataset, collate_minibatch
        from omvqvae.models.vqvae import OmicsVQVAE
        from omvqvae.train import TrainConfig, train
        from omvqvae.utils.tracking import ConsoleTracker

        torch.manual_seed(self.config.seed)
        dataset = CountsDataset(counts, self.organism)
        loader: Any = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_minibatch,
        )
        model = OmicsVQVAE(n_genes, **self.config.model_kwargs())
        train_config = TrainConfig(
            max_epochs=self.config.max_epochs,
            lr=self.config.lr,
            grad_clip_norm=self.config.grad_clip_norm,
            max_steps=self.config.max_steps,
        )
        logger.info("Fitting OmvqvaeLatentModel %r (%s).", self.name, n_genes)
        train(
            model,
            loader,
            config=train_config,
            tracker=ConsoleTracker(run_name=self.name),
        )
        self._model = model

    def embed(self, counts: "CountMatrix") -> np.ndarray:
        """
        Embed raw counts into the ``(n_cells, n_latent)`` OQAE latent.

        Returns the continuous pre-quantization latent, or the post-quantization
        latent when ``use_quantized`` is set.

        Raises
        ------
        RuntimeError
            If the model has not been fit yet.
        """
        if self._model is None:
            raise RuntimeError("OmvqvaeLatentModel.embed called before fit().")
        from omvqvae.inference import encode

        encoded = encode(self._model, counts)
        rep = encoded.quantized if self.use_quantized else encoded.latent
        return np.asarray(rep.detach().cpu().numpy(), dtype=np.float64)


class ScviLatentModel:
    """
    scVI adapter: train ``scvi.model.SCVI`` and embed via its latent space.

    ``scvi-tools`` is **lazy-imported** inside :meth:`fit` / :meth:`embed`, so it
    is only required when this baseline is actually used (install the optional
    ``baselines`` extra). Constructing the object without ``scvi-tools`` is fine;
    calling :meth:`fit` without it raises a clear, actionable error.

    Parameters
    ----------
    name : str, default "scVI"
        Row label in the comparison table.
    n_latent : int, default 10
        scVI latent dimensionality (its default; need not match OQAE's — the
        latent metrics are dimension-agnostic).
    max_epochs : int, optional
        Passed to ``SCVI.train``. Leave ``None`` (default) so scVI uses its own
        epoch heuristic / early stopping — a realistic budget, not the tiny epoch
        counts used for OQAE smoke sweeps (see the fairness note in the module
        docstring).
    seed : int, default 0
        Seed passed to ``scvi.settings.seed`` before training for reproducibility.
    train_kwargs : dict, optional
        Extra keyword arguments forwarded to ``SCVI.train`` (e.g.
        ``{"early_stopping": True}``).
    model_kwargs : dict, optional
        Extra keyword arguments forwarded to the ``SCVI`` constructor (e.g.
        ``gene_likelihood``, ``n_layers``).

    Notes
    -----
    The adapter trains **unconditionally** (no ``batch_key``) to match v1 OQAE.
    The protocol's ``fit`` carries no covariate channel, which is exactly why
    conditioning is out of scope here; document any covariate difference if you
    change this.
    """

    def __init__(
        self,
        *,
        name: str = "scVI",
        n_latent: int = 10,
        max_epochs: Optional[int] = None,
        seed: int = 0,
        train_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.n_latent = n_latent
        self.max_epochs = max_epochs
        self.seed = seed
        self.train_kwargs = dict(train_kwargs or {})
        self.model_kwargs = dict(model_kwargs or {})
        self._model: Any = None
        self._genes: Optional[List[str]] = None

    def _build_anndata(  # pragma: no cover - exercised on the live scVI path
        self, counts: "CountMatrix", genes: Sequence[str]
    ) -> Any:
        """Wrap a raw-count matrix + gene ids in an AnnData scVI can consume."""
        import pandas as pd  # lazy: anndata/pandas only needed on the scVI path
        from anndata import AnnData

        array = _counts_to_array(counts)
        if array.shape[1] != len(genes):
            raise ValueError(
                f"counts has {array.shape[1]} genes but {len(genes)} gene ids "
                "were given."
            )
        var = pd.DataFrame(index=[str(g) for g in genes])
        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(array.shape[0])])
        return AnnData(X=array, obs=obs, var=var)

    def fit(
        self,
        train_counts: "CountMatrix",
        *,
        genes: Sequence[str],
        labels: Optional[Sequence[object]] = None,
    ) -> None:
        """
        Train scVI on raw counts ``(n_cells, n_genes)``.

        ``labels`` are ignored (scVI is unsupervised here, matching unconditional
        OQAE). ``genes`` fixes the feature space; the same ids must be supplied to
        :meth:`embed`.

        Raises
        ------
        ImportError
            If the optional ``scvi-tools`` dependency is not installed.
        """
        try:
            import scvi
        except ImportError as exc:
            raise ImportError(_SCVI_MISSING_MSG) from exc

        # The rest is the live scVI training shell (heavy; needs the extra), so it
        # is exercised only under the network-marked integration test.
        scvi.settings.seed = self.seed  # pragma: no cover
        self._genes = [str(g) for g in genes]  # pragma: no cover
        adata = self._build_anndata(train_counts, self._genes)  # pragma: no cover
        scvi.model.SCVI.setup_anndata(adata)  # pragma: no cover
        self._model = scvi.model.SCVI(  # pragma: no cover
            adata, n_latent=self.n_latent, **self.model_kwargs
        )
        logger.info(  # pragma: no cover
            "Fitting ScviLatentModel %r (n_latent=%d, max_epochs=%s).",
            self.name,
            self.n_latent,
            self.max_epochs,
        )
        self._model.train(  # pragma: no cover
            max_epochs=self.max_epochs, **self.train_kwargs
        )

    def embed(self, counts: "CountMatrix") -> np.ndarray:
        """
        Embed raw counts into scVI's ``(n_cells, n_latent)`` latent space.

        Raises
        ------
        RuntimeError
            If the model has not been fit yet.
        """
        if self._model is None or self._genes is None:
            raise RuntimeError("ScviLatentModel.embed called before fit().")
        # Live scVI embedding shell (needs a fitted model + the extra).
        adata = self._build_anndata(counts, self._genes)  # pragma: no cover
        latent = self._model.get_latent_representation(adata)  # pragma: no cover
        return np.asarray(latent, dtype=np.float64)  # pragma: no cover


@dataclass
class LatentModelReport:
    """
    One model's row in the latent comparison.

    Attributes
    ----------
    name : str
        The model's display name.
    n_latent : int
        Embedding dimensionality returned by ``embed``.
    separability : float
        Nearest-centroid separability of the embedding given the labels
        (:func:`~omvqvae.benchmark.metrics.separability_score`), or ``nan`` when
        undefined.
    nmi : float
        scIB NMI between the labels and a KMeans clustering of the embedding, or
        ``nan`` when clustering was not requested.
    ari : float
        scIB ARI for the same clustering, or ``nan``.
    cell_type_asw : float
        Cell-type average silhouette width, rescaled to ``[0, 1]``, or ``nan``.
    """

    name: str
    n_latent: int
    separability: float
    nmi: float = float("nan")
    ari: float = float("nan")
    cell_type_asw: float = float("nan")


def compare_latent_models(
    models: Sequence[LatentModel],
    eval_counts: "CountMatrix",
    labels: Sequence[object],
    *,
    batch_key: Optional[Sequence[object]] = None,
    compute_clustering: bool = True,
) -> List[LatentModelReport]:
    """
    Embed shared held-out cells through each model and score the latent spaces.

    Every model must already be **fit**. Each one embeds the *same*
    ``eval_counts`` and is scored with the *same* latent metrics, so the rows are
    directly comparable — the whole point of the baseline. Reconstruction NLL is
    intentionally **not** reported: its units are not comparable across different
    likelihoods/normalizations (see the module docstring's fairness note).

    Parameters
    ----------
    models : Sequence[LatentModel]
        Fitted models to compare (e.g. an :class:`OmvqvaeLatentModel` and a
        :class:`ScviLatentModel`).
    eval_counts : numpy.ndarray or scipy.sparse.spmatrix
        Shared held-out raw counts ``(n_cells, n_genes)`` embedded by every model.
    labels : Sequence
        Per-cell labels (e.g. cell type) used for every latent metric.
    batch_key : Sequence, optional
        Per-cell batch annotations for the same cells. Currently reserved for the
        UMAP batch view (pass it to
        :func:`~omvqvae.benchmark.viz.plot_latent_umap`'s ``color_by``); it does
        **not** change the scored metrics. Kept in the signature so the covariate
        channel is explicit and available to callers.
    compute_clustering : bool, default True
        Also compute the scIB NMI / ARI / cell-type-ASW metrics (needs the
        optional ``benchmark`` extra). When False, only ``separability`` is
        populated and the rest stay ``nan`` (a dependency-light path).

    Returns
    -------
    List[LatentModelReport]
        One report per model, in input order.

    Raises
    ------
    ValueError
        If an embedding's row count disagrees with ``labels`` / ``eval_counts``.
    """
    from omvqvae.benchmark.metrics import separability_score

    n_labels = len(list(labels))
    if batch_key is not None and len(list(batch_key)) != n_labels:
        raise ValueError(
            f"batch_key has length {len(list(batch_key))} but labels has "
            f"{n_labels}."
        )

    reports: List[LatentModelReport] = []
    for model in models:
        latent = np.asarray(model.embed(eval_counts))
        if latent.ndim != 2:
            raise ValueError(
                f"model {model.name!r} embed returned a {latent.ndim}-D array; "
                "expected 2-D (n_cells, d)."
            )
        if latent.shape[0] != n_labels:
            raise ValueError(
                f"model {model.name!r} embedded {latent.shape[0]} cells but "
                f"labels has {n_labels}."
            )
        separability = separability_score(latent, labels)
        nmi = ari = cell_type_asw = float("nan")
        if compute_clustering:
            from omvqvae.benchmark.clustering import clustering_metrics

            cluster = clustering_metrics(latent, labels)
            nmi, ari, cell_type_asw = (
                cluster.nmi,
                cluster.ari,
                cluster.cell_type_asw,
            )
        reports.append(
            LatentModelReport(
                name=model.name,
                n_latent=int(latent.shape[1]),
                separability=separability,
                nmi=nmi,
                ari=ari,
                cell_type_asw=cell_type_asw,
            )
        )
    return reports


def latent_comparison_to_dicts(
    reports: Sequence[LatentModelReport],
) -> List[Dict[str, Any]]:
    """
    Flatten :class:`LatentModelReport` rows into plain dicts (CSV / DataFrame).

    Parameters
    ----------
    reports : Sequence[LatentModelReport]
        Reports to flatten.

    Returns
    -------
    List[Dict[str, Any]]
        One scalar-valued row per report.
    """
    return [
        {
            "name": report.name,
            "n_latent": report.n_latent,
            "separability": report.separability,
            "nmi": report.nmi,
            "ari": report.ari,
            "cell_type_asw": report.cell_type_asw,
        }
        for report in reports
    ]


def format_latent_comparison(reports: Sequence[LatentModelReport]) -> str:
    """
    Render latent-model reports as a Markdown comparison table.

    Parameters
    ----------
    reports : Sequence[LatentModelReport]
        Reports to tabulate (one row each).

    Returns
    -------
    str
        A GitHub-flavoured Markdown table; ``"(no models)"`` if empty. The scIB
        clustering columns appear only when at least one report computed them.
    """
    if not reports:
        return "(no models)"

    show_clustering = any(report.nmi == report.nmi for report in reports)  # not nan

    headers = ["model", "n_latent", "separability"]
    if show_clustering:
        headers += ["nmi", "ari", "ct_asw"]

    def _fmt(value: float) -> str:
        return "nan" if value != value else f"{value:.4g}"

    rows: List[List[str]] = []
    for report in reports:
        row = [report.name, str(report.n_latent), _fmt(report.separability)]
        if show_clustering:
            row += [_fmt(report.nmi), _fmt(report.ari), _fmt(report.cell_type_asw)]
        rows.append(row)

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    sep = "| " + " | ".join("-" * widths[col] for col in range(len(headers))) + " |"
    header_line = (
        "| "
        + " | ".join(headers[col].ljust(widths[col]) for col in range(len(headers)))
        + " |"
    )
    body = [
        "| "
        + " | ".join(row[col].ljust(widths[col]) for col in range(len(headers)))
        + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep, *body])
