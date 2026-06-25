"""
HuggingFace-Hub serialization for OQAE models.

A trained :class:`~omvqvae.models.vqvae.OmicsVQVAE` is only useful if it can be
reloaded onto the *exact* feature space it was trained on. This module persists
everything needed for that round-trip — the model weights (state dict, including
the learned codebooks), the architecture hyper-parameters
(:meth:`OmicsVQVAE.get_config`), and the gene vocabulary (``organism`` +
ordered ``gene_ids``) — to a plain local directory, and loads it back into a
fully-reconstructed model on the right :class:`~omvqvae.data.dataset.GeneVocabulary`.

The on-disk layout follows HuggingFace conventions so a saved directory *is* a
Hub-ready model repo:

```
<dir>/
├── config.json          # {format_version, organism, gene_ids, model, experiment}
└── pytorch_model.bin     # torch.save(model.state_dict())
```

The ``model`` block of ``config.json`` is exactly ``OmicsVQVAE.get_config()``,
so it shares the architecture-defining fields with the CLI checkpoint bundle
(``{state_dict, organism, gene_ids, config}`` written by ``omvqvae.train.cli``);
:func:`from_checkpoint` bridges a CLI checkpoint into this directory format.

Following the project's "pure core + thin I/O shell" convention, the
save/load round-trip (:func:`save_pretrained` / :func:`load_pretrained`) is pure
local I/O and fully tested offline; the networked Hub transfer
(:func:`push_to_hub` / :func:`from_pretrained`) is a thin shell over
``huggingface_hub`` marked ``# pragma: no cover`` and ``@pytest.mark.network``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from omvqvae.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omvqvae.data.dataset import GeneVocabulary
    from omvqvae.models.vqvae import OmicsVQVAE

logger = get_logger(__name__)

#: Filename of the JSON metadata sidecar (architecture + vocabulary).
CONFIG_NAME = "config.json"
#: Filename of the serialized model weights (``torch.save`` of the state dict).
WEIGHTS_NAME = "pytorch_model.bin"
#: Version stamp written into ``config.json`` so the format can evolve.
FORMAT_VERSION = 1

__all__ = [
    "CONFIG_NAME",
    "WEIGHTS_NAME",
    "FORMAT_VERSION",
    "LoadedModel",
    "save_pretrained",
    "load_pretrained",
    "from_checkpoint",
    "push_to_hub",
    "from_pretrained",
]


@dataclass
class LoadedModel:
    """
    A model reconstructed from a saved directory.

    Attributes
    ----------
    model : OmicsVQVAE
        The reconstructed model with its trained weights loaded (in ``eval``
        mode).
    vocabulary : GeneVocabulary
        The organism-aware reference gene vocabulary the model was trained on;
        its ``n_genes`` matches ``model.n_genes``.
    experiment_config : Dict[str, Any] or None
        The full training ``ExperimentConfig`` captured at train time, if one was
        persisted alongside the model (provenance only — not needed to use the
        model).
    """

    model: "OmicsVQVAE"
    vocabulary: "GeneVocabulary"
    experiment_config: Optional[Dict[str, Any]] = None


def _build_metadata(
    model: "OmicsVQVAE",
    vocabulary: "GeneVocabulary",
    experiment_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the JSON-serializable ``config.json`` payload."""
    if model.n_genes != vocabulary.n_genes:
        raise ValueError(
            f"Model feature size ({model.n_genes}) does not match the gene "
            f"vocabulary size ({vocabulary.n_genes}); they must describe the "
            "same feature space."
        )
    return {
        "format_version": FORMAT_VERSION,
        "organism": vocabulary.organism,
        "gene_ids": vocabulary.gene_ids,
        "model": model.get_config(),
        "experiment_config": experiment_config,
    }


def save_pretrained(
    model: "OmicsVQVAE",
    vocabulary: "GeneVocabulary",
    save_directory: Union[str, Path],
    *,
    experiment_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a trained model + gene vocabulary to a local directory.

    Writes ``config.json`` (architecture hyper-parameters + ``organism`` +
    ordered ``gene_ids`` + optional experiment config) and ``pytorch_model.bin``
    (the model state dict, including the learned codebooks). The directory is a
    Hub-ready model repo and round-trips through :func:`load_pretrained`.

    Parameters
    ----------
    model : OmicsVQVAE
        The trained model to serialize.
    vocabulary : GeneVocabulary
        The reference gene vocabulary the model was trained on. Its ``n_genes``
        must equal ``model.n_genes``.
    save_directory : str or pathlib.Path
        Destination directory (created if missing).
    experiment_config : Dict[str, Any], optional
        Full training configuration to persist for provenance.

    Returns
    -------
    pathlib.Path
        The directory written to.

    Raises
    ------
    ValueError
        If ``model.n_genes`` disagrees with ``vocabulary.n_genes``.
    """
    import torch

    metadata = _build_metadata(model, vocabulary, experiment_config)
    out = Path(save_directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / CONFIG_NAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), out / WEIGHTS_NAME)
    logger.info("Saved OQAE model to %s", out)
    return out


def load_pretrained(
    load_directory: Union[str, Path],
    *,
    map_location: str = "cpu",
) -> LoadedModel:
    """
    Load a model + gene vocabulary previously saved with :func:`save_pretrained`.

    Rebuilds the architecture from ``config.json`` via
    :meth:`OmicsVQVAE.from_config`, restores the weights from
    ``pytorch_model.bin``, and reconstructs the
    :class:`~omvqvae.data.dataset.GeneVocabulary`. The returned model is in
    ``eval`` mode.

    Parameters
    ----------
    load_directory : str or pathlib.Path
        A directory written by :func:`save_pretrained` (or a Hub snapshot).
    map_location : str, default "cpu"
        Device passed to ``torch.load`` for the weights.

    Returns
    -------
    LoadedModel
        The reconstructed model, its gene vocabulary, and any persisted
        experiment config.

    Raises
    ------
    FileNotFoundError
        If ``config.json`` or ``pytorch_model.bin`` is missing.
    ValueError
        If the restored model's feature size disagrees with the saved gene
        vocabulary.
    """
    import torch

    from omvqvae.data.dataset import GeneVocabulary
    from omvqvae.models.vqvae import OmicsVQVAE

    directory = Path(load_directory)
    config_path = directory / CONFIG_NAME
    weights_path = directory / WEIGHTS_NAME
    if not config_path.exists():
        raise FileNotFoundError(f"No {CONFIG_NAME} in {directory}.")
    if not weights_path.exists():
        raise FileNotFoundError(f"No {WEIGHTS_NAME} in {directory}.")

    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    organism: str = metadata["organism"]
    gene_ids: List[str] = list(metadata["gene_ids"])
    vocabulary = GeneVocabulary(organism, gene_ids)

    model = OmicsVQVAE.from_config(metadata["model"])
    if model.n_genes != vocabulary.n_genes:
        raise ValueError(
            f"Saved model feature size ({model.n_genes}) does not match the "
            f"saved gene vocabulary size ({vocabulary.n_genes})."
        )
    # weights_only=True restricts unpickling to tensors/plain containers, so a
    # malicious checkpoint cannot execute arbitrary code on load (CWE-502).
    state_dict = torch.load(weights_path, map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Loaded OQAE model from %s", directory)
    return LoadedModel(
        model=model,
        vocabulary=vocabulary,
        experiment_config=metadata.get("experiment_config"),
    )


def from_checkpoint(
    checkpoint_path: Union[str, Path],
    save_directory: Union[str, Path],
    *,
    map_location: str = "cpu",
) -> Path:
    """
    Convert a CLI training checkpoint into a :func:`save_pretrained` directory.

    The training CLI persists a ``{state_dict, organism, gene_ids, config}``
    bundle (``omvqvae.train.cli``). This rebuilds the model from that bundle's
    ``config["model"]`` block and re-emits it in the Hub-ready directory format,
    so a CLI-trained checkpoint and an HF-pushed model share one on-disk shape.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path
        Path to a ``torch.save``'d CLI checkpoint bundle.
    save_directory : str or pathlib.Path
        Destination directory for the converted model.
    map_location : str, default "cpu"
        Device passed to ``torch.load`` for the checkpoint.

    Returns
    -------
    pathlib.Path
        The directory written to.

    Raises
    ------
    KeyError
        If the checkpoint bundle is missing a required field.
    """
    import torch

    from omvqvae.data.dataset import GeneVocabulary
    from omvqvae.models.vqvae import OmicsVQVAE

    # weights_only=True is safe here: the CLI bundle holds only a tensor state
    # dict plus plain str/list/dict metadata (no custom classes) — and it blocks
    # arbitrary-code execution from an untrusted checkpoint (CWE-502).
    bundle = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    for key in ("state_dict", "organism", "gene_ids", "config"):
        if key not in bundle:
            raise KeyError(f"Checkpoint is missing the required {key!r} field.")

    experiment_config: Dict[str, Any] = dict(bundle["config"])
    gene_ids: List[str] = list(bundle["gene_ids"])
    vocabulary = GeneVocabulary(bundle["organism"], gene_ids)

    model_config = dict(experiment_config["model"])
    model_config["n_genes"] = vocabulary.n_genes
    model = OmicsVQVAE.from_config(model_config)
    model.load_state_dict(bundle["state_dict"])

    return save_pretrained(
        model,
        vocabulary,
        save_directory,
        experiment_config=experiment_config,
    )


# --------------------------------------------------------------------------- #
# Networked Hub transfer (thin I/O shell)
# --------------------------------------------------------------------------- #
def push_to_hub(
    model: "OmicsVQVAE",
    vocabulary: "GeneVocabulary",
    repo_id: str,
    *,
    experiment_config: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload OQAE model",
) -> str:  # pragma: no cover - requires network / HF Hub credentials
    """
    Serialize a model and upload it to a HuggingFace Hub repository.

    The model is written to a temporary directory with :func:`save_pretrained`
    (the fully-tested pure step) and then uploaded; the upload is the only
    networked operation.

    Parameters
    ----------
    model : OmicsVQVAE
        The trained model to push.
    vocabulary : GeneVocabulary
        The model's reference gene vocabulary.
    repo_id : str
        Target repository, e.g. ``"user/oqae-human"``.
    experiment_config : Dict[str, Any], optional
        Full training configuration to persist for provenance.
    token : str, optional
        HuggingFace access token (falls back to the cached login).
    private : bool, default False
        Whether to create the repo as private if it does not exist.
    commit_message : str, default "Upload OQAE model"
        Commit message for the upload.

    Returns
    -------
    str
        The ``repo_id`` that was pushed to.
    """
    import tempfile

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    with tempfile.TemporaryDirectory() as tmp:
        save_pretrained(model, vocabulary, tmp, experiment_config=experiment_config)
        api.upload_folder(
            folder_path=tmp,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
    logger.info("Pushed OQAE model to %s", repo_id)
    return repo_id


def from_pretrained(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    map_location: str = "cpu",
) -> LoadedModel:  # pragma: no cover - requires network / HF Hub
    """
    Download a model from the HuggingFace Hub and load it.

    Snapshots the repo locally (the networked step) and then delegates to
    :func:`load_pretrained` (the fully-tested pure step).

    Parameters
    ----------
    repo_id : str
        Source repository, e.g. ``"user/oqae-human"``.
    revision : str, optional
        Git revision (branch, tag, or commit) to download.
    token : str, optional
        HuggingFace access token (falls back to the cached login).
    cache_dir : str or pathlib.Path, optional
        Directory for the downloaded snapshot.
    map_location : str, default "cpu"
        Device passed to ``torch.load`` for the weights.

    Returns
    -------
    LoadedModel
        The reconstructed model, its gene vocabulary, and any persisted
        experiment config.
    """
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id,
        revision=revision,
        token=token,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        allow_patterns=[CONFIG_NAME, WEIGHTS_NAME],
    )
    return load_pretrained(local_dir, map_location=map_location)
