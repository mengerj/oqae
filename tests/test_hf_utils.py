"""Offline tests for HuggingFace-Hub model serialization.

These exercise the pure save/load round-trip and the CLI-checkpoint bridge
without any network access. The networked Hub transfer (:func:`push_to_hub` /
:func:`from_pretrained`) is a thin shell over ``huggingface_hub`` and is not
exercised here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from omvqvae.data.dataset import GeneVocabulary
from omvqvae.hf_utils import (
    CONFIG_NAME,
    FORMAT_VERSION,
    WEIGHTS_NAME,
    LoadedModel,
    from_checkpoint,
    load_pretrained,
    save_pretrained,
)
from omvqvae.models.vqvae import OmicsVQVAE

GENE_IDS = ["ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5", "ENSG6"]


def _make_model(*, seed: int = 0) -> OmicsVQVAE:
    """A small, deterministically-initialized model on ``GENE_IDS``."""
    torch.manual_seed(seed)
    return OmicsVQVAE(
        len(GENE_IDS),
        n_latent=4,
        hidden_dims=(8,),
        likelihood="nb",
        codebook_size=16,
        n_codebooks=2,
    )


def _counts(n_cells: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """A tiny synthetic raw-count batch and its size factors."""
    torch.manual_seed(123)
    counts = torch.randint(0, 20, (n_cells, len(GENE_IDS))).float()
    size_factors = counts.sum(dim=1).clamp(min=1.0)
    return counts, size_factors


def _assert_same_weights(a: OmicsVQVAE, b: OmicsVQVAE) -> None:
    """Assert two models have identical state dicts."""
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for key in sa:
        torch.testing.assert_close(sa[key], sb[key])


# --------------------------------------------------------------------------- #
# save_pretrained / load_pretrained
# --------------------------------------------------------------------------- #
def test_save_pretrained_writes_expected_files(tmp_path: Path) -> None:
    model = _make_model()
    vocab = GeneVocabulary("homo_sapiens", GENE_IDS)

    out = save_pretrained(model, vocab, tmp_path / "model")

    assert out == tmp_path / "model"
    assert (out / CONFIG_NAME).exists()
    assert (out / WEIGHTS_NAME).exists()

    metadata = json.loads((out / CONFIG_NAME).read_text())
    assert metadata["format_version"] == FORMAT_VERSION
    assert metadata["organism"] == "homo_sapiens"
    assert metadata["gene_ids"] == GENE_IDS
    assert metadata["model"] == model.get_config()
    assert metadata["experiment_config"] is None


def test_round_trip_reconstructs_identical_model(tmp_path: Path) -> None:
    model = _make_model(seed=1)
    model.eval()
    vocab = GeneVocabulary("mus_musculus", GENE_IDS)

    save_pretrained(model, vocab, tmp_path)
    loaded = load_pretrained(tmp_path)

    assert isinstance(loaded, LoadedModel)
    assert loaded.vocabulary.organism == "mus_musculus"
    assert loaded.vocabulary.gene_ids == GENE_IDS
    assert loaded.experiment_config is None
    assert loaded.model.get_config() == model.get_config()
    _assert_same_weights(model, loaded.model)

    # Same input → identical discrete codes and reconstruction (eval mode).
    counts, size_factors = _counts()
    torch.testing.assert_close(
        model.encode_codes(counts), loaded.model.encode_codes(counts)
    )
    torch.testing.assert_close(
        model.expected_counts(
            model.quantize(model.encode(counts)).quantized, size_factors
        ),
        loaded.model.expected_counts(
            loaded.model.quantize(loaded.model.encode(counts)).quantized,
            size_factors,
        ),
    )


def test_round_trip_after_training_preserves_codebooks(tmp_path: Path) -> None:
    """A trained model (non-trivial codebooks) round-trips bit-for-bit."""
    model = _make_model(seed=2)
    counts, size_factors = _counts(n_cells=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(5):
        optimizer.zero_grad()
        out = model(counts, size_factors)
        out.loss.backward()
        optimizer.step()
    model.eval()

    vocab = GeneVocabulary("homo_sapiens", GENE_IDS)
    save_pretrained(model, vocab, tmp_path)
    loaded = load_pretrained(tmp_path).model

    _assert_same_weights(model, loaded)
    torch.testing.assert_close(model.encode_codes(counts), loaded.encode_codes(counts))


def test_experiment_config_round_trips(tmp_path: Path) -> None:
    model = _make_model()
    vocab = GeneVocabulary("homo_sapiens", GENE_IDS)
    experiment_config: Dict[str, Any] = {
        "seed": 0,
        "model": model.get_config(),
        "training": {"lr": 1e-3, "max_epochs": 2},
    }

    save_pretrained(model, vocab, tmp_path, experiment_config=experiment_config)
    loaded = load_pretrained(tmp_path)

    assert loaded.experiment_config == experiment_config


def test_loaded_model_is_in_eval_mode(tmp_path: Path) -> None:
    model = _make_model()
    vocab = GeneVocabulary("homo_sapiens", GENE_IDS)
    save_pretrained(model, vocab, tmp_path)

    assert load_pretrained(tmp_path).model.training is False


# --------------------------------------------------------------------------- #
# Validation / error paths
# --------------------------------------------------------------------------- #
def test_save_rejects_vocab_size_mismatch(tmp_path: Path) -> None:
    model = _make_model()
    vocab = GeneVocabulary("homo_sapiens", GENE_IDS[:-1])  # one gene short
    with pytest.raises(ValueError, match="feature space"):
        save_pretrained(model, vocab, tmp_path)


def test_load_missing_config_raises(tmp_path: Path) -> None:
    (tmp_path / WEIGHTS_NAME).write_bytes(b"")
    with pytest.raises(FileNotFoundError, match=CONFIG_NAME):
        load_pretrained(tmp_path)


def test_load_missing_weights_raises(tmp_path: Path) -> None:
    (tmp_path / CONFIG_NAME).write_text("{}")
    with pytest.raises(FileNotFoundError, match=WEIGHTS_NAME):
        load_pretrained(tmp_path)


def test_load_rejects_inconsistent_metadata(tmp_path: Path) -> None:
    """Hand-crafted metadata whose model size disagrees with the vocabulary."""
    model = _make_model()
    vocab = GeneVocabulary("homo_sapiens", GENE_IDS)
    save_pretrained(model, vocab, tmp_path)

    metadata = json.loads((tmp_path / CONFIG_NAME).read_text())
    metadata["gene_ids"] = GENE_IDS + ["ENSG7"]  # 7 genes vs the model's 6
    (tmp_path / CONFIG_NAME).write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="does not match"):
        load_pretrained(tmp_path)


# --------------------------------------------------------------------------- #
# from_checkpoint (CLI bundle bridge)
# --------------------------------------------------------------------------- #
def _write_cli_checkpoint(path: Path, model: OmicsVQVAE) -> Dict[str, Any]:
    """Write a CLI-style ``{state_dict, organism, gene_ids, config}`` bundle."""
    config: Dict[str, Any] = {
        "seed": 0,
        "model": model.get_config(),
        "training": {"lr": 1e-3},
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "organism": "homo_sapiens",
            "gene_ids": GENE_IDS,
            "config": config,
        },
        path,
    )
    return config


def test_from_checkpoint_bridges_cli_bundle(tmp_path: Path) -> None:
    model = _make_model(seed=3)
    model.eval()
    ckpt = tmp_path / "ckpt.pt"
    config = _write_cli_checkpoint(ckpt, model)

    out = from_checkpoint(ckpt, tmp_path / "hf")
    loaded = load_pretrained(out)

    assert loaded.vocabulary.gene_ids == GENE_IDS
    assert loaded.experiment_config == config
    _assert_same_weights(model, loaded.model)
    counts, _ = _counts()
    torch.testing.assert_close(
        model.encode_codes(counts), loaded.model.encode_codes(counts)
    )


def test_from_checkpoint_requires_all_fields(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"state_dict": {}, "organism": "homo_sapiens"}, ckpt)
    with pytest.raises(KeyError):
        from_checkpoint(ckpt, tmp_path / "hf")
