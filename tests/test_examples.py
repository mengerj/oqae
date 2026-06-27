"""
Smoke tests for the runnable examples under ``examples/``.

The offline examples (1 and 2) are executed end to end on tiny synthetic data so
the documented workflows are guaranteed to keep working; the Census example (3)
needs network access, so it is only *imported* here (to catch syntax / import
regressions) and its ``main`` is exercised under the ``network`` marker.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _load_example(filename: str) -> ModuleType:
    """Import an example script by file path (handles digit-prefixed names)."""
    # Examples import their sibling ``synthetic_data`` helper by bare name, which
    # works when run directly (script dir on ``sys.path``); replicate that here.
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    module_name = f"oqae_example_{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLES_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_data_has_structure() -> None:
    """The shared helper builds a tiny raw-count AnnData with program labels."""
    module = _load_example("synthetic_data.py")
    adata = module.make_synthetic_anndata(n_cells=32, n_genes=10, n_programs=3)
    assert adata.shape == (32, 10)
    assert (adata.X >= 0).all()
    assert "program" in adata.obs
    assert len(module.synthetic_gene_ids(5, "mus_musculus")) == 5


def test_example_01_trains_and_saves(tmp_path: Path) -> None:
    """Example 1 trains a model and writes a loadable Hub-ready directory."""
    from omvqvae.hf_utils import load_pretrained

    module = _load_example("01_train_local_anndata.py")
    out = module.main(save_directory=tmp_path / "model")

    assert (out / "config.json").exists()
    assert (out / "pytorch_model.bin").exists()
    loaded = load_pretrained(out)
    assert loaded.model.n_genes == loaded.vocabulary.n_genes


def test_example_02_inspects_and_generates(tmp_path: Path) -> None:
    """Example 2 runs the encode / inspect / decode / generate workflow."""
    module = _load_example("02_inspect_and_generate_codes.py")
    # Runs end to end without raising; exercises the full inference API.
    module.main(workdir=tmp_path)


def test_example_03_imports() -> None:
    """The Census example imports cleanly (its ``main`` is network-gated)."""
    module = _load_example("03_census_streaming.py")
    assert callable(module.main)


@pytest.mark.network
def test_example_03_streams() -> None:  # pragma: no cover - requires live Census
    """Run the Census streaming example end to end (skipped by default)."""
    module = _load_example("03_census_streaming.py")
    module.main()
