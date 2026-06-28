# 🧬 OQAE Project Plan: Omics Quantized Auto Encoder

## 🎯 **Project Overview**

**OQAE** (Omics Quantized Auto Encoder) is a lightweight, production-ready
Python library that learns a **discrete, universal latent space for omics
data** using a residual-quantized VQ-VAE.

The central idea: represent any single-cell RNA-seq (scRNA-seq) sample as a
**set of discrete code vectors** (indices into learned codebooks). Those codes
are a compact, universal "vocabulary" of expression patterns. Because the
codes are discrete and the decoder is generative, the same codes can be fed
back into the trained decoder to **reconstruct or generate** expression
profiles — enabling representation learning, compression, integration, and
in-silico data generation from a shared latent vocabulary.

We **start with scRNA-seq** because the CZ CELLxGENE Census hosts a very large,
standardized corpus of single-cell RNA-seq, which lets us train at scale by
streaming. The architecture is designed so the same discrete-codebook approach
can later extend to other omics modalities.

### **Core Value Proposition**
- **Discrete universal latent space**: Every cell → a set of discrete codes
  drawn from shared codebooks; codes are composable and decoder-pluggable.
- **Train at scale by streaming**: Stream millions of cells directly from the
  CZ CELLxGENE Census via TileDB-SOMA — no need to download or hold the full
  corpus in memory.
- **Bring-your-own-data**: Train or fine-tune on a local `.h5ad` or `.zarr`
  AnnData file with the same interface.
- **Raw counts in, counts out**: Model raw counts directly with a count
  likelihood (Negative Binomial / Zero-Inflated NB) — no mandatory external
  normalization pipeline.
- **Generative decoder**: Discrete codes → decoder → expression, for
  reconstruction and synthetic data generation.
- **Production Ready**: strict typing (mypy), high test coverage, W&B
  experiment tracking, comprehensive CI/CD.

## 🏗️ **Architecture Decisions**

### **Data Sources & Loading Strategy**

OQAE consumes data through a single unified loader interface backed by two
sources:

1. **CZ CELLxGENE Census (primary, streaming)** — the recommended path for
   large-scale training. We use the maintained TileDB-SOMA stack:
   - `cellxgene_census` to open a pinned Census version
     (e.g. `census_version="2025-01-30"`).
   - `tiledbsoma` `ExperimentAxisQuery` to define the slice of cells via
     `obs_query` value filters (e.g. tissue, assay, disease).
   - `tiledbsoma_ml.ExperimentDataset` — a PyTorch `IterableDataset` that
     streams batches without materializing the full result in memory — wrapped
     with `experiment_dataloader()` for correct/performant multi-worker
     iteration. Raw counts are read via the `"raw"` layer.

   > Note: this supersedes the deprecated
   > `cellxgene_census.experimental.ml.ExperimentDataPipe` API.

2. **Local AnnData (`.h5ad` / `.zarr`)** — for training/fine-tuning on private
   or curated datasets. Larger-than-memory local files are read in chunks
   (zarr-backed / backed-mode AnnData) so the same memory discipline applies.

Both sources are normalized to a common minibatch contract:
`(counts, obs_covariates)` where `counts` is a dense/sparse cell × gene tensor
of **raw counts** and `obs_covariates` carries optional metadata fields
(e.g. batch/dataset id, organism). A per-organism **gene-space mapping** aligns
each source to the model's expected feature ordering (see below).

#### **Multi-organism support (human + mouse)**

The Census hosts multiple organisms; **v1 supports both human and mouse**. Each
organism has its own gene/feature universe, so OQAE is **organism-aware**:

- `organism` is an explicit parameter of the data layer (and is recorded in
  model metadata). Census slices are queried per-organism (the Census exposes
  `homo_sapiens` and `mus_musculus` experiments separately).
- Each organism has its own **reference gene set** (the Census `var` index for
  that organism, optionally restricted to a configurable panel). Local AnnData
  is aligned/subset to the reference for its organism — genes present in the
  reference but missing locally are zero-filled; extra local genes are dropped —
  with a warning when overlap is low.
- A trained model carries a fixed organism + gene vocabulary; **v1 trains one
  model per organism** (separate codebooks/feature spaces). Cross-organism
  unification into a single shared latent space is an explicit **open design
  question deferred past v1** (would require ortholog mapping or a shared gene
  embedding); it does not block the v1 data layer or model.

### **Data Flow Architecture**
```
CZ CELLxGENE Census (TileDB-SOMA) ─┐
                                   ├─► Streaming DataLoader ─► raw counts ─┐
Local AnnData (.h5ad / .zarr) ─────┘   (batched, shuffled)                │
                                                                          ▼
                    Encoder ─► Residual Vector Quantizer ─► Decoder ─► NB/ZINB
                       │         (discrete codes / codebooks)    │      params
                       │                                         │
                  internal log1p                          reconstructs raw
                  (numerical stability)                   counts (library-size
                                                          aware)
                                          │
                                   W&B experiment tracking
                                   (losses, codebook usage/perplexity)
```

### **Input Representation & Normalization**

A deliberate change from the original plan: **the model ingests raw counts and
performs normalization internally**, rather than requiring a fixed external
log1p→CPM→standardize pipeline.

This follows the design of modern single-cell VAEs (e.g. the scVI family),
which feed unnormalized counts and use a **Negative Binomial (NB)** — or
**Zero-Inflated NB (ZINB)** — reconstruction likelihood. Library-size / depth
variation is handled inside the model (via an observed size factor or a learned
library term) rather than by pre-normalizing the data. This keeps the count
statistics intact and avoids baking normalization choices into the dataset.

Concretely:
- **Decoder likelihood**: NB by default; ZINB and Gaussian-on-log1p available
  as pluggable alternatives so we can benchmark them.
- **Encoder input**: raw counts with an **internal log1p (and optional
  per-cell normalization)** applied for numerical stability before encoding —
  this is an internal transform, not a user-facing preprocessing step.
- **Size factors**: derived from observed total counts per cell, fed to the
  decoder so it reconstructs depth-appropriate counts.
- **Pluggable**: likelihood and internal-normalization are configurable, so a
  user can opt into a log-normalized/Gaussian setup if desired.

> **Open design question (to validate empirically):** raw-count + NB vs.
> log-normalized + Gaussian for the *quantized* setting. The plan is to support
> both and benchmark reconstruction, codebook utilization, and downstream
> separability (see PR #9).

### **Model Architecture**
- **Encoder → Residual Vector Quantizer → Decoder.**
- **Residual quantization**: configurable number of codebook levels (default
  2); each cell is encoded as a set/sequence of codebook indices — this *is*
  the discrete universal representation.
- **Straight-through estimator** for gradient flow through the discrete
  bottleneck; **codebook/commitment losses** and **perplexity** monitoring to
  track codebook utilization and guard against collapse.
- **Generative decoder**: maps quantized codes (+ size factor) to NB/ZINB
  parameters over genes.
- **Conditioning**: **v1 is unconditional** (no batch/dataset covariate fed to
  the model). Optional categorical conditioning to encourage cross-study
  integration is deferred; we will first **benchmark whether batch effects are
  actually a problem** in the discrete latent space (PR #9) and only add
  conditioning if needed. The data layer still *carries* covariates in the
  minibatch so this can be enabled later without a data-format change.
- **CPU/GPU**: inference on CPU, training optimized for GPU.

### **Experiment Tracking & Monitoring**
- **Weights & Biases (W&B)** is the primary monitoring tool: log training/val
  losses, reconstruction metrics, codebook usage/perplexity, learning-rate, and
  resource stats, with run config captured for reproducibility.
- W&B is **optional/offline-friendly** (a no-op/console logger is used when W&B
  is disabled), so the library remains usable without an account.
- The previous bespoke `system_monitor` utility has been removed in favor of
  this integration.

## 📊 **Technical Requirements**

### **Performance & Scale**
- **Streaming-first**: train over Census-scale corpora (tens of millions of
  cells) without downloading them, bounded host memory via batched streaming.
- **Local large files**: chunked/backed reads for `.h5ad` / `.zarr` larger than
  RAM.
- **Speed**: efficient GPU training; smoke-test toy training in minutes.

### **Hardware Compatibility**
- **CPU**: full inference and small-scale training.
- **GPU**: optimized training for large datasets.

### **Integration Requirements**
- **CZ CELLxGENE Census**: streaming training data via TileDB-SOMA.
- **AnnData ecosystem**: `.h5ad` / `.zarr` interoperability.
- **HuggingFace Hub**: model + codebook sharing and versioning.
- **Weights & Biases**: experiment tracking.

## 🗂️ **Package Structure**

```
src/omvqvae/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── census.py          # CELLxGENE Census streaming loaders (TileDB-SOMA)
│   ├── anndata_io.py      # Local .h5ad / .zarr AnnData loaders (chunked)
│   ├── dataset.py         # Unified minibatch contract + gene-space alignment
│   └── normalize.py       # Internal normalization / size-factor helpers
├── layers/
│   ├── __init__.py
│   └── residual_vq.py     # Residual vector-quantization layers
├── models/
│   ├── __init__.py
│   ├── vqvae.py           # Encoder/decoder VQ-VAE
│   └── likelihoods.py     # NB / ZINB / Gaussian reconstruction heads
├── train/
│   ├── __init__.py
│   ├── cli.py             # Training/fine-tuning CLI (typer + OmegaConf)
│   └── loop.py            # Training loop + W&B logging
├── inference/
│   ├── __init__.py
│   └── codes.py           # encode → discrete codes; decode codes → expression
├── utils/
│   ├── __init__.py
│   ├── logging.py         # Centralized logging
│   └── tracking.py        # W&B / offline experiment-tracking wrapper
└── hf_utils.py            # HuggingFace Hub integration
```

## 🛣️ **Development Roadmap**

### **Phase 1: Foundation & Data (PRs 1–3)**

#### **PR #1: Logging & Tooling — ✅ DONE**
- Core logging (`utils/logging.py`), uv-based dev workflow, CI/CD
  (black/isort/flake8/**mypy**/pytest/bandit), pre-commit, Makefile.
- Exit criteria: green CI, package imports a logger, strict mypy passes.

#### **PR #2: Data Layer — Census streaming + local AnnData — ✅ DONE**
- **Status**: complete. *Slice 1* (local AnnData + organism-aware gene alignment +
  normalization + shared `Minibatch` contract) merged; *Slice 2* (Census
  streaming via TileDB-SOMA with a network-gated live test) landed in
  `data/census.py`. Both human and mouse stream through the same loader API.
- **Scope**: unified data interface over two sources, **organism-aware
  (human + mouse)**.
- **Files**: `data/census.py`, `data/anndata_io.py`, `data/dataset.py`,
  `data/normalize.py`.
- **Key features**:
  - CELLxGENE Census streaming via `cellxgene_census` + `tiledbsoma` +
    `tiledbsoma_ml` (pinned Census version, `obs_query` filtering, raw layer),
    with `organism` selecting the `homo_sapiens` / `mus_musculus` experiment.
  - Local `.h5ad` / `.zarr` loaders with chunked/backed reads.
  - **Per-organism reference gene set** + alignment helper that maps any source
    (Census or local) onto that organism's feature ordering (zero-fill missing,
    drop extra, warn on low overlap).
  - Common `(raw_counts, covariates)` minibatch contract (covariates carry
    organism + batch/dataset id even though v1 is unconditional).
  - Size-factor computation; internal-normalization helpers.
- **Dependencies to add** (with refreshed `uv.lock`): `cellxgene-census`,
  `tiledbsoma`, `tiledbsoma-ml`.
- **Exit criteria**: stream a small Census slice (human *and* mouse) and iterate
  a local AnnData through the *same* DataLoader API; tests with a tiny offline
  fixture (live-Census tests marked/skippable); strict mypy + bounded memory.

#### **PR #3: Residual Vector Quantizer Layer — ✅ DONE**
- **Scope**: configurable residual VQ with straight-through estimator.
- **Files**: `layers/residual_vq.py`.
- **Key features**: `VectorQuantizer` (single codebook) + `ResidualVQ` (stacked
  over residuals, `n_codebooks` default 2), straight-through estimator,
  commitment + codebook losses, optional EMA codebook updates (Laplace-smoothed)
  and dead-code reset, per-forward perplexity/utilization metrics returned via
  `QuantizerOutput` / `ResidualVQOutput`.
- **Exit criteria**: 100% offline test coverage on the module, straight-through
  gradient-flow verification, strict mypy — all met; `make ci` green.

### **Phase 2: Core Model (PRs 4–6)**

#### **PR #4: VQ-VAE Core Model (raw-count, NB/ZINB) — ✅ DONE**
- **Scope**: encoder/decoder, count likelihoods, size-factor conditioning, loss
  composition (recon + VQ).
- **Files**: `models/vqvae.py`, `models/likelihoods.py`.
- **Status**: complete. *Slice 1* (reconstruction likelihoods / decoder heads)
  merged; *Slice 2* (`OmicsVQVAE` wiring the encoder, `ResidualVQ`, and a
  `ReconstructionHead` end to end, returning a `VQVAEOutput`) landed in
  `models/vqvae.py`. v1 is unconditional, so covariates are carried by the data
  layer but not fed to the model.
- **Exit criteria** (met): 40-step smoke train on synthetic counts; NB and ZINB
  heads both train (recon loss decreases); codebooks are utilized (non-trivial
  perplexity); 100% offline coverage; `make ci` green.

#### **PR #5: Training/Fine-tuning CLI + W&B**
- **Scope**: OmegaConf config, typer CLI, training loop, W&B tracking, toy
  fine-tuning from a checkpoint.
- **Files**: `train/cli.py`, `train/loop.py`, `utils/tracking.py`, config
  schemas.
- **Exit criteria**: CLI trains a toy model from Census *and* from a local
  `.h5ad` in minutes; runs log to W&B (and work offline).
- **Status**: **DONE**. Slice 1 — `utils/tracking.py` (offline-friendly
  `ExperimentTracker`/`ConsoleTracker`/`WandbTracker`/`build_tracker` +
  `vqvae_metrics`) and `train/loop.py` (`train` + `TrainConfig`/`EpochMetrics`/
  `TrainResult`, source-agnostic over any `Minibatch` iterable). Slice 2 —
  `train/cli.py`: an OmegaConf-validated config schema (`ExperimentConfig` and
  per-section dataclasses) + a typer `oqae-train` entry point. Pure builders map
  each config section to an object (`build_model`, `build_train_config`,
  `build_tracker_from_config`, `build_data`); `run_experiment` wires them and
  calls `train`. Local-AnnData source derives its `GeneVocabulary` from the
  file's genes; the Census source is gated/`# pragma: no cover`. Optional
  checkpoint persists `{state_dict, organism, gene_ids, config}`. Ships
  `configs/train_toy.yaml`; offline tests cover loading/builders/wiring and the
  Typer `CliRunner` end-to-end.

#### **PR #6: HuggingFace Hub Integration — ✅ DONE**
- **Scope**: serialize/deserialize model + codebooks + config; push/pull.
- **Files**: `hf_utils.py`.
- **Status**: complete. `save_pretrained` / `load_pretrained` round-trip a
  trained `OmicsVQVAE` (state dict + codebooks), its architecture config, and
  the gene vocabulary through a HuggingFace-style directory (`config.json` +
  `pytorch_model.bin`); the model is self-describing via
  `OmicsVQVAE.get_config()` / `from_config()`. `from_checkpoint` bridges a CLI
  checkpoint bundle; `push_to_hub` / `from_pretrained` are thin `huggingface_hub`
  shells (`# pragma: no cover`, network-gated) over the tested pure step. Added
  direct dep `huggingface-hub` and refreshed `uv.lock`.
- **Exit criteria** (met): offline round-trip of state+codebooks+config+vocab
  through a local dir rebuilds the exact model/feature space; 100% offline
  coverage on `hf_utils.py`; `make ci` green. The live HF-repo round-trip is the
  network-gated shell.

### **Phase 3: Latent API, Examples & Release (PRs 7–10)**

#### **PR #7: Discrete-Code Inference API — ✅ DONE**
- **Scope**: `encode(adata) → discrete codes` and
  `decode(codes) → expression`; the universal-latent use cases (compression,
  generation, plugging codes into the decoder).
- **Files**: `inference/codes.py` (+ `inference/__init__.py`).
- **Status**: complete. `encode` / `encode_anndata` produce an `EncodedCells`
  bundle (`codes` `(n_cells, n_codebooks)` int64 + per-cell `size_factors` +
  continuous `latent`); `encode_anndata` aligns a local AnnData to the model's
  `GeneVocabulary` first. `decode` maps codes → expected counts and
  `decode_to_params` exposes the full head distribution. The inverse path is a
  tested model/layer method (`VectorQuantizer.lookup` / `ResidualVQ.lookup`,
  `OmicsVQVAE.decode_codes` / `codes_to_params`), not internal poking. Inference
  runs in `eval` + `no_grad` (EMA codebooks untouched) and restores the model's
  prior mode; `encode`/`decode` are batched. No new deps.
- **Exit criteria** (met): round-trip encode→decode on held-out synthetic cells;
  documented code-vector format (see `inference/codes.py` module docstring); 100%
  offline coverage on the module; `make ci` green.

#### **PR #8: Examples & Documentation — ✅ DONE**
- **Scope**: examples (Census streaming, local fine-tuning, code
  inspection/generation), Sphinx docs.
- **Status**: complete. *Slice 1* (example scripts) — `examples/` holds three
  runnable scripts (`01_train_local_anndata.py`,
  `02_inspect_and_generate_codes.py`, `03_census_streaming.py`), a `README.md`,
  and a shared offline `synthetic_data.py` helper; offline examples are
  smoke-tested in `tests/test_examples.py`. *Slice 2* (Sphinx docs) — a Sphinx
  project under `docs/source/` (`autodoc` + `napoleon` over the NumPy
  docstrings, `furo` theme): `index.rst`, a narrative `getting_started.rst`
  linking the examples, and `api.rst` autodoccing the public API by module. A
  `make docs` target (`sphinx-build -W`) and a `docs` CI job keep the build
  warning-clean; added a `docs` extra (`sphinx`, `furo`) + refreshed `uv.lock`.
- **Exit criteria** (met): docs build clean (warnings-as-errors); examples run
  on small data (offline, CI-tested).

#### **PR #9: Benchmarking & Scaling**
- **Scope**: raw-count+NB vs log-normalized+Gaussian comparison; codebook
  config sweeps; streaming throughput/scaling.
- **Exit criteria**: benchmark report (reconstruction, codebook utilization,
  downstream separability, throughput).

#### **PR #10: v1.0 Release**
- **Scope**: API freeze, final docs, PyPI release.
- **Exit criteria**: PyPI 1.0.0, complete documentation.

## 🧪 **Development Workflow**

### **Testing Strategy**
- **Unit tests**: pytest with a high coverage requirement; tiny synthetic/
  fixture datasets so tests stay fast and offline.
- **Integration tests**: end-to-end loader → model → loss on small fixtures.
- **Network-gated tests**: live Census streaming tests marked and skippable
  (CI runs offline by default).

### **Code Quality Standards**
- **Type hints**: strict mypy (enabled in CI).
- **Docstrings**: NumPy style for public APIs.
- **Formatting**: black + isort + flake8.
- **Security**: bandit + safety scanning.

## 📋 **Dependencies**

### **Core (planned, added as the implementing PRs land)**
```toml
dependencies = [
    "torch>=2.0.0",            # Deep learning framework
    "anndata>=0.8.0",          # Omics data structure
    "numpy>=1.21.0",
    "pandas>=1.5.0",
    "scipy>=1.9.0",
    "zarr>=2.12.0",            # Chunked array storage (local .zarr)
    # --- CELLxGENE Census streaming (PR #2) ---
    "cellxgene-census>=1.15.0",
    "tiledbsoma>=1.12.0",
    "tiledbsoma-ml>=0.1.0",
    # --- Modeling / training / tracking ---
    "transformers>=4.20.0",    # HuggingFace integration
    "huggingface-hub>=0.20.0",
    "omegaconf>=2.3.0",        # Configuration management
    "typer>=0.9.0",            # CLI framework
    "rich>=13.0.0",            # Rich terminal output
    "wandb>=0.16.0",           # Experiment tracking
]
```
> The current `pyproject.toml` still lists the original/foundation
> dependencies; Census/W&B-specific entries are added (with a refreshed
> `uv.lock`) in the PRs that first use them, to avoid premature lock churn.

### **Development**
- **Testing**: pytest, pytest-cov
- **Quality**: black, isort, flake8, mypy, bandit, safety
- **Docs**: sphinx, jupyter
- **CI/CD**: pre-commit, GitHub Actions

## 🎯 **Success Metrics**
- **Scale**: train over Census-scale corpora by streaming, bounded host memory.
- **Coverage**: high test coverage on implemented modules.
- **Type Safety**: strict mypy compliance (CI-enforced).
- **Latent quality**: well-utilized codebooks (healthy perplexity, no
  collapse); faithful encode→decode reconstruction; meaningful structure in the
  discrete space.

## 🧭 **Design Decisions Log**

A running record of decisions so future sessions don't re-litigate them.

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-06-17 | **Goal**: discrete, universal latent space for omics — cells → sets of discrete codes that plug into a generative decoder. Start with scRNA-seq. | Codes are composable/decoder-pluggable; enables compression, integration, and generation from a shared vocabulary. |
| 2026-06-17 | **Primary data = stream from CZ CELLxGENE Census** (TileDB-SOMA); also support local `.h5ad`/`.zarr`. Dropped Zarr/Dask-primary framing. | Census hosts a huge standardized scRNA-seq corpus; streaming avoids downloading it. |
| 2026-06-17 | **Ingest raw counts; reconstruct with NB/ZINB**, library size handled internally. Log-normalized/Gaussian kept as pluggable alternative. | Matches scVI-family practice; preserves count statistics; avoids baking in normalization. |
| 2026-06-17 | **Monitoring = Weights & Biases** (offline-friendly); removed bespoke `system_monitor`. | Standard tool; less code to maintain. |
| 2026-06-17 | **Multi-organism: support human + mouse**, organism-aware with a per-organism gene space; **one model per organism in v1**. Cross-organism unification deferred. | Each organism has a distinct gene universe; ortholog/shared-embedding mapping is a separate research question. |
| 2026-06-17 | **v1 model is unconditional** (no batch/covariate input). Revisit only if benchmarking shows batch effects hurt the latent space. | Keep v1 simple; data layer still carries covariates so conditioning can be added later without a format change. |
| 2026-06-17 | **Next implementation = PR #2 (data layer).** | Unblocks the quantizer and model PRs. |
| 2026-06-18 | **Split PR #2 into two slices**: (1) local AnnData + organism-aware gene alignment + normalization + shared `Minibatch` contract (offline-testable, no heavy deps); (2) Census streaming (TileDB-SOMA deps + network-gated test). Slice 1 landed first. | Keeps each PR focused and fully CI-verifiable offline; the Census path needs heavy deps and a live network it can't exercise in CI, so it slots into the contract slice 1 establishes. |
| 2026-06-18 | ~~**Pin `zarr>=2.12.0,<3`.**~~ Superseded same day (see next row). | `anndata` (0.11.x) did not support zarr-python v3; an unconstrained `zarr>=2.12.0` resolved to 3.x and broke `import anndata`. |
| 2026-06-18 | **Require `zarr>=3.0.0` + `anndata>=0.12.0`** (reverses the `zarr<3` pin above). | `anndata` 0.12 adds zarr-python v3 support; standardize on zarr v3 rather than holding back on the v2 line. |
| 2026-06-18 | **Minibatch contract = `Minibatch(counts, size_factors, covariates)`** as a dataclass; per-cell samples are dicts (`counts`, `size_factor`, `organism`, `batch`) stacked by `collate_minibatch`. Covariates always carry `organism` + `batch`. | One contract shared by every source (local now, Census next); v1 ignores covariates but they travel with the batch so conditioning can be added without a format change. |
| 2026-06-19 | **Census streaming uses `tiledbsoma_ml.ExperimentDataset` + `experiment_dataloader`** wrapped by `CensusMinibatchLoader`, which adapts each streamed `(X, obs)` chunk to the shared `Minibatch` via `census_chunk_to_minibatch`. The chunk→`Minibatch` glue is pure-Python (tested offline with synthetic fixtures); the live TileDB-SOMA wiring is a single `network`-marked test (skipped by default). | Keeps the heavy/networked path thin and the contract glue 100%-covered offline; reuses `GeneVocabulary`/`align_to_reference` so Census and local AnnData share one downstream API. |
| 2026-06-19 | **Pin `DEFAULT_CENSUS_VERSION = "2025-11-08"`** (newest LTS at implementation time), configurable per call. Default reference gene set = the full Census `var` index for the organism. | Reproducible streaming; supersedes the earlier `2025-01-30` note. A curated/HVG gene panel can be selected later via `var_value_filter` in the model PRs. |
| 2026-06-19 | **Register a `network` pytest marker, skipped by default** via `addopts = [..., "-m", "not network"]`; run live tests with `pytest -m network -o addopts=""`. | Keeps CI offline-by-default while still shipping an executable end-to-end Census check. |
| 2026-06-20 | **Residual VQ = `VectorQuantizer` (one codebook) composed by `ResidualVQ` (stacks N over residuals, default 2).** EMA codebook updates are the default (`ema=True`) with dead-code reset; non-EMA falls back to a gradient-trained codebook + codebook-pull loss. Forward returns a `QuantizerOutput`/`ResidualVQOutput` dataclass bundling quantized vectors, indices, split losses, and perplexity/utilization metrics. | EMA + dead-code reset is the standard collapse-resistant VQ recipe (VQ-VAE-2); the dataclass result keeps the model/training PRs decoupled from the quantizer internals and gives PR #5/#9 their monitoring signals for free. |
| 2026-06-21 | **Reconstruction heads use the scVI count parameterization** (`models/likelihoods.py`): a softmax over genes gives mean *proportions* (`px_scale`), scaled by the observed library size (size factor) to the NB mean `px_rate`; dispersion is a learned **gene-wise** `theta`. NB/ZINB/Gaussian share one `ReconstructionHead` interface (`forward`/`reconstruction_loss`/`expected_counts`) via a `build_reconstruction_head` factory. | Keeps depth handling inside the model and count statistics intact; one interface lets the model and W&B PRs swap likelihoods without changing call sites. Gene-wise dispersion matches scVI's default and is enough for v1. |
| 2026-06-21 | **Split PR #4 into slices; do the likelihood heads first, independently of the residual VQ.** PR #3 (residual VQ) was concurrently in flight as PR #24, so this run built `models/likelihoods.py` (no VQ dependency) rather than duplicating or blocking on it; once PR #24 merged, `main` was merged back in. The VQ-VAE core (`models/vqvae.py`) is slice 2 and composes `ResidualVQ` + a `ReconstructionHead`. | Avoids duplicating in-flight work and a docs merge conflict; keeps each run to one coherent, CI-verifiable chunk. |
| 2026-06-22 | **`OmicsVQVAE` = symmetric MLP encoder/decoder around `ResidualVQ` + a `ReconstructionHead`.** Encoder input is internally `log1p`-transformed for numerical stability; the reconstruction *target* is raw counts for NB/ZINB and `log1p` expression for the Gaussian head. Decoder hidden widths mirror the encoder (`hidden_dims` reversed); the head consumes the final decoder hidden dim. `forward` returns a `VQVAEOutput` composing recon + VQ loss with the per-level codes and codebook metrics. Total loss = `reconstruction_loss + vq.loss` (per-cell-mean recon NLL + mean VQ loss), unweighted in v1. | Keeps depth/normalization handling inside the model and count statistics intact; one symmetric, configurable module covers all three likelihoods. The `VQVAEOutput` bundle hands PR #5 its loss + W&B monitoring signals without coupling to the layer internals. Loss-term weighting is left as a future tuning knob. |
| 2026-06-25 | **HF serialization = self-describing model + a HuggingFace-style directory.** Added `OmicsVQVAE.get_config()` / `from_config()` (the model carries every constructor hyper-parameter), so `hf_utils.save_pretrained` writes `config.json` (`{format_version, organism, gene_ids, model=get_config(), experiment_config}`) + `pytorch_model.bin` and `load_pretrained` rebuilds the exact model/feature space — decoupled from the training CLI. `hf_utils.py` lives at the **package top level** (per the package-structure diagram, not under `models/` or `inference/`). `from_checkpoint` converts a CLI `{state_dict, organism, gene_ids, config}` bundle into the same directory shape so both share one format; `push_to_hub` / `from_pretrained` are thin `huggingface_hub` shells (`# pragma: no cover`, network-gated) wrapping the tested pure save/load step. Added direct dep `huggingface-hub>=0.20.0` (already transitive via `transformers`). | Making the model self-describing is the standard HF `PreTrainedModel` pattern and keeps serialization independent of `train.cli`/OmegaConf. A plain JSON+weights directory *is* a Hub-ready repo and is fully offline-testable, leaving only upload/download as the networked edge. Reusing the CLI bundle's architecture fields via `from_checkpoint` avoids two divergent on-disk formats. |
| 2026-06-26 | **Discrete-code inference API = free functions over a trained model returning an `EncodedCells` bundle, with the code→vector inverse path pushed into the layer/model.** `inference/codes.py` exposes `encode`/`encode_anndata` (→ `EncodedCells{codes (n_cells, n_codebooks) int64, size_factors, latent}`) and `decode`/`decode_to_params`. The codes alone don't reconstruct depth — decoding needs a per-cell `size_factor`, so `encode` returns it in the bundle and `decode` reuses it (overridable). The inverse `indices → summed quantized vector` is a tested `ResidualVQ.lookup` / `OmicsVQVAE.decode_codes` method rather than reaching into codebook buffers. All inference forces `eval` + `no_grad` (so EMA/dead-code stats aren't mutated) and restores the prior mode; `encode`/`decode` are batched. `encode_anndata` takes a `LoadedModel` and aligns via `align_to_reference` (annotation under `TYPE_CHECKING` to avoid an import cycle / heavy import). | Free functions keep the API thin and match the functional style of the data/train layers; the `EncodedCells` bundle makes the codes+size-factor pair (the actual compressed representation) explicit and gives a frictionless `decode(model, encode(model, x))` round-trip. Putting `lookup`/`decode_codes` on the layer/model keeps inference decoupled from quantizer internals and is reusable by generation/benchmarking PRs. Forcing eval/no_grad prevents inference from silently drifting the codebooks. |
| 2026-06-27 | **Examples = standalone runnable scripts under `examples/` (not notebooks), smoke-tested offline.** Three scripts share one synthetic-data helper (`synthetic_data.py`): local-AnnData training+`save_pretrained` (1), the full `omvqvae.inference` encode→inspect→decode→generate walk via a `save_pretrained`/`load_pretrained` round-trip (2), and Census streaming (3, network-gated). Each exposes a `main()`; `tests/test_examples.py` imports them via `importlib` and runs the offline ones end to end (example 3's `main` only under `@pytest.mark.network`). Examples live outside the `src` coverage scope, so they don't move the coverage gate. Split PR #8 into slices: scripts first (this PR), Sphinx docs second. | Plain `.py` scripts are diffable, lint/format-clean, and — unlike notebooks — runnable in CI with no `nbconvert`/`jupyter` execution layer, so the documented workflows are guaranteed to keep working. A shared synthetic helper keeps them tiny and offline. Reusing `save_pretrained`/`load_pretrained` in example 2 also exercises the HF round-trip the way a real user would (`from_pretrained` → `LoadedModel` → `encode_anndata`). Slicing keeps each run to one CI-verifiable chunk; the Sphinx scaffold needs new docs deps + a build target and slots in next. |
| 2026-06-28 | **Docs = a Sphinx `autodoc` + `napoleon` project under `docs/source/`, built warnings-as-errors and CI-gated.** `api.rst` autodocs the *implementation* modules (e.g. `omvqvae.data.dataset`, not the re-exporting `omvqvae.data` `__init__`) since autodoc skips imported members by default; `getting_started.rst` links the `examples/` scripts on GitHub rather than `literalinclude`-ing them. Build output goes to `docs/_build/` (already gitignored). Added a `docs` extra (`sphinx>=7`, `furo`) and a `make docs` target; added a dedicated **`docs` CI job** (`sphinx-build -W`) rather than deferring it to PR #10. Markdown code fences in two module docstrings were converted to RST literal blocks to keep the build warning-clean. | Reusing the already-written NumPy docstrings via autodoc keeps one source of truth for the API and makes drift a CI failure. Documenting implementation modules (not the package `__init__`) is what surfaces the actual classes/functions. Building warnings-as-errors in CI now (cheap, one job) prevents docstring rot accumulating until the release PR. `furo` is a low-config, modern theme with no extra build steps. |
| 2026-06-24 | **Training CLI = OmegaConf structured config + a single-command typer app (`oqae-train`).** The config schema is plain dataclasses (`ExperimentConfig`/`ModelConfig`/`DataConfig`/`TrainingConfig`/`TrackingConfig`) so OmegaConf validates the YAML, rejects unknown keys, and supports `--set a.b=c` dot-list overrides; `OmegaConf.to_object` yields typed objects for mypy. Pure builders (`build_model`/`build_train_config`/`build_tracker_from_config`/`build_data`) map one config section → one object and `run_experiment` wires them, so the config→objects path is unit-tested offline. The **local-AnnData** source derives its `GeneVocabulary` (hence the model's `n_genes`) from the file's own genes; the **Census** source is the one networked branch (`# pragma: no cover`). The optional checkpoint stores `{state_dict, organism, gene_ids, config}`. | A declarative, schema-checked config keeps runs reproducible and overridable from the shell; the pure-builder split keeps the heavy/networked I/O at the edges and everything else offline-testable (incl. via Typer's `CliRunner`). Persisting `gene_ids`/`organism`/`config` alongside the weights is what PR #6 (HF Hub) and PR #7 (inference) need to reload a model against the right feature space. Single-command typer means `oqae-train config.yaml` with no redundant subcommand name. |

## 🔄 **Future Roadmap (Post-v1.0)**
- **Other omics modalities**: extend the discrete-codebook approach beyond
  scRNA-seq (ATAC, protein, multi-omics joint training).
- **Foundation-model use**: treat the codebook as a token vocabulary for
  downstream sequence models.
- **Distributed streaming/training**; **auto-tuning** of codebook/architecture.
- **Scanpy/ecosystem integration**; standardized evaluation suite.

## 📞 **Contact & Contribution**
- **License**: MIT
- **Repository**: https://github.com/mengerj/oqae
- **Issues**: GitHub issue tracker

---

**Last Updated**: 2026-06-28 — PR #8 complete: Sphinx docs (slice 2) added under
`docs/source/` (autodoc + napoleon over the public API, a narrative
getting-started page linking the `examples/` scripts, `furo` theme), with a
`make docs` target and a `docs` CI job building warnings-as-errors.
**Current Focus**: PR #9 — benchmarking & scaling (raw-count+NB vs
log-normalized+Gaussian, codebook config sweeps, streaming throughput); see
`docs/STATUS.md`.
**Next Review**: After PR #9 (benchmarking & scaling).
