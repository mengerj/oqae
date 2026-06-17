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
of **raw counts** and `obs_covariates` carries optional conditioning fields
(e.g. batch/dataset id). A shared **gene-space mapping** aligns each source to
the model's expected feature ordering.

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
- **Generative decoder**: maps quantized codes (optionally + size factor +
  covariates) to NB/ZINB parameters over genes.
- **Covariate conditioning**: optional categorical conditioning (e.g.
  batch/dataset id) to encourage a shared latent space across studies.
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

#### **PR #2: Data Layer — Census streaming + local AnnData**
- **Scope**: unified data interface over two sources.
- **Files**: `data/census.py`, `data/anndata_io.py`, `data/dataset.py`,
  `data/normalize.py`.
- **Key features**:
  - CELLxGENE Census streaming via `cellxgene_census` + `tiledbsoma` +
    `tiledbsoma_ml` (pinned Census version, `obs_query` filtering, raw layer).
  - Local `.h5ad` / `.zarr` loaders with chunked/backed reads.
  - Common `(raw_counts, covariates)` minibatch contract + gene-space
    alignment.
  - Size-factor computation; internal-normalization helpers.
- **Exit criteria**: stream a small Census slice and iterate a local AnnData
  through the *same* DataLoader API; tests with a tiny fixture; bounded memory.

#### **PR #3: Residual Vector Quantizer Layer**
- **Scope**: configurable residual VQ with straight-through estimator.
- **Files**: `layers/residual_vq.py`.
- **Key features**: configurable codebook count/size, commitment loss,
  perplexity/utilization metrics, codebook reset/EMA option.
- **Exit criteria**: high test coverage, gradient-flow verification.

### **Phase 2: Core Model (PRs 4–6)**

#### **PR #4: VQ-VAE Core Model (raw-count, NB/ZINB)**
- **Scope**: encoder/decoder, count likelihoods, size-factor + covariate
  conditioning, loss composition (recon + VQ).
- **Files**: `models/vqvae.py`, `models/likelihoods.py`.
- **Exit criteria**: 2-epoch smoke test on synthetic counts; NB and ZINB heads
  both train; codebooks are utilized (non-trivial perplexity).

#### **PR #5: Training/Fine-tuning CLI + W&B**
- **Scope**: OmegaConf config, typer CLI, training loop, W&B tracking, toy
  fine-tuning from a checkpoint.
- **Files**: `train/cli.py`, `train/loop.py`, `utils/tracking.py`, config
  schemas.
- **Exit criteria**: CLI trains a toy model from Census *and* from a local
  `.h5ad` in minutes; runs log to W&B (and work offline).

#### **PR #6: HuggingFace Hub Integration**
- **Scope**: serialize/deserialize model + codebooks + config; push/pull.
- **Files**: `hf_utils.py`.
- **Exit criteria**: a trained model round-trips through a HF test repo.

### **Phase 3: Latent API, Examples & Release (PRs 7–10)**

#### **PR #7: Discrete-Code Inference API**
- **Scope**: `encode(adata) → discrete codes` and
  `decode(codes) → expression`; the universal-latent use cases (compression,
  generation, plugging codes into the decoder).
- **Files**: `inference/codes.py`.
- **Exit criteria**: round-trip encode→decode on held-out cells; documented
  code-vector format.

#### **PR #8: Examples & Documentation**
- **Scope**: notebooks (Census streaming, local fine-tuning, code
  inspection/generation), Sphinx docs.
- **Exit criteria**: docs build; examples run on small data.

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

**Last Updated**: 2026-06-17 — pivot to CELLxGENE Census streaming, raw-count
(NB/ZINB) modeling, discrete universal latent space, and W&B-based monitoring.
**Next Review**: After PR #2 (data layer).
