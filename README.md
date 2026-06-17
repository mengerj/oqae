# 🧬 OQAE: Omics Quantized Auto Encoder

A lightweight VQ-VAE library that learns a **discrete, universal latent space**
for single-cell omics. Every scRNA-seq cell is represented as a set of discrete
code vectors that can be plugged back into the trained decoder to reconstruct or
generate expression — trained at scale by **streaming the CZ CELLxGENE Census**,
or fine-tuned on your own AnnData.

[![CI](https://github.com/mengerj/oqae/actions/workflows/ci.yml/badge.svg)](https://github.com/mengerj/oqae/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mengerj/oqae/branch/main/graph/badge.svg)](https://codecov.io/gh/mengerj/oqae)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## ✨ Features

- 🧩 **Discrete Universal Latent Space**: Encode each cell as a set of discrete
  codes from shared codebooks — composable and decoder-pluggable
- 🌍 **Train by Streaming**: Stream millions of cells directly from the
  [CZ CELLxGENE Census](https://chanzuckerberg.github.io/cellxgene-census/) via
  TileDB-SOMA — no full-corpus download
- 📁 **Bring Your Own Data**: Train or fine-tune on local `.h5ad` / `.zarr`
  AnnData with the same interface
- 🔢 **Raw Counts In, Counts Out**: Model raw counts directly with a Negative
  Binomial / Zero-Inflated NB likelihood — no mandatory external normalization
- 🔄 **Residual Quantization**: Configurable VQ-VAE with multiple codebook layers
- 🧪 **Generative Decoder**: Feed discrete codes to the decoder to reconstruct or
  generate expression profiles
- 🤗 **HuggingFace Integration**: Model + codebook sharing and versioning
- 📈 **W&B Tracking**: Experiment tracking with Weights & Biases (offline-friendly)
- 🔧 **Production Ready**: strict typing (mypy), high test coverage, comprehensive CI/CD

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager
- Git
- [GitHub CLI](https://cli.github.com/) (optional, for automated workflows)

### Installation

```bash
pip install oqae
```

### Planned Usage (API in development)

> ⚠️ The library is in early development — the API below illustrates the target
> design and is **not yet implemented**. Track progress in
> [PROJECT_PLAN.md](docs/PROJECT_PLAN.md).

```python
from omvqvae import VQVAEModel
from omvqvae.data import census_dataloader, anndata_dataloader

# Option A: stream raw counts directly from the CZ CELLxGENE Census
loader = census_dataloader(
    census_version="2025-01-30",
    obs_query="tissue_general == 'blood' and is_primary_data == True",
    batch_size=512,
)

# Option B: train / fine-tune on a local AnnData file (.h5ad or .zarr)
# loader = anndata_dataloader("path/to/data.h5ad", batch_size=512)

# Model consumes RAW counts; normalization happens internally
model = VQVAEModel(
    n_codebooks=2,        # residual quantization levels
    codebook_size=512,
    likelihood="nb",      # negative binomial (or "zinb" / "gaussian")
)
model.fit(loader)

# Encode any cell to its discrete codes, and decode codes back to expression
codes = model.encode(adata)          # set of discrete code vectors per cell
expression = model.decode(codes)     # plug codes into the decoder

# Share the trained model + codebooks
model.push_to_hub("username/my-omics-model")
```

## 📋 Current Status

**🏗️ Under Active Development** — following the roadmap in
[PROJECT_PLAN.md](docs/PROJECT_PLAN.md):

- ✅ **PR #1**: Project setup, logging, and CI/CD (incl. strict mypy)
- ⏳ **PR #2**: Data layer — CELLxGENE Census streaming + local AnnData
- ⏳ **PR #3**: Residual vector-quantizer layer
- ⏳ **PR #4**: Core VQ-VAE model (raw-count NB/ZINB)
- ⏳ **PR #5**: Training/fine-tuning CLI + W&B tracking
- ⏳ **PR #6**: HuggingFace Hub integration
- ⏳ **PR #7**: Discrete-code inference/generation API

## 🏗️ Architecture Overview

### Goal
Learn a **discrete, universal latent space** for omics. Each scRNA-seq cell maps
to a set of discrete codes drawn from shared codebooks; the generative decoder
turns codes back into expression — enabling compression, integration, and
in-silico generation from a common vocabulary. We start with scRNA-seq (the
modality CELLxGENE hosts) and design for other omics later.

### Data Sources
- **CZ CELLxGENE Census (streaming)**: primary training data via TileDB-SOMA
  (`cellxgene_census` + `tiledbsoma` + `tiledbsoma-ml`), streamed in batches.
- **Local AnnData (`.h5ad` / `.zarr`)**: train or fine-tune on your own data;
  larger-than-memory files read in chunks.

### Input Modeling
Raw counts go in directly. The decoder uses a count likelihood (Negative
Binomial / Zero-Inflated NB) with library-size handled internally — following
modern single-cell VAEs (scVI family). The encoder applies an internal log1p for
numerical stability; no external normalization pipeline is required.

### Data Flow
```
CELLxGENE Census (TileDB-SOMA) ─┐
                                ├─► Streaming DataLoader ─► raw counts
Local AnnData (.h5ad / .zarr) ──┘                              │
                                                               ▼
        Encoder ─► Residual VQ (discrete codes) ─► Decoder ─► NB/ZINB → counts
                                                       │
                                                W&B experiment tracking
```

### Package Structure
```
src/omvqvae/
├── __init__.py
├── data/              # Census streaming + local AnnData loaders
├── layers/            # Residual vector-quantization layers
├── models/            # VQ-VAE model + NB/ZINB likelihoods
├── train/             # Training/fine-tuning CLI + loop (W&B)
├── inference/         # encode → codes; decode codes → expression
├── utils/             # Logging and experiment-tracking utilities
└── hf_utils.py        # HuggingFace Hub integration
```

## 🧪 Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/mengerj/oqae.git
cd oqae

# Set up development environment with uv
make setup-env
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Verify setup works
make ci
```

### Development Workflow

```bash
# View available commands
make help

# Create feature branch from issue
make branch-from-issue

# Development cycle (auto-fix mode)
make format        # Auto-fix code formatting
make test-watch    # Run tests in watch mode

# Before committing (check mode - matches GitHub CI)
make ci            # Run full pipeline exactly like GitHub Actions

# Create pull request
make pr
```

**Key Commands:**
- `make format` - **Auto-fixes** formatting issues (for development)
- `make format-check` - **Checks** formatting without fixing (matches CI)
- `make ci` - Runs the exact same checks as GitHub Actions

### Available Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make setup-env` | Set up development environment with uv |
| `make test` | Run tests with coverage |
| `make test-watch` | Run tests in watch mode |
| `make ci` | Run full CI pipeline locally |
| `make format` | Format code (black + isort) |
| `make lint` | Run linting (flake8) |
| `make type-check` | Run type checking (mypy strict) |
| `make workflow-status` | Check GitHub Actions workflow status |
| `make auto-fix` | Automatically fix workflow failures |

## 📚 Documentation

- [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) - Complete project roadmap, architecture, and design-decisions log
- [docs/STATUS.md](docs/STATUS.md) - Live status and the current next task
- [CLAUDE.md](CLAUDE.md) - Orientation guide for AI/contributor sessions
- [docs/DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md) - Development guidelines
- [docs/WORKFLOW_MONITORING.md](docs/WORKFLOW_MONITORING.md) - CI/CD monitoring

## 🤝 Contributing

We welcome contributions! This project follows a structured development approach:

1. **Check the roadmap** in [PROJECT_PLAN.md](docs/PROJECT_PLAN.md)
2. **Create an issue** for new features or bugs
3. **Follow the development workflow** with TDD and quality checks
4. **Maintain high test coverage** for new code
5. **Add comprehensive docstrings** (NumPy style)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with the [Cursor AI Python Template](https://github.com/mengerj/cursor-python-template)
- Data streaming powered by the [CZ CELLxGENE Census](https://chanzuckerberg.github.io/cellxgene-census/) and TileDB-SOMA
- Count-likelihood modeling inspired by the [scVI](https://scvi-tools.org/) family of single-cell VAEs
- Built on the AnnData ecosystem; leverages VQ-VAE for discrete representation learning

---

**Status**: Early development (PR #1 completed)
**Contact**: [GitHub Issues](https://github.com/mengerj/oqae/issues)
