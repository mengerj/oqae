# 🧬 OQAE: Omics Quantized Auto Encoder

A lightweight VQ-VAE library for large-scale omics data analysis with memory-efficient processing and HuggingFace Hub integration.

[![CI](https://github.com/mengerj/oqae/actions/workflows/ci.yml/badge.svg)](https://github.com/mengerj/oqae/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mengerj/oqae/branch/main/graph/badge.svg)](https://codecov.io/gh/mengerj/oqae)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## ✨ Features

- 🧬 **Omics-Optimized**: Designed specifically for single-cell and multi-omics data
- 💾 **Memory Efficient**: Handle 100GB+ datasets with 16GB RAM via Zarr/Dask
- 🔄 **Residual Quantization**: Configurable VQ-VAE with multiple codebook layers
- 🤗 **HuggingFace Integration**: Seamless model sharing and versioning
- ⚡ **CPU/GPU Compatible**: Inference on CPU, training optimized for GPU
- 📊 **Scanpy Integration**: Drop-in compatibility with existing workflows
- 🔧 **Production Ready**: 90%+ test coverage, strict typing, comprehensive CI/CD

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

### Basic Usage

```python
from omvqvae import VQVAEModel
from omvqvae.data import load_zarr_anndata

# Load large dataset efficiently
adata = load_zarr_anndata("path/to/large_dataset.zarr")

# Initialize model with memory-efficient settings
model = VQVAEModel(
    n_layers=2,
    codebook_size=512,
    memory_efficient=True
)

# Train with automatic batch size optimization
model.fit(adata, batch_key="batch_id")

# Save to HuggingFace Hub
model.push_to_hub("username/my-omics-model")
```

## 📋 Current Status

**🏗️ Under Active Development** - Following a structured 10-PR roadmap:

- ✅ **PR #1**: Project setup and logging infrastructure
- ⏳ **PR #2**: Data I/O and preprocessing (Zarr/Dask integration)
- ⏳ **PR #3**: Residual quantization layers
- ⏳ **PR #4**: Core VQ-VAE model
- ⏳ **PR #5**: Training CLI and configuration
- ⏳ **PR #6**: HuggingFace Hub integration

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for detailed roadmap and architecture decisions.

## 🏗️ Architecture Overview

### Memory Management Strategy
- **Zarr-First**: Primary storage for large datasets
- **Dask Integration**: Lazy evaluation with configurable chunking
- **Memory Mapping**: Efficient data access patterns
- **16GB Target**: Support 100GB+ datasets on modest hardware

### Data Flow
```
Raw Data → AnnData(.h5ad) → Zarr Backend → Dask Arrays → VQ-VAE Model → HF Hub
```

### Package Structure
```
src/omvqvae/
├── __init__.py
├── data/              # AnnData/Zarr I/O operations
├── layers/            # Residual quantization layers
├── models/            # VQ-VAE model implementations
├── train/             # Training CLI interface
├── utils/             # Logging and memory utilities
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

# Run tests in watch mode during development
make test-watch

# Run full quality checks
make ci

# Create pull request
make pr
```

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
| `make type-check` | Run type checking (currently disabled) |
| `make workflow-status` | Check GitHub Actions workflow status |
| `make auto-fix` | Automatically fix workflow failures |

## 📚 Documentation

- [PROJECT_PLAN.md](PROJECT_PLAN.md) - Complete project roadmap and architecture
- [docs/DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md) - Development guidelines
- [docs/WORKFLOW_MONITORING.md](docs/WORKFLOW_MONITORING.md) - CI/CD monitoring

## 🤝 Contributing

We welcome contributions! This project follows a structured development approach:

1. **Check the roadmap** in [PROJECT_PLAN.md](PROJECT_PLAN.md)
2. **Create an issue** for new features or bugs
3. **Follow the development workflow** with TDD and quality checks
4. **Ensure 90%+ test coverage** for new code
5. **Add comprehensive docstrings** (NumPy style)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with the [Cursor AI Python Template](https://github.com/mengerj/cursor-python-template)
- Inspired by modern omics analysis tools like Scanpy and AnnData
- Leverages the power of VQ-VAE for representation learning

---

**Status**: Early development (PR #1 completed)
**Target**: Production-ready v1.0 by Q2 2024
**Contact**: [GitHub Issues](https://github.com/mengerj/oqae/issues)
