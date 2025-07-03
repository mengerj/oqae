# 🧬 OQAE Project Plan: Omics Quantized Auto Encoder

## 🎯 **Project Overview**

**OQAE** (Omics Quantized Auto Encoder) is a lightweight, production-ready Python library that implements residual-quantized VQ-VAEs for diverse omics data stored in AnnData format. The library supports large-scale datasets (100GB+) on modest hardware (16GB RAM) through intelligent Zarr/Dask integration and provides seamless model sharing via Hugging Face Hub.

### **Core Value Proposition**
- **Memory Efficient**: Handle 100GB+ datasets with 16GB RAM constraint
- **Scalable**: Local Dask processing with distributed computing support
- **Reproducible**: HuggingFace Hub integration for model sharing
- **Flexible**: Configurable architecture for diverse omics modalities
- **Production Ready**: 90%+ test coverage, strict typing, comprehensive CI/CD

## 🏗️ **Architecture Decisions**

### **Memory Management Strategy**
- **Zarr-First Approach**: Primary storage backend for large datasets
- **AnnData Compatibility**: Support h5ad → zarr conversion for existing workflows
- **Chunked Processing**: Never load full datasets into memory
- **Dask Integration**: Lazy evaluation with configurable chunk sizes
- **Memory Mapping**: Efficient data access patterns

### **Data Flow Architecture**
```
Raw Data → AnnData(.h5ad) → Zarr Backend → Dask Arrays → VQ-VAE Model → HF Hub
```

### **Model Architecture**
- **Residual Quantization**: Configurable number of codebook layers (default: 2)
- **Batch Correction**: Categorical batch embeddings (continuous support later)
- **Modular Design**: Pluggable encoders, decoders, and quantizers
- **CPU/GPU Compatibility**: Inference on CPU, training optimized for GPU

### **Normalization Pipeline**
```python
Raw Counts → Log1p Transform → CPM Normalization → Standardization (Optional)
```
- **Pluggable Normalizers**: Support custom normalization functions
- **Batch-Aware**: Handle batch effects during normalization if needed
- **Configurable**: Skip/modify steps based on data characteristics

## 📊 **Technical Requirements**

### **Performance Constraints**
- **Memory**: Support 100GB+ datasets with 16GB RAM
- **Speed**: Training epochs < 10 minutes for 100K cells (GPU)
- **Scalability**: Linear scaling with cell count via chunking

### **Hardware Compatibility**
- **CPU**: Full inference and small-scale training support
- **GPU**: Optimized training for large datasets
- **Memory**: Efficient memory usage patterns

### **Integration Requirements**
- **Scanpy Compatibility**: Seamless integration with existing workflows
- **HuggingFace Hub**: Model sharing and versioning
- **Dask Distributed**: Future support for cluster computing

## 🗂️ **Package Structure**

```
src/omvqvae/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── io.py              # AnnData/Zarr I/O operations
│   └── preprocessing.py   # Normalization pipelines
├── layers/
│   ├── __init__.py
│   └── residual_vq.py    # Residual quantization layers
├── models/
│   ├── __init__.py
│   └── vqvae.py          # Main VQ-VAE model
├── train/
│   ├── __init__.py
│   └── cli.py            # Training CLI interface
├── utils/
│   ├── __init__.py
│   ├── logging.py        # Centralized logging
│   └── memory.py         # Memory management utilities
└── hf_utils.py           # HuggingFace Hub integration
```

## 🛣️ **Development Roadmap (10 PRs)**

### **Phase 1: Foundation (PRs 1-3)**

#### **PR #1: Logging & Utilities**
- **Scope**: Core utilities and logging infrastructure
- **Files**: `utils/logging.py`, `utils/memory.py`, basic project structure
- **Exit Criteria**: All modules can import logger, CI pipeline green
- **Memory Focus**: Memory monitoring utilities, chunk size calculators

#### **PR #2: Data I/O & Preprocessing**
- **Scope**: AnnData/Zarr I/O, Dask integration, normalization pipeline
- **Files**: `data/io.py`, `data/preprocessing.py`
- **Exit Criteria**: Round-trip tests pass, memory usage < 16GB for 100GB dataset
- **Key Features**:
  - Zarr-backed AnnData loading
  - Chunked data processing
  - Pluggable normalization pipeline
  - Memory-efficient batch processing

#### **PR #3: Residual Quantizer Layer**
- **Scope**: Core VQ layer with configurable residual levels
- **Files**: `layers/residual_vq.py`
- **Exit Criteria**: 90%+ test coverage, gradient flow verification
- **Key Features**:
  - Configurable codebook count
  - Straight-through estimator
  - Perplexity monitoring

### **Phase 2: Core Model (PRs 4-6)**

#### **PR #4: VQ-VAE Core Model**
- **Scope**: Encoder/decoder architecture, loss computation
- **Files**: `models/vqvae.py`
- **Exit Criteria**: 2-epoch smoke test on synthetic data
- **Memory Focus**: Gradient checkpointing, efficient forward pass

#### **PR #5: Training CLI & Configuration**
- **Scope**: OmegaConf-based configuration, Rich progress bars
- **Files**: `train/cli.py`, configuration schemas
- **Exit Criteria**: CLI trains toy model in < 2 minutes
- **Key Features**:
  - Memory-efficient training loop
  - Automatic batch size optimization
  - CPU/GPU compatibility

#### **PR #6: HuggingFace Integration**
- **Scope**: Model serialization, hub upload/download
- **Files**: `hf_utils.py`, model serialization
- **Exit Criteria**: Model appears on HF test repository
- **Key Features**:
  - Efficient model serialization
  - Metadata preservation
  - Version management

### **Phase 3: Production (PRs 7-10)**

#### **PR #7: Examples & Documentation**
- **Scope**: Jupyter notebooks, Sphinx documentation
- **Files**: `examples/`, comprehensive docs
- **Exit Criteria**: RTD build passes, examples run successfully

#### **PR #8: Performance & Scaling**
- **Scope**: Dask cluster support, benchmarking
- **Files**: Scaling utilities, benchmarking scripts
- **Exit Criteria**: Benchmark report shows linear scaling

#### **PR #9: Pre-v1 Polish**
- **Scope**: mypy strict mode, contribution guidelines
- **Files**: Type annotations, developer docs
- **Exit Criteria**: Release pipeline passes all checks

#### **PR #10: v1.0 Release**
- **Scope**: API freeze, final testing, announcement
- **Exit Criteria**: PyPI 1.0.0 release, complete documentation

## 🧪 **Development Workflow**

### **Testing Strategy**
- **Unit Tests**: pytest with 90%+ coverage requirement
- **Integration Tests**: Full pipeline tests with real data
- **Performance Tests**: Memory usage and speed benchmarks
- **Smoke Tests**: < 10 second execution time

### **Code Quality Standards**
- **Type Hints**: Strict mypy configuration
- **Docstrings**: NumPy style for all public APIs
- **Formatting**: Black + isort + flake8
- **Security**: Bandit scanning for dependencies

### **AI-First Development**
- **Cursor Integration**: Optimized for AI assistance
- **Clean Architecture**: SOLID principles throughout
- **Comprehensive Logging**: Rich debugging information
- **Test-Driven Development**: Tests before implementation

## 📋 **Dependencies**

### **Core Dependencies**
```toml
dependencies = [
    "torch>=2.0.0",           # Deep learning framework
    "anndata>=0.8.0",         # Omics data structure
    "zarr>=2.12.0",           # Chunked array storage
    "dask[array]>=2023.1.0",  # Parallel computing
    "scanpy>=1.9.0",          # Single-cell analysis
    "numpy>=1.21.0",          # Numerical computing
    "pandas>=1.5.0",          # Data manipulation
    "scipy>=1.9.0",           # Scientific computing
    "transformers>=4.20.0",   # HuggingFace integration
    "omegaconf>=2.3.0",       # Configuration management
    "rich>=13.0.0",           # Rich terminal output
    "typer>=0.9.0",           # CLI framework
]
```

### **Development Dependencies**
- **Testing**: pytest, hypothesis, pytest-cov
- **Quality**: black, isort, flake8, mypy, bandit
- **Docs**: sphinx, jupyter, nbsphinx
- **CI/CD**: pre-commit, GitHub Actions

## 🎯 **Success Metrics**

### **Performance Targets**
- **Memory**: < 16GB for 100GB datasets
- **Speed**: < 10 min/epoch for 100K cells (GPU)
- **Coverage**: > 90% test coverage
- **Type Safety**: 100% mypy compliance

### **User Experience Goals**
- **Installation**: `pip install oqae` → working in < 5 minutes
- **Learning Curve**: First model trained in < 30 minutes
- **Integration**: Drop-in replacement for existing workflows
- **Documentation**: Complete examples for all use cases

## 🔄 **Future Roadmap (Post-v1.0)**

### **Advanced Features**
- **Multi-modal VQ-VAE**: Joint training across modalities
- **Distributed Training**: Dask cluster optimization
- **Advanced Metrics**: Batch mixing, biological preservation
- **Auto-tuning**: Hyperparameter optimization

### **Ecosystem Integration**
- **Scanpy Plugin**: Native scanpy tool integration
- **Bioconductor Bridge**: R package for broader adoption
- **Cloud Deployment**: Docker containers, Kubernetes support
- **Benchmark Suite**: Standardized evaluation metrics

## 📞 **Contact & Contribution**

- **License**: MIT (academic/commercial friendly)
- **Repository**: https://github.com/mengerj/oqae
- **Documentation**: (To be set up with Sphinx)
- **Issues**: GitHub issue tracker for bugs/features

---

**Last Updated**: Initial version created during project setup
**Next Review**: After PR #3 completion
