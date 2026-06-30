# 🧬 OQAE: Omics Quantized Auto Encoder

OQAE learns a **discrete, universal latent space for single-cell omics** with a
residual VQ-VAE. Every scRNA-seq cell is encoded as a small set of discrete codes
(codebook indices); a generative decoder turns those codes back into expression.
Train at scale by **streaming the CZ CELLxGENE Census**, or on your own AnnData —
the model ingests **raw counts** and reconstructs them with a Negative-Binomial /
Zero-Inflated-NB likelihood (scVI-style library-size handling), so no external
normalization is required.

The Python package is `omvqvae`; the distribution name is `oqae`.

[![CI](https://github.com/mengerj/oqae/actions/workflows/ci.yml/badge.svg)](https://github.com/mengerj/oqae/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mengerj/oqae/branch/main/graph/badge.svg)](https://codecov.io/gh/mengerj/oqae)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## ✨ What you get

- **Discrete universal latent space** — each cell becomes an `int64`
  `(n_codebooks,)` code drawn from shared codebooks; codes are decoder-pluggable.
- **Train by streaming or locally** — stream millions of cells from the
  [CZ CELLxGENE Census](https://chanzuckerberg.github.io/cellxgene-census/) via
  TileDB-SOMA, or train/fine-tune on local `.h5ad` / `.zarr` AnnData through the
  same `Minibatch` contract.
- **Raw counts in, counts out** — NB / ZINB likelihood with internal library-size
  handling; a log-normalized Gaussian head is a pluggable alternative.
- **Residual vector quantization** — configurable number of codebooks and entries.
- **Discrete-code inference** — `encode` cells → codes, `decode` codes → expected
  counts, and generate novel profiles from edited codes.
- **W&B tracking** (offline-friendly) and a **benchmarking harness** to pick an
  architecture before committing to a full Census run.
- **HuggingFace-style serialization** — `save_pretrained` / `load_pretrained` /
  `push_to_hub`.

## 📦 Installation

```bash
pip install oqae
```

For development (uv-managed virtualenv with all extras):

```bash
git clone https://github.com/mengerj/oqae.git
cd oqae
make setup-env          # uv sync --all-extras → .venv
source .venv/bin/activate
make ci                 # format-check + flake8 + mypy + pytest
```

## 🚀 Quick start

### Train on a local AnnData (CLI)

The fastest path is the `oqae-train` console script driven by a YAML config.
Point `data.path` at a local `.h5ad` / `.zarr` of **raw counts** — the reference
gene vocabulary is derived from that file's genes:

```bash
oqae-train configs/train_toy.yaml -s data.path=your_cells.h5ad
```

`-s a.b=c` overrides any config key. See [`configs/train_toy.yaml`](configs/train_toy.yaml)
for the full set of knobs (model, data, training, tracking).

### Train in Python

```python
from omvqvae.data import GeneVocabulary, load_anndata, extract_counts, build_anndata_dataloader
from omvqvae.models import OmicsVQVAE
from omvqvae.train import TrainConfig, train

adata = load_anndata("your_cells.h5ad")          # raw counts
counts, gene_ids = extract_counts(adata)
vocab = GeneVocabulary("homo_sapiens", gene_ids)
loader = build_anndata_dataloader(adata, vocab, batch_size=128, shuffle=True)

model = OmicsVQVAE(
    n_genes=vocab.n_genes,
    n_latent=32,
    hidden_dims=(256, 128),
    likelihood="nb",        # "nb" | "zinb" | "gaussian"
    n_codebooks=2,          # residual quantization levels
    codebook_size=512,
)
train(model, loader, config=TrainConfig(max_epochs=10, lr=1e-3))
```

### Encode → inspect → decode → generate

```python
from omvqvae.inference import encode, decode, decode_to_params

enc = encode(model, counts)        # EncodedCells: .codes (n_cells, n_codebooks) int64, .latent, .size_factors
expected = decode(model, enc)      # codes → expected counts (generative decoder)
params = decode_to_params(model, enc.codes, enc.size_factors)  # full head distribution → sample/generate
```

Save and share a trained model (HuggingFace-style directory):

```python
from omvqvae.hf_utils import save_pretrained, load_pretrained
save_pretrained(model, vocab, "my-omics-model")
loaded = load_pretrained("my-omics-model")     # .model, .vocabulary, .experiment_config
```

Runnable, self-contained versions of all of the above live in
[`examples/`](examples/) (examples 1, 2, 4 run offline in seconds).

## 📊 Choosing an architecture: benchmark on a Census subset

Before committing GPU-hours to a full Census run, use the **benchmarking harness**
to compare likelihood and codebook configurations on a *small but meaningful*
slice of the Census, then pick the winner.

### 1. Pull a representative Census slice to a local file

Materialize a few thousand cells with `cellxgene_census` so the sweep is fast and
re-runs deterministically. Choose a slice broad enough to contain real biological
structure (multiple cell types / tissues) — that structure is what the
separability metric scores:

```python
import cellxgene_census
from omvqvae.data import DEFAULT_CENSUS_VERSION

with cellxgene_census.open_soma(census_version=DEFAULT_CENSUS_VERSION) as census:
    adata = cellxgene_census.get_anndata(
        census,
        organism="homo_sapiens",
        obs_value_filter=(
            "tissue_general == 'blood' and is_primary_data == True "
            "and assay == '10x 3\\' v3'"
        ),
        column_names={"obs": ["cell_type", "tissue_general"]},
    )
adata.write_h5ad("census_blood_subset.h5ad")     # raw counts + a `cell_type` label
```

Keep it to roughly **2k–20k cells**: large enough that codebook utilization and
separability are meaningful, small enough to sweep many configs in minutes.

### 2. Run the sweep

`run_suite` trains each config on the *same* data and evaluates it on a held-out
set, returning one row per config. Use the slice's `cell_type` as the
separability label:

```python
from omvqvae.data import GeneVocabulary, load_anndata, extract_counts, build_anndata_dataloader
from omvqvae.benchmark import BenchmarkConfig, run_suite, format_results_table

adata = load_anndata("census_blood_subset.h5ad")
counts, gene_ids = extract_counts(adata)
vocab = GeneVocabulary("homo_sapiens", gene_ids)
labels = list(adata.obs["cell_type"])
loader = build_anndata_dataloader(adata, vocab, batch_size=256, shuffle=True)

configs = [
    BenchmarkConfig(name="nb-2x512",       likelihood="nb",       n_codebooks=2, codebook_size=512, max_epochs=20),
    BenchmarkConfig(name="zinb-2x512",     likelihood="zinb",     n_codebooks=2, codebook_size=512, max_epochs=20),
    BenchmarkConfig(name="gaussian-2x512", likelihood="gaussian", n_codebooks=2, codebook_size=512, max_epochs=20),
    BenchmarkConfig(name="nb-4x512",       likelihood="nb",       n_codebooks=4, codebook_size=512, max_epochs=20),
    BenchmarkConfig(name="nb-2x1024",      likelihood="nb",       n_codebooks=2, codebook_size=1024, max_epochs=20),
]

results = run_suite(configs, loader, n_genes=vocab.n_genes, eval_counts=counts, eval_labels=labels)
print(format_results_table(results))
```

This prints a Markdown comparison table:

```
| name      | likelihood | codebooks | train_loss | eval_nll | eval_mae | perplexity | utilization | separability |
| --------- | ---------- | --------- | ---------- | -------- | -------- | ---------- | ----------- | ------------ |
| nb-2x512  | nb         | 2x512     | ...        | ...      | ...      | ...        | ...         | ...          |
| ...       | ...        | ...       | ...        | ...      | ...      | ...        | ...         | ...          |
```

[`examples/04_benchmark_configs.py`](examples/04_benchmark_configs.py) runs this
end-to-end on synthetic data (no network) if you want to see it first.

### 3. How to read the table — picking the winner

| Metric | What it measures | What "good" looks like |
|--------|------------------|------------------------|
| `eval_nll` | Held-out reconstruction negative log-likelihood (the model's own likelihood). | **Lower is better.** Comparable *only within a fixed likelihood* (NB vs NB, sweeping codebooks) — **not** across NB/ZINB/Gaussian, which have different units. |
| `eval_mae` | Mean abs. error of expected vs. target in the head's native space (raw counts for NB/ZINB, `log1p` for Gaussian). | **Lower is better.** A cross-likelihood sanity check, but note the differing target spaces. |
| `perplexity` | Effective number of codebook entries in use (ceiling = `codebook_size`). | **Higher is better.** Near 1 means **codebook collapse** — the bottleneck degenerated to a few codes. Want it a healthy fraction of `codebook_size`. |
| `utilization` | Fraction of codebook entries used at least once, in `[0, 1]`. | **Higher is better.** Low utilization means most of the codebook is wasted; shrink `codebook_size` or check for collapse. |
| `separability` | Nearest-centroid accuracy of the latent against `cell_type` labels. | **Higher is better** (chance ≈ `1 / n_cell_types`). This is the downstream-usefulness proxy: does the latent keep biologically distinct cells apart? |

**Selection rule of thumb:** prefer the config with the **highest separability**
that also keeps **perplexity / utilization high** (no collapse) and competitive
reconstruction. If a large codebook shows low utilization, it's over-provisioned —
drop to a smaller `codebook_size`. If reconstruction is poor everywhere, widen
`hidden_dims` / `n_latent` or train longer. Record the chosen config (and this
table) so the full-scale run is reproducible.

> **Likelihood guidance.** Start with `nb` for raw counts; try `zinb` if your data
> is very sparse / zero-inflated (compare `eval_mae` and `separability`, since
> `eval_nll` isn't comparable across the two). `gaussian` targets `log1p`
> expression and is the log-normalized alternative — only pick it if you have a
> reason to model normalized data.

### 4. A reproducible written report

A committed, regenerable version of this analysis lives in
[docs/benchmark_report.md](docs/benchmark_report.md) — a full likelihood ×
codebook sweep with the results table **and** a written interpretation
(reconstruction, no-collapse, separability, raw-count NB vs log-normalized
Gaussian, and the batch-effect question). On the reference fixture it lands on
**raw-count NB** with a **modestly sized codebook** (large codebooks show low
utilization at small data scale); use it as the template for your own data:

```python
from omvqvae.benchmark import generate_report, make_benchmark_fixture, default_report_configs
# swap make_benchmark_fixture(...) for your own (loader, eval_counts, eval_labels)
```

`omvqvae.benchmark.generate_report` writes the same Markdown for any sweep;
[`examples/05_benchmark_report.py`](examples/05_benchmark_report.py) regenerates
`docs/benchmark_report.md` offline in seconds.

## 🌍 Train at scale by streaming the Census (with W&B)

Once you've chosen an architecture, train it on the full slice by **streaming**
from the Census — no full-corpus download. Set the chosen hyper-parameters in your
config and switch the data source to `census`:

```yaml
# my_census_run.yaml  (merged onto omvqvae.train.cli.ExperimentConfig defaults)
model:
  n_latent: 32
  hidden_dims: [256, 128]
  likelihood: nb            # <- the winner from the benchmark
  n_codebooks: 2
  codebook_size: 512
data:
  source: census
  organism: homo_sapiens
  obs_value_filter: "tissue_general == 'blood' and is_primary_data == True"
  batch_size: 512
training:
  max_epochs: 1
  max_steps: 50000
  lr: 1.0e-3
  device: cuda
  checkpoint_path: runs/blood_nb_2x512.pt
tracking:
  backend: wandb            # console | none | wandb
  project: oqae
  run_name: blood-nb-2x512
  offline: false            # true for air-gapped / sync later with `wandb sync`
```

```bash
oqae-train my_census_run.yaml
```

Or stream directly in Python with `build_census_dataloader` (see
[`examples/03_census_streaming.py`](examples/03_census_streaming.py)).

**Sizing the run.** Before launching a long job, profile streaming throughput on
your filter with `omvqvae.benchmark.benchmark_census_throughput` (cells/s,
batches/s, optionally including a per-batch train step) to estimate wall-clock to
N steps — see [`examples/06_census_throughput.py`](examples/06_census_throughput.py):

```python
from omvqvae.benchmark import benchmark_census_throughput, format_throughput_table
res = benchmark_census_throughput(
    "homo_sapiens",
    obs_value_filter="tissue_general == 'blood' and is_primary_data == True",
    batch_size=512, max_batches=50,
)
print(format_throughput_table([res]))
```

### What to watch on the W&B dashboard

The training loop logs these scalar series each step (prefix `train/`):

| Metric | Read it as | Success signal |
|--------|-----------|----------------|
| `train/loss` | Total objective (reconstruction + VQ). | Decreasing, then plateauing. |
| `train/reconstruction_loss` | How well counts are reconstructed. | Steady decrease — the main quality signal. |
| `train/vq_loss`, `train/commitment_loss`, `train/codebook_loss` | Quantizer health: encoder commits to codes and codebooks track the encoder. | Decreasing / stable; a blowing-up commitment loss means the encoder and codebook are fighting. |
| `train/perplexity` | Mean effective codes in use across levels. | **High and stable.** A crash toward 1 = codebook collapse — the run is degenerate even if loss looks fine. |
| `train/perplexity/codebook_{j}`, `train/usage/codebook_{j}` | Per-level diversity / utilization. | Each level staying utilized; a dead level suggests too many codebooks. |

**How to judge success:** reconstruction loss should fall and plateau *while*
perplexity/utilization stay high. Loss dropping *because* the codebook collapsed
(perplexity → 1) is a failure mode, not a win. After training, re-run the
benchmark metrics (`eval_nll`, `eval_mae`, `separability`) on a held-out Census
slice to confirm the latent generalizes and separates cell types — that's the
real measure of a good universal latent space.

## 🏗️ How it fits together

```
CELLxGENE Census (TileDB-SOMA) ─┐
                                ├─► DataLoader → Minibatch (raw counts + size factors)
Local AnnData (.h5ad / .zarr) ──┘                        │
                                                         ▼
   Encoder ─► Residual VQ (discrete codes) ─► Decoder ─► NB/ZINB/Gaussian → counts
                                                  │
                                          W&B experiment tracking
```

| Package | Role |
|---------|------|
| `omvqvae.data` | `GeneVocabulary`, `build_anndata_dataloader`, `build_census_dataloader` — organism-aware loaders yielding `Minibatch`. |
| `omvqvae.layers` | `ResidualVQ` / `VectorQuantizer` — the discrete bottleneck. |
| `omvqvae.models` | `OmicsVQVAE` (encoder → residual VQ → decoder) and the NB/ZINB/Gaussian heads. |
| `omvqvae.train` | `train` (source-agnostic loop) + `oqae-train` config-driven CLI. |
| `omvqvae.inference` | `encode` / `encode_anndata` / `decode` / `decode_to_params`. |
| `omvqvae.benchmark` | `run_suite` / `format_results_table` (compare configs), `generate_report` (written report), `benchmark_census_throughput` (streaming throughput). |
| `omvqvae.hf_utils` | `save_pretrained` / `load_pretrained` / `push_to_hub`. |

**Design decisions** (raw-count NB/ZINB, Census streaming, human+mouse, v1
unconditional, W&B monitoring) and the full roadmap live in
[docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md).

## 📚 Documentation

- [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) — roadmap, architecture, decisions log.
- [docs/STATUS.md](docs/STATUS.md) — live status and the current next task.
- [docs/benchmark_report.md](docs/benchmark_report.md) — architecture-selection benchmark + interpretation.
- [examples/](examples/) — runnable end-to-end scripts (01–06).
- API reference — `make docs` builds the Sphinx site into `docs/_build/html`.

## 📄 License

MIT — see [LICENSE](LICENSE).
