# OQAE examples

Runnable, self-contained scripts demonstrating the OQAE API end to end. Examples
1, 2, and 4 run **offline in seconds** on tiny synthetic counts (see
[`synthetic_data.py`](synthetic_data.py)); example 3 streams from the CZ
CELLxGENE Census and needs network access.

| # | Script | What it shows | Offline |
|---|--------|---------------|:-------:|
| 1 | [`01_train_local_anndata.py`](01_train_local_anndata.py) | Train / fine-tune on a local `.h5ad` / `.zarr` of raw counts, then save a Hub-ready model directory. | ✅ |
| 2 | [`02_inspect_and_generate_codes.py`](02_inspect_and_generate_codes.py) | Encode cells to discrete codes, inspect codebook usage, decode codes back to expression, and generate a novel profile from edited codes. | ✅ |
| 3 | [`03_census_streaming.py`](03_census_streaming.py) | Train at scale by streaming a Census slice via TileDB-SOMA (same `Minibatch` contract as example 1). | ❌ (network) |
| 4 | [`04_benchmark_configs.py`](04_benchmark_configs.py) | Benchmark likelihood / codebook configs on shared data and print a comparison table (reconstruction, codebook utilization, separability). | ✅ |

## Running

From the repository root, with the dev environment set up (`make setup-env`):

```bash
uv run python examples/01_train_local_anndata.py
uv run python examples/02_inspect_and_generate_codes.py
uv run python examples/03_census_streaming.py   # requires network (Census)
uv run python examples/04_benchmark_configs.py
```

Each script exposes a `main()` function, so it can also be imported and driven
programmatically. The offline examples are smoke-tested in
[`tests/test_examples.py`](../tests/test_examples.py).

## Using your own data

- **Local file** (example 1): replace `make_synthetic_anndata(...)` with
  `omvqvae.data.load_anndata("your_cells.h5ad")`. Counts must be **raw** (the
  model normalizes internally). The same run is a one-liner via the CLI:

  ```bash
  oqae-train configs/train_toy.yaml -s data.path=your_cells.h5ad
  ```

- **Census** (example 3): widen the `obs_value_filter`, raise `max_steps`, and
  wire a Weights & Biases tracker for monitoring.

## API at a glance

```text
raw counts ──encode──► codes (n_cells, n_codebooks) ──decode──► expected counts
   (.h5ad / Census)     + per-cell size factor          (generative decoder)
```

- `omvqvae.data` — `GeneVocabulary`, `build_anndata_dataloader`,
  `build_census_dataloader` (organism-aware loaders → `Minibatch`).
- `omvqvae.models.OmicsVQVAE` — encoder → residual VQ → NB/ZINB/Gaussian decoder.
- `omvqvae.train.train` — source-agnostic training loop (console or W&B tracking).
- `omvqvae.inference` — `encode` / `encode_anndata` / `decode` /
  `decode_to_params` (the discrete-code latent API).
- `omvqvae.hf_utils` — `save_pretrained` / `load_pretrained` / `push_to_hub` /
  `from_pretrained` (Hub-ready serialization).
- `omvqvae.benchmark` — `run_suite` / `format_results_table` (compare
  likelihood / codebook configs: reconstruction, codebook usage, separability).
