# OQAE — Live Status & Handoff

> Single source of truth for "where are we and what's next." **Update this at the
> end of every working session** so the next session can pick up cold. Keep it
> short; deep rationale lives in `docs/PROJECT_PLAN.md`.

**Last updated:** 2026-06-23

## Current state

- **PR #1 (Foundation) — DONE and merged to `main`.**
  - Logging (`src/omvqvae/utils/logging.py`), uv dev workflow, CI/CD
    (black/isort/flake8/**mypy strict**/pytest, bandit/safety), pre-commit.
  - Template leftovers and the old `system_monitor` utility have been removed.
  - `pyproject.toml` core deps trimmed; `wandb` kept for tracking.
  - `make ci` is green; pytest at 100% coverage on the current (small) codebase.
- **PR #2 (data layer) — DONE (both slices).**
  - Slice 1 (merged): `data/normalize.py` (size factors + internal log1p/depth
    normalization), `data/dataset.py` (organism-aware `GeneVocabulary`,
    `align_to_reference`, the shared `Minibatch`
    `(counts, size_factors, covariates)` contract, `CountsDataset`,
    `collate_minibatch`), `data/anndata_io.py` (local `.h5ad` / `.zarr` loaders +
    `build_anndata_dataloader`). Human *and* mouse AnnData iterate one API.
  - Slice 2 (this PR): `data/census.py` — streams raw counts from the CELLxGENE
    Census via `cellxgene_census` + `tiledbsoma` + `tiledbsoma_ml`
    (`ExperimentDataset` + `experiment_dataloader()`), behind the *same*
    `Minibatch` contract. `organism` selects the `homo_sapiens` / `mus_musculus`
    experiment; `obs_value_filter` / `var_value_filter` slice cells/genes; the
    raw layer is read; the per-organism reference defaults to the Census `var`
    index. `CensusMinibatchLoader` adapts the streamed `(X, obs)` chunks to
    `Minibatch`; `census_chunk_to_minibatch` is the offline-tested glue.
  - Added deps `cellxgene-census`, `tiledbsoma`, `tiledbsoma-ml`; `uv.lock`
    refreshed. Pinned default Census release `DEFAULT_CENSUS_VERSION =
    "2025-11-08"` (newest LTS). Offline tests at 100% coverage; a `network`-marked
    live test streams **human and mouse** end-to-end (skipped by default,
    verified passing live this run). `make ci` green.
- **PR #3 (residual VQ layer) — DONE and merged to `main` (PR #24).**
  - `layers/residual_vq.py` — `VectorQuantizer` (single codebook: straight-through
    estimator, commitment + codebook losses, optional EMA codebook updates with
    Laplace smoothing, dead-code reset) and `ResidualVQ` (stacks `n_codebooks`
    quantizers over successive residuals; per-level indices = the cell's discrete
    code; summed quantized vectors approximate the input). Per-forward
    perplexity/utilization metrics via `QuantizerOutput` / `ResidualVQOutput`.
- **PR #4 slice 1 (reconstruction likelihoods) — DONE and merged to `main`.**
  - `models/likelihoods.py`: pure log-prob functions `log_nb_positive`,
    `log_zinb_positive`, `log_gaussian` (SciPy-checked) plus the decoder heads
    `NBHead`, `ZINBHead`, `GaussianHead` behind one `ReconstructionHead`
    interface (`forward` → params, `reconstruction_loss`, `expected_counts`)
    and a `build_reconstruction_head` factory. scVI-style count parameterization
    (softmax `px_scale` × library size → NB mean; gene-wise dispersion). No new
    deps. Offline tests at 100% coverage; `make ci` green.
- **PR #4 slice 2 (encoder/decoder VQ-VAE core) — DONE (this PR). PR #4 complete.**
  - `models/vqvae.py`: `OmicsVQVAE` (`nn.Module`) wires the three stages — an
    MLP encoder (raw counts → internal `log1p` → hidden → `n_latent`), the
    `ResidualVQ` bottleneck (`omvqvae.layers`), and a mirrored decoder feeding a
    `ReconstructionHead` (`omvqvae.models.likelihoods`). `forward(counts,
    size_factors)` returns a `VQVAEOutput` composing recon + VQ loss and exposing
    the per-level discrete codes, the latent / quantized vectors, and codebook
    perplexity/utilization. Convenience methods: `encode`, `quantize`,
    `encode_codes`, `decode`, `expected_counts`. NB/ZINB reconstruct raw counts;
    the Gaussian head targets `log1p` expression. No new deps. Offline tests at
    100% coverage incl. a synthetic 40-step smoke-train (NB + ZINB) where the
    recon loss decreases and codebooks stay utilized; `make ci` green.
- **PR #5 slice 1 (tracker + training loop) — DONE (this run).**
  - `utils/tracking.py`: an offline-friendly `ExperimentTracker` (ABC) with a
    `ConsoleTracker` default (logs metrics through the OQAE logger; no `wandb`
    needed) and a `WandbTracker` thin shell over an injected live run. The lazy
    `wandb.init` lives in `_init_wandb_run` (`# pragma: no cover` I/O shell);
    `build_tracker(backend="console"|"none"|"wandb", ...)` dispatches and logs
    the run config. `vqvae_metrics` flattens a `VQVAEOutput` (losses + codebook
    perplexity/usage, expanded per level) to `{name: float}`.
  - `train/loop.py`: `train(model, data_source, *, config, optimizer, tracker)`
    — a source-agnostic loop that pulls `Minibatch`es from any iterable
    (local-AnnData `DataLoader` now, Census later), runs `OmicsVQVAE`, steps an
    (injectable) optimizer, clips grads, honors `max_steps`, and logs through
    the tracker. `TrainConfig`/`EpochMetrics`/`TrainResult` bundle the knobs and
    outcome. Exported from `train/__init__.py`.
  - No new deps (`wandb`/`omegaconf`/`typer`/`rich` already declared). Offline
    tests at ~99% coverage incl. a synthetic multi-epoch run where the
    reconstruction loss drops; `make ci` green.
- Plan/docs reflect the pivot: Census streaming, raw-count NB/ZINB modeling,
  discrete universal latent space, human+mouse, v1 unconditional, W&B monitoring.

## Next task — PR #5 slice 2: training/fine-tuning CLI

Slice 1 (tracker + training loop) is **DONE** (this run). Slice 2 wires a
config-driven entry point on top of the loop:

1. `train/cli.py` — an OmegaConf + typer CLI to launch training/fine-tuning from
   a config (organism, data source, model hyper-params, likelihood, tracking
   backend). All needed deps (`omegaconf`, `typer`, `rich`, `wandb`) are
   **already in `pyproject.toml`** — verified this run — so no `uv.lock` refresh
   is expected unless a new dep is introduced.
2. Build a model from the config (`OmicsVQVAE`), build the data source from the
   config (local `.h5ad`/`.zarr` via `build_anndata_dataloader` now;
   Census-gated path optional), build a tracker via
   `omvqvae.utils.tracking.build_tracker`, and call
   `omvqvae.train.train(model, loader, config=TrainConfig(...), tracker=...)`.
3. Ship an example config (e.g. `configs/train_toy.yaml`) and a tiny synthetic
   `.h5ad` fixture (or generate one in a test) so the CLI is exercised offline.

**Definition of done:** a CLI trains a toy model from a local `.h5ad` (and,
network-gated, from Census) in minutes; runs log to W&B and also work offline;
offline tests on the CLI's config→objects wiring (invoke via typer's
`CliRunner`); CI green; PR opened.

### Available building blocks (slice 1, this run)

- `omvqvae.utils.tracking`: `ExperimentTracker` (ABC), `ConsoleTracker`
  (offline default, logs via the OQAE logger), `WandbTracker` (wraps an
  injected live run), `build_tracker(backend=...)` (lazy `wandb.init` lives in
  `_init_wandb_run`), and `vqvae_metrics(output, prefix=...)` flattening a
  `VQVAEOutput` to `{name: float}`.
- `omvqvae.train`: `train(model, data_source, *, config, optimizer, tracker)`
  plus `TrainConfig` / `EpochMetrics` / `TrainResult`. Source-agnostic: pass any
  iterable of `Minibatch` (a `DataLoader` for multi-epoch). `TrainConfig`
  carries `max_epochs`/`lr`/`weight_decay`/`grad_clip_norm`/`max_steps`/
  `log_every`/`device`.

## Open questions / parked

- **Cross-organism unification** (single shared latent across human+mouse):
  deferred past v1; needs ortholog mapping or shared gene embedding.
- **NB vs ZINB vs log-normalized+Gaussian** for the quantized setting: support
  both; benchmark in PR #9.
- **Batch-effect conditioning**: v1 unconditional; benchmark whether it's needed
  (PR #9) before adding.
- **Census version pin**: now `DEFAULT_CENSUS_VERSION = "2025-11-08"` (newest LTS
  at implementation time). Bump when a newer LTS lands; it is a configurable
  arg, so callers can override per run.
- **Census gene panel size**: the default reference is the *full* Census `var`
  index (~tens of thousands of genes per organism). PR #4/#9 may want a curated
  highly-variable-gene panel via `var_value_filter`; left to the model PRs.

## Changelog (most recent first)

- **2026-06-23** — PR #5 slice 1: experiment tracker + training loop. Added
  `utils/tracking.py` (`ExperimentTracker` ABC, `ConsoleTracker`,
  `WandbTracker`, `build_tracker`, `vqvae_metrics`) — offline by default
  (console logger); the `wandb` import is lazy and isolated in `_init_wandb_run`,
  while `WandbTracker` wraps an injected run so its logic is tested offline with
  a fake. Added `train/loop.py` (`train`, `TrainConfig`, `EpochMetrics`,
  `TrainResult`) — a source-agnostic loop over any iterable of `Minibatch`,
  injectable optimizer/tracker, grad clipping, and `max_steps`. Exported from
  `train/__init__.py`. No new deps; `uv.lock` unchanged. Offline tests at ~99%
  coverage (synthetic `CountsDataset` + `DataLoader`, recording fake tracker,
  loss-decrease smoke run); `make ci` green. Slice 2 (typer/OmegaConf CLI) is
  the next chunk.
- **2026-06-22** — PR #4 slice 2: encoder/decoder VQ-VAE core. Added
  `models/vqvae.py` — `OmicsVQVAE` composing an MLP encoder (raw counts →
  internal `log1p` → `n_latent`), the `ResidualVQ` bottleneck, and a mirrored
  decoder feeding a `ReconstructionHead`. `forward(counts, size_factors)`
  returns a `VQVAEOutput` bundling the composed reconstruction + VQ loss, the
  per-level discrete codes, latent/quantized vectors, and codebook
  perplexity/utilization; plus `encode`/`quantize`/`encode_codes`/`decode`/
  `expected_counts` helpers. NB/ZINB reconstruct raw counts; the Gaussian head
  targets `log1p` expression. Exported from `models/__init__.py`. No new deps;
  `uv.lock` unchanged. Offline tests at 100% coverage incl. a synthetic 40-step
  smoke-train (NB + ZINB) where the reconstruction loss decreases and codebooks
  stay utilized; `make ci` green. **PR #4 is now complete.**
- **2026-06-21** — PR #4 slice 1: reconstruction likelihoods. Added
  `models/likelihoods.py` — pure log-prob functions (`log_nb_positive`,
  `log_zinb_positive`, `log_gaussian`, validated against SciPy) and decoder heads
  (`NBHead`, `ZINBHead`, `GaussianHead`) under a shared `ReconstructionHead`
  interface plus a `build_reconstruction_head` factory. scVI-style count
  parameterization (softmax proportions × library size → NB mean; gene-wise
  dispersion); Gaussian head for the log-normalized alternative. Exported from
  `models/__init__.py`. No new deps (`torch`/`scipy` already present);
  `uv.lock` unchanged. Offline tests at 100% coverage; `make ci` green. Built as
  an independent slice (no VQ dependency) while PR #3 was in flight; merged `main`
  in after PR #24 landed.
- **2026-06-20** — PR #3: residual vector-quantizer layer (PR #24). Added
  `layers/residual_vq.py` (`VectorQuantizer`, `ResidualVQ`, and the
  `QuantizerOutput` / `ResidualVQOutput` result bundles): straight-through
  estimator, commitment + codebook losses, optional EMA codebook updates with
  Laplace smoothing, dead-code reset, and per-forward perplexity/utilization
  metrics. `ResidualVQ` stacks `n_codebooks` (default 2) quantizers over
  successive residuals so each cell becomes a small set of codebook indices.
  Exported from `layers/__init__.py`. Offline synthetic tests at 100% coverage;
  no new dependencies; `make ci` green.
- **2026-06-19** — PR #2 slice 2: CELLxGENE Census streaming. Added
  `data/census.py` (`open_census`, `census_gene_vocabulary`,
  `census_chunk_to_minibatch`, `CensusMinibatchLoader`,
  `build_census_dataloader`) streaming raw counts via TileDB-SOMA
  (`ExperimentDataset` + `experiment_dataloader`) behind the shared `Minibatch`
  contract; organism-aware (human + mouse), `obs`/`var` value-filterable, raw
  layer. Added deps `cellxgene-census`/`tiledbsoma`/`tiledbsoma-ml` and refreshed
  `uv.lock`; pinned `DEFAULT_CENSUS_VERSION = "2025-11-08"`. Registered a
  `network` pytest marker (skipped by default via `addopts`); offline glue tests
  at 100% coverage plus a live human+mouse streaming test (verified passing).
  `make ci` green. **PR #2 is now complete.**
- **2026-06-18** — PR #2 slice 1: organism-aware **local** data layer. Added
  `data/normalize.py` (size factors + internal log1p/depth normalization),
  `data/dataset.py` (`GeneVocabulary`, `align_to_reference`, `Minibatch`,
  `CountsDataset`, `collate_minibatch`), `data/anndata_io.py`
  (`.h5ad`/`.zarr` loaders + `build_anndata_dataloader`). Human *and* mouse
  iterate the same DataLoader API. Standardized on `zarr>=3` + `anndata>=0.12`
  and refreshed `uv.lock`. 100% coverage, fully offline, `make ci` green.
  Census streaming deferred to slice 2.
- **2026-06-17** — PR #5 merged: project pivot (Census streaming, raw-count
  NB/ZINB, discrete latent), cleanup (removed template/system_monitor code),
  strict mypy re-enabled. Added CLAUDE.md + this STATUS.md; clarified
  multi-organism support and v1-unconditional decision.
