# OQAE — Live Status & Handoff

> Single source of truth for "where are we and what's next." **Update this at the
> end of every working session** so the next session can pick up cold. Keep it
> short; deep rationale lives in `docs/PROJECT_PLAN.md`.

**Last updated:** 2026-06-19

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
- Plan/docs reflect the pivot: Census streaming, raw-count NB/ZINB modeling,
  discrete universal latent space, human+mouse, v1 unconditional, W&B monitoring.

## Next task — PR #3: Residual Vector Quantizer layer

With the data layer complete, build the discrete bottleneck.

1. `layers/residual_vq.py` — a configurable residual VQ module: `n_codebooks`
   (default 2), codebook size/dim, straight-through estimator for gradient flow,
   commitment + codebook losses, and EMA / dead-code reset option.
2. Expose per-forward metrics for monitoring: codebook **perplexity** /
   utilization and quantization loss, so PR #5's W&B logging and PR #9's
   collapse checks have them.
3. Tests (offline, synthetic): shapes/dtypes, gradient flow through the
   straight-through estimator, that codebooks are utilized (non-trivial
   perplexity), and that EMA/reset paths run. Keep `make ci` green (strict mypy,
   coverage).

**Definition of done:** a residual VQ layer that quantizes encoder outputs to a
set of codebook indices, returns quantized vectors + indices + losses + metrics,
back-props through the bottleneck; offline tests; CI green; PR opened.

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
