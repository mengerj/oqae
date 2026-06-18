# OQAE — Live Status & Handoff

> Single source of truth for "where are we and what's next." **Update this at the
> end of every working session** so the next session can pick up cold. Keep it
> short; deep rationale lives in `docs/PROJECT_PLAN.md`.

**Last updated:** 2026-06-18

## Current state

- **PR #1 (Foundation) — DONE and merged to `main`.**
  - Logging (`src/omvqvae/utils/logging.py`), uv dev workflow, CI/CD
    (black/isort/flake8/**mypy strict**/pytest, bandit/safety), pre-commit.
  - Template leftovers and the old `system_monitor` utility have been removed.
  - `pyproject.toml` core deps trimmed; `wandb` kept for tracking.
  - `make ci` is green; pytest at 100% coverage on the current (small) codebase.
- **PR #2 (data layer) — slice 1 of 2 DONE (this PR), Census slice next.**
  - `data/normalize.py` — size factors + internal log1p/depth normalization.
  - `data/dataset.py` — organism-aware `GeneVocabulary`, `align_to_reference`
    (zero-fill missing / drop extra / warn on low overlap), the shared
    `Minibatch` `(counts, size_factors, covariates)` contract, `CountsDataset`,
    `collate_minibatch`.
  - `data/anndata_io.py` — local `.h5ad` / `.zarr` loaders +
    `build_anndata_dataloader` producing the shared `Minibatch` contract.
  - Human *and* mouse AnnData iterate through the *same* DataLoader API.
  - Requires `zarr>=3.0.0` + `anndata>=0.12.0` (zarr-python v3 support);
    `uv.lock` refreshed. Tests are 100%-covered and fully offline. `make ci`
    green.
- Plan/docs reflect the pivot: Census streaming, raw-count NB/ZINB modeling,
  discrete universal latent space, human+mouse, v1 unconditional, W&B monitoring.

## Next task — PR #2 slice 2: Census streaming

Finish the data layer by adding the CELLxGENE Census source behind the *same*
`Minibatch` contract already established in slice 1.

1. Add deps and refresh the lockfile: `cellxgene-census`, `tiledbsoma`,
   `tiledbsoma-ml` (`uv lock`). Confirm the newest stable Census version and pin
   it (latest known stable was `2025-01-30`).
2. `data/census.py` — stream raw counts from the Census via
   `cellxgene_census` + `tiledbsoma` + `tiledbsoma_ml`
   (`ExperimentDataset` + `experiment_dataloader()`), with `organism`
   selecting `homo_sapiens` / `mus_musculus`, `obs_query` filtering, raw layer.
   Reuse `GeneVocabulary` / `align_to_reference` and emit `Minibatch` (build the
   per-organism reference gene set from the Census `var` index).
3. Tests: tiny **offline** synthetic fixtures for the alignment/contract glue;
   mark the live-Census streaming test skippable (network-gated). Keep `make ci`
   green (strict mypy, coverage).

**Definition of done:** stream a small Census slice (human *and* mouse) through
the same `build_*` DataLoader API as local AnnData; live test marked skippable;
CI green; PR opened.

> Note: this run deliberately split PR #2. Slice 1 (local AnnData + alignment +
> normalization + contract) is fully offline-testable and lands here; slice 2
> (Census streaming) carries the heavy TileDB-SOMA deps and a network-gated test
> and is the next chunk — see PROJECT_PLAN.md decisions log.

## Open questions / parked

- **Cross-organism unification** (single shared latent across human+mouse):
  deferred past v1; needs ortholog mapping or shared gene embedding.
- **NB vs ZINB vs log-normalized+Gaussian** for the quantized setting: support
  both; benchmark in PR #9.
- **Batch-effect conditioning**: v1 unconditional; benchmark whether it's needed
  (PR #9) before adding.
- **Census version pin**: latest stable is `2025-01-30`; confirm the newest
  stable at implementation time and pin it.

## Changelog (most recent first)

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
