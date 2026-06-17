# OQAE — Live Status & Handoff

> Single source of truth for "where are we and what's next." **Update this at the
> end of every working session** so the next session can pick up cold. Keep it
> short; deep rationale lives in `docs/PROJECT_PLAN.md`.

**Last updated:** 2026-06-17

## Current state

- **PR #1 (Foundation) — DONE and merged to `main`.**
  - Logging (`src/omvqvae/utils/logging.py`), uv dev workflow, CI/CD
    (black/isort/flake8/**mypy strict**/pytest, bandit/safety), pre-commit.
  - Template leftovers and the old `system_monitor` utility have been removed.
  - `pyproject.toml` core deps trimmed; `wandb` kept for tracking.
  - `make ci` is green; pytest at 100% coverage on the current (small) codebase.
- Plan/docs reflect the pivot: Census streaming, raw-count NB/ZINB modeling,
  discrete universal latent space, human+mouse, v1 unconditional, W&B monitoring.

## Next task — PR #2: organism-aware data layer

Implement the unified data interface (see PROJECT_PLAN.md → "PR #2" for full
scope and exit criteria). In short:

1. Add deps and refresh the lockfile: `cellxgene-census`, `tiledbsoma`,
   `tiledbsoma-ml` (`uv lock`).
2. `data/census.py` — stream raw counts from the Census via
   `cellxgene_census` + `tiledbsoma` + `tiledbsoma_ml`
   (`ExperimentDataset` + `experiment_dataloader()`), with `organism`
   selecting `homo_sapiens` / `mus_musculus`, `obs_query` filtering, raw layer.
3. `data/anndata_io.py` — load local `.h5ad` / `.zarr` (chunked/backed).
4. `data/dataset.py` — per-organism reference gene set + alignment
   (zero-fill missing genes, drop extras, warn on low overlap); common
   `(raw_counts, covariates)` minibatch contract (covariates carry organism +
   batch/dataset id, unused by v1 model).
5. `data/normalize.py` — size factors + internal-normalization helpers.
6. Tests: tiny **offline** synthetic fixtures; mark live-Census tests skippable.
   Keep `make ci` green (strict mypy, coverage).

**Definition of done:** iterate a local AnnData and a small Census slice (human
*and* mouse) through the *same* DataLoader API; CI green; PR opened.

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

- **2026-06-17** — PR #5 merged: project pivot (Census streaming, raw-count
  NB/ZINB, discrete latent), cleanup (removed template/system_monitor code),
  strict mypy re-enabled. Added CLAUDE.md + this STATUS.md; clarified
  multi-organism support and v1-unconditional decision.
