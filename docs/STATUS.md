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
- **PR #2 (data layer) — DONE.** Both slices landed.
  - *Slice 1:* `data/normalize.py` (size factors + internal log1p/depth
    normalization), `data/dataset.py` (organism-aware `GeneVocabulary`,
    `align_to_reference`, the shared `Minibatch`
    `(counts, size_factors, covariates)` contract, `CountsDataset`,
    `collate_minibatch`), `data/anndata_io.py` (local `.h5ad` / `.zarr` loaders
    + `build_anndata_dataloader`).
  - *Slice 2 (this PR):* `data/census.py` — CELLxGENE Census streaming via
    `cellxgene_census` + `tiledbsoma` + `tiledbsoma_ml`
    (`ExperimentDataset` + `experiment_dataloader()`), `organism` selecting
    `homo_sapiens` / `mus_musculus`, `obs`/`var` value filters, raw layer. Reuses
    `GeneVocabulary` / `align_to_reference`, builds the per-organism reference
    from the Census `var` index, and yields the **same** `Minibatch` contract via
    `CensusMinibatchLoader`. Added deps `cellxgene-census` + `tiledbsoma` +
    `tiledbsoma-ml` (Census pinned to `2025-01-30`); `uv.lock` refreshed.
  - Human *and* mouse iterate through the *same* `build_*` DataLoader API for
    both local AnnData and Census.
  - Tests: alignment/contract glue + orchestration are fully offline (TileDB-SOMA
    stack faked); the live-Census streaming test is network-gated
    (`OQAE_RUN_CENSUS_TESTS=1`). `make ci` green, 99% coverage.
- Plan/docs reflect the pivot: Census streaming, raw-count NB/ZINB modeling,
  discrete universal latent space, human+mouse, v1 unconditional, W&B monitoring.

## Next task — PR #3: Residual Vector Quantizer layer

The data layer is complete; the next roadmap chunk is the residual VQ layer that
the model will sit on top of.

1. `layers/residual_vq.py` — a configurable residual vector quantizer with a
   straight-through estimator: configurable number of codebook levels (default
   2) and codebook size, commitment/codebook losses, and perplexity / codebook
   utilization metrics. Include an EMA-update / codebook-reset option to guard
   against codebook collapse.
2. Tests (offline, fast, synthetic): shape/round-trip of encode→indices→dequant,
   gradient flow through the straight-through estimator, commitment-loss
   behaviour, and a perplexity/utilization sanity check. Keep `make ci` green
   (strict mypy, coverage).

**Definition of done:** a residual VQ module that quantizes a batch of latent
vectors into per-level codebook indices and reconstructs them, exposes
commitment/codebook losses + perplexity, verified gradient flow; CI green; PR
opened.

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

- **2026-06-19** — PR #2 slice 2: CELLxGENE Census streaming. Added
  `data/census.py` (`census_batch_to_minibatch`, `gene_vocabulary_from_var`,
  `CensusMinibatchLoader`, `build_census_dataloader`) streaming raw counts via
  `cellxgene_census` + `tiledbsoma` + `tiledbsoma_ml` behind the shared
  `Minibatch` contract; `organism` selects the human/mouse experiment, reference
  genes come from the Census `var`. Added the three TileDB-SOMA deps (Census
  pinned `2025-01-30`) and refreshed `uv.lock`. Offline-faked orchestration tests
  + network-gated live test; 99% coverage; `make ci` green. **PR #2 complete.**
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
