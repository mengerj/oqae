# OQAE — Live Status & Handoff

> Single source of truth for "where are we and what's next." **Update this at the
> end of every working session** so the next session can pick up cold. Keep it
> short; deep rationale lives in `docs/PROJECT_PLAN.md`.

**Last updated:** 2026-06-20

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
- **PR #3 (residual VQ layer) — DONE (this PR).**
  - `layers/residual_vq.py` — `VectorQuantizer` (single codebook: straight-through
    estimator, commitment + codebook losses, optional EMA codebook updates with
    Laplace smoothing, dead-code reset) and `ResidualVQ` (stacks `n_codebooks`
    quantizers over successive residuals; per-level indices = the cell's discrete
    code; summed quantized vectors approximate the input).
  - Per-forward monitoring metrics exposed via `QuantizerOutput` /
    `ResidualVQOutput` dataclasses: codebook **perplexity** (per-level + mean)
    and **utilization**, plus split commitment/codebook/total losses — ready for
    PR #5's W&B logging and PR #9's collapse checks.
  - Offline synthetic tests (`tests/layers/test_residual_vq.py`): shapes/dtypes,
    leading-dim preservation, straight-through gradient flow (identity gradient),
    EMA-buffer vs trained-parameter codebook, EMA update moves the codebook,
    dead-code reset (active + no-op paths), non-trivial perplexity, residual-norm
    reduction with more levels, and input-validation errors. 100% coverage on the
    new module; `make ci` green. No new dependencies (torch already present).
- Plan/docs reflect the pivot: Census streaming, raw-count NB/ZINB modeling,
  discrete universal latent space, human+mouse, v1 unconditional, W&B monitoring.

## Next task — PR #4: VQ-VAE core model (raw-count, NB/ZINB)

With the data layer and the discrete bottleneck in place, assemble the model.

1. `models/likelihoods.py` — NB (default) and ZINB reconstruction heads (and a
   Gaussian-on-log1p alternative), each mapping decoder outputs (+ observed size
   factor) to a likelihood and a negative-log-likelihood loss over genes.
2. `models/vqvae.py` — encoder → `ResidualVQ` → decoder. Encoder applies the
   internal log1p (see `data/normalize.py`); decoder consumes quantized codes +
   size factor and emits likelihood params. Compose the loss = reconstruction NLL
   + VQ loss; surface codebook perplexity/utilization for logging.
3. Tests (offline, synthetic): a 2-epoch smoke train on synthetic counts where
   the loss decreases; NB and ZINB heads both train; codebooks stay utilized
   (non-trivial perplexity). Keep `make ci` green (strict mypy, coverage).

**Definition of done:** a VQ-VAE that ingests raw counts, quantizes via
`ResidualVQ`, reconstructs counts under NB/ZINB, trains on a synthetic smoke
test; offline tests; CI green; PR opened.

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

- **2026-06-20** — PR #3: residual vector-quantizer layer. Added
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
