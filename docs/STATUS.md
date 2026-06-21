# OQAE — Live Status & Handoff

> Single source of truth for "where are we and what's next." **Update this at the
> end of every working session** so the next session can pick up cold. Keep it
> short; deep rationale lives in `docs/PROJECT_PLAN.md`.

**Last updated:** 2026-06-21

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
- **PR #3 (residual VQ layer) — IN FLIGHT (open PR #24, not yet merged).**
  - `layers/residual_vq.py` implements `VectorQuantizer` + `ResidualVQ`
    (straight-through estimator, commitment/codebook losses, EMA, dead-code
    reset, perplexity/utilization metrics) on branch
    `claude/wonderful-euler-k7rfi7`. Awaiting human review/merge.
- **PR #4 slice 1 (reconstruction likelihoods) — DONE (this PR).**
  - `models/likelihoods.py`: pure log-prob functions `log_nb_positive`,
    `log_zinb_positive`, `log_gaussian` (SciPy-checked) plus the decoder heads
    `NBHead`, `ZINBHead`, `GaussianHead` behind one `ReconstructionHead`
    interface (`forward` → params, `reconstruction_loss`, `expected_counts`)
    and a `build_reconstruction_head` factory. scVI-style count parameterization
    (softmax `px_scale` × library size → NB mean; gene-wise dispersion). No new
    deps. Offline tests at 100% coverage; `make ci` green. Independent of the
    VQ layer, so it does not touch/conflict with PR #24.
- Plan/docs reflect the pivot: Census streaming, raw-count NB/ZINB modeling,
  discrete universal latent space, human+mouse, v1 unconditional, W&B monitoring.

## Next task — PR #4 slice 2: encoder/decoder VQ-VAE core

The reconstruction heads (slice 1, this PR) and the residual VQ layer (PR #24)
are the two halves of the bottleneck + output. Slice 2 wires them into the model.

1. `models/vqvae.py` — an `nn.Module` encoder (raw counts → internal log1p →
   hidden), the `ResidualVQ` bottleneck (from PR #24), and a decoder that feeds a
   `ReconstructionHead` (from this PR). Compose recon + VQ losses; expose codes
   (per-level indices) and codebook metrics.
2. Tests (offline, synthetic): a 2-epoch smoke train on synthetic counts for the
   NB and ZINB heads; loss decreases; codebooks utilized (non-trivial
   perplexity); shapes/round-trip. Keep `make ci` green.

**Dependency note:** slice 2 imports `ResidualVQ`, so it should land *after*
PR #24 merges (or be branched off it). Until then it is blocked on review of
PR #24 — flag for the human rather than duplicating the VQ layer.

**Definition of done:** an end-to-end VQ-VAE that encodes raw counts to discrete
codes and reconstructs counts via NB/ZINB; a synthetic smoke-train converges;
offline tests; CI green; PR opened.

## Open questions / parked

- **PR ordering vs. PR #24**: PR #3 (residual VQ) is open but unmerged as PR #24.
  This run did the *independent* PR #4 slice (likelihoods) to avoid conflicting
  with it. The VQ-VAE core (PR #4 slice 2) depends on PR #24's `ResidualVQ` —
  merge PR #24 first, or branch slice 2 off it. **Decision for the human:** merge
  order of PR #24 then the next VQ-VAE PR.
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

- **2026-06-21** — PR #4 slice 1: reconstruction likelihoods. Added
  `models/likelihoods.py` — pure log-prob functions (`log_nb_positive`,
  `log_zinb_positive`, `log_gaussian`, validated against SciPy) and decoder heads
  (`NBHead`, `ZINBHead`, `GaussianHead`) under a shared `ReconstructionHead`
  interface plus a `build_reconstruction_head` factory. scVI-style count
  parameterization (softmax proportions × library size → NB mean; gene-wise
  dispersion); Gaussian head for the log-normalized alternative. Exported from
  `models/__init__.py`. No new deps (`torch`/`scipy` already present);
  `uv.lock` unchanged. Offline tests at 100% coverage; `make ci` green. Chosen as
  an independent slice because PR #3's residual VQ is already in flight as the
  unmerged **PR #24** — this slice does not touch the VQ layer.
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
