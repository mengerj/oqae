# OQAE — Live Status & Handoff

> Single source of truth for "where are we and what's next." **Update this at the
> end of every working session** so the next session can pick up cold. Keep it
> short; deep rationale lives in `docs/PROJECT_PLAN.md`.

**Last updated:** 2026-07-01

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
- **PR #5 (training/fine-tuning CLI + W&B) — DONE (both slices). PR #5 complete.**
  - Slice 2 (this run): `train/cli.py` — a config-driven entry point on top of
    the loop. An OmegaConf **structured config** (`ExperimentConfig` +
    `ModelConfig`/`DataConfig`/`TrainingConfig`/`TrackingConfig` dataclasses)
    validates a YAML, rejects unknown keys, and takes `--set a.b=c` overrides.
    Pure builders map each section to an object (`build_model`,
    `build_train_config`, `build_tracker_from_config`, `build_data`);
    `run_experiment` wires them and calls `omvqvae.train.train`. The
    local-`.h5ad`/`.zarr` source derives its `GeneVocabulary` (and the model's
    `n_genes`) from the file's own genes; the Census source is the one networked
    branch (`# pragma: no cover`). Optional checkpoint persists
    `{state_dict, organism, gene_ids, config}`. A single-command typer app is
    exposed as the `oqae-train` console script (added to `pyproject.toml`).
    Shipped `configs/train_toy.yaml`. All CLI deps
    (`omegaconf`/`typer`/`rich`/`wandb`) were already declared → **no `uv.lock`
    change**. Offline tests cover config loading, each builder, `run_experiment`
    wiring against a synthetic `.h5ad`, and the Typer `CliRunner` end-to-end;
    `make ci` green (~99% coverage).
- **PR #5 slice 1 (tracker + training loop) — DONE.**
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

- **PR #6 (HuggingFace Hub integration) — DONE (this PR). PR #6 complete.**
  - `hf_utils.py` (top-level, matching the PROJECT_PLAN package diagram) —
    `save_pretrained(model, vocabulary, dir, *, experiment_config=None)` and
    `load_pretrained(dir)` round-trip a trained `OmicsVQVAE` (state dict +
    codebooks), its architecture hyper-parameters, and the gene vocabulary
    (`organism`, ordered `gene_ids`) through a HuggingFace-style directory
    (`config.json` + `pytorch_model.bin`). Loading rebuilds the exact
    model/feature space (validated) and returns a `LoadedModel`
    (`model`/`vocabulary`/`experiment_config`).
  - The model is now **self-describing**: `OmicsVQVAE.get_config()` /
    `OmicsVQVAE.from_config()` capture/restore every constructor hyper-parameter
    (`get_config` is the `model` block of `config.json`). `from_config` ignores
    unknown keys (forward-compatible) and requires `n_genes`.
  - `from_checkpoint(ckpt, dir)` bridges a CLI checkpoint
    (`{state_dict, organism, gene_ids, config}`) into the same directory format,
    so a CLI-trained checkpoint and an HF-pushed model share one on-disk shape.
  - `push_to_hub` / `from_pretrained` are thin `huggingface_hub` shells
    (`# pragma: no cover`): they reuse the tested pure save/load step and only
    add the networked upload/download. Added direct dep `huggingface-hub>=0.20.0`
    (already present transitively via `transformers`); `uv.lock` refreshed.
  - Offline tests at 100% coverage on `hf_utils.py` (save/load round-trip incl.
    a trained model preserving codebooks, `experiment_config` round-trip,
    eval-mode, every validation/error path, and the `from_checkpoint` bridge)
    plus `get_config`/`from_config` tests on the model; `make ci` green (~99.7%).

- **PR #7 (discrete-code inference API) — DONE (this PR). PR #7 complete.**
  - `inference/codes.py` — the user-facing latent interface on top of a trained
    `OmicsVQVAE`. `encode(model, counts, ...) → EncodedCells` turns raw counts
    (tensor / ndarray / sparse) into discrete codes; `encode_anndata(loaded,
    adata, ...)` aligns a local AnnData to the model's `GeneVocabulary`
    (`align_to_reference`) first. `decode(model, codes_or_bundle, size_factors)`
    maps codes back to **expected counts**; `decode_to_params` exposes the full
    head distribution (for sampling/generation). All run the model in `eval` +
    `no_grad` (EMA codebooks untouched) and restore its prior train/eval mode;
    `encode`/`decode` are batched.
  - **Code-vector format**: `EncodedCells.codes` is an `int64`
    `(n_cells, n_codebooks)` tensor — row = a cell's discrete code, column `j` =
    the index chosen from residual codebook `j` (in `[0, codebook_size)`).
    Decoding also needs a per-cell `size_factor` (observed depth), which `encode`
    returns alongside the codes; `decode` reuses the bundle's factors unless
    overridden. `EncodedCells` also carries the continuous pre-quantization
    `latent` for quantization-error inspection.
  - **Inverse path added to the layer/model**: `VectorQuantizer.lookup` /
    `ResidualVQ.lookup` (indices → summed quantized vector) and
    `OmicsVQVAE.decode_codes` / `codes_to_params` (codes → expected counts /
    head params), so codes → quantized vectors → `expected_counts` is a tested
    model method rather than reaching into internals.
  - No new deps; `uv.lock` unchanged. Offline tests at 100% coverage on
    `inference/codes.py` (round-trip on held-out synthetic cells, batched ==
    unbatched, numpy/sparse inputs, AnnData alignment incl. reordered/missing/
    extra genes, size-factor scaling, eval-mode side-effect freedom, every
    validation/error path) plus `lookup`/`decode_codes` tests; `make ci` green
    (~99.7%).

- **PR #8 slice 1 (example scripts) — DONE (this PR).**
  - `examples/` — three runnable, self-contained scripts plus a `README.md`
    overview and a shared `synthetic_data.py` helper (a tiny raw-count AnnData
    with latent "programs", offline). `01_train_local_anndata.py` trains an
    `OmicsVQVAE` on a local AnnData (derive `GeneVocabulary` → `train` →
    `save_pretrained`); `02_inspect_and_generate_codes.py` walks the full
    `omvqvae.inference` API (`encode_anndata` → inspect codebook usage /
    program structure → `decode` → `decode_to_params` → generate a novel profile
    from edited codes, round-tripping through `save_pretrained`/`load_pretrained`);
    `03_census_streaming.py` streams a Census slice (network-gated, same
    `Minibatch` contract as example 1).
  - `tests/test_examples.py` smoke-tests the offline examples end to end (each
    `main()` runs on tiny synthetic data via `importlib`) and imports the Census
    example to catch regressions (its `main` runs only under `@pytest.mark.network`).
    Examples live under `examples/` (outside the `src` coverage scope), so they
    don't affect the coverage gate; `make ci` green (~99.7%). No new deps.

- **PR #8 slice 2 (Sphinx documentation) — DONE (this PR). PR #8 complete.**
  - `docs/source/` — a Sphinx project (`conf.py` with `autodoc` + `napoleon`
    for the NumPy docstrings, `viewcode`, `intersphinx`; `furo` theme). `index.rst`
    is the landing page (project overview + highlights + toctree);
    `getting_started.rst` is a narrative walk-through (install → train local /
    Census → encode/decode → save/share) linking the `examples/` scripts on
    GitHub; `api.rst` autodocs the full public API by implementation module
    (`data.dataset`/`normalize`/`anndata_io`/`census`, `layers.residual_vq`,
    `models.likelihoods`/`vqvae`, `train.loop`/`cli`, `inference.codes`,
    `hf_utils`, `utils.tracking`).
  - Wired a `make docs` (and `docs-clean`) target running `sphinx-build -W
    --keep-going` (warnings-as-errors) into `docs/_build/html` (already
    gitignored). Added a `docs` extra (`sphinx>=7`, `furo`) and refreshed
    `uv.lock`. Added a **`docs` CI job** that builds the docs warnings-as-errors
    so they stay buildable (the optional CI-docs decision — done now rather than
    deferred to PR #10).
  - Fixed two module docstrings (`models/vqvae.py`, `inference/codes.py`) whose
    Markdown ```` ``` ```` code fences aren't valid reStructuredText → converted
    to RST `.. code-block:: text` literal blocks so the build is warning-clean.
    `make ci` green (~99.7%); `make docs` clean.

- **PR #9 slice 1 (benchmarking harness) — DONE (this PR).**
  - `src/omvqvae/benchmark/` — the offline metrics/reporting scaffold for PR #9.
    `metrics.py` holds dependency-light pure functions: `codebook_usage`
    (dataset-level per-level **perplexity** + **utilization** = the collapse
    signal), `separability_score` (nearest-centroid resubstitution accuracy of
    the latent vs known labels = downstream-separability proxy), and
    `reconstruction_metrics` (mean per-cell **NLL** + expected-vs-target **MAE**
    in the head's native space, run in `eval`+`no_grad` so the EMA codebooks are
    untouched). `harness.py` adds `BenchmarkConfig` / `BenchmarkResult`,
    `evaluate_model`, `run_benchmark` / `run_suite` (build a model from a config
    → `omvqvae.train.train` over an injected re-iterable `Minibatch` source →
    `evaluate_model` on held-out counts/labels; seeded for reproducibility), and
    `format_results_table` / `results_to_dicts` (Markdown comparison table / CSV
    rows). Reuses `omvqvae.inference.encode` for the codes+latent and
    `OmicsVQVAE.decode_codes` for the MAE.
  - `examples/04_benchmark_configs.py` runs a tiny NB-vs-Gaussian + codebook-
    capacity sweep on synthetic data and prints the comparison table (offline,
    smoke-tested). Wired into `examples/README.md` and the Sphinx `api.rst`.
  - No new deps; `uv.lock` unchanged. Offline tests at 100% on the benchmark
    package (pure-metric edge cases, eval-mode side-effect freedom, harness
    train+evaluate, reproducibility, suite + reporting); `make ci` green
    (~99.6%).

- **PR #9 slice 2 (empirical sweeps + report) — DONE (this PR).**
  - `src/omvqvae/benchmark/report.py` — turns the slice-1 scaffold into a
    reproducible **benchmark report**. `make_benchmark_fixture` builds a larger
    pure-NumPy synthetic raw-count fixture (latent "programs", train/eval split,
    a re-iterable `Minibatch` `DataLoader` train source — no AnnData dep);
    `default_report_configs` is the fuller grid (NB vs ZINB vs **Gaussian** at a
    shared `2x64` anchor, plus `codebook_size` 16/64/256 and `n_codebooks` 1/2/4
    sweeps); `generate_report` runs `run_suite` over the fixture and renders a
    Markdown report with an **auto-generated interpretation** (best NB
    reconstruction, codebook collapse check at <50% utilization, separability
    ranking, NB-vs-Gaussian on the comparable separability axis, batch-effect
    note). Exported from `omvqvae.benchmark`.
  - `examples/05_benchmark_report.py` regenerates the committed
    `docs/benchmark_report.md` (offline, ~seconds). Wired into `examples/README.md`
    and Sphinx `api.rst`. The report's empirical read on this synthetic fixture:
    NB/ZINB cleanly recover the program structure (separability ~1.0) while the
    log-normalized Gaussian and ZINB lag on separability (~0.45–0.49); larger
    codebooks (`2x256`, `1x64`) under-utilize at this data scale — so **NB stays
    the v1 default** and codebook capacity should track data scale.
  - No new deps; `uv.lock` unchanged. Offline tests at ~100% on `report.py`
    (fixture split/re-iterability/degenerate-split guard, the config grid, the
    interpretation branches incl. collapse / Gaussian-wins / nan-separability,
    and an end-to-end `generate_report`); example 5 smoke-tested. `make ci` green
    (~99.8%); `make docs` clean.

- **PR #9 slice 3 (Census streaming throughput/scaling) — DONE (this PR). PR #9 complete.**
  - `src/omvqvae/benchmark/throughput.py` — the streaming/scaling benchmark for
    the data path. `measure_stream_throughput(source, *, max_batches, max_cells,
    warmup_batches, step_fn, clock, label)` is the **pure timing core**: it
    iterates any re-iterable of `Minibatch` (a local `DataLoader` or a
    `CensusMinibatchLoader`), optionally applies a per-batch `step_fn`, and
    returns a `ThroughputResult` (cells/s, batches/s, seconds/batch,
    time-to-first-batch). Leading `warmup_batches` are processed but excluded
    from the steady-state window (and the clock only starts once they finish), so
    a cold start doesn't depress the rates; an injectable `clock` makes the whole
    thing deterministic offline. `make_train_step_fn(model, *, optimizer,
    grad_clip_norm, lr, device)` builds a single-optimizer-step closure (mirrors
    the `train` inner loop) so end-to-end streaming-plus-training throughput is
    measurable; it reuses the `BenchmarkConfig` model contract (build a model
    from `config.model_kwargs()`, pass `config.lr`/`config.grad_clip_norm`).
    `throughput_to_dicts` / `format_throughput_table` mirror the slice-1
    reporting helpers.
  - `benchmark_census_throughput(organism, *, config, census_version,
    obs/var_value_filter, batch_size, max_batches, warmup_batches, ...)` is the
    one networked shell (`# pragma: no cover`): it opens a pinned Census, builds
    `build_census_dataloader`, and feeds it to the offline timing core (raw
    streaming when `config is None`, end-to-end when a `BenchmarkConfig` is
    given). `examples/06_census_throughput.py` profiles raw vs end-to-end human
    streaming (network-gated, imported in CI; `main` runs under
    `@pytest.mark.network`).
  - No new deps; `uv.lock` unchanged. Offline tests at 100% on `throughput.py`
    (fake-clock rate maths, warmup exclusion, `max_batches`/`max_cells` stops,
    empty/all-warmup edge cases, every validation path, `step_fn` application +
    timing, `make_train_step_fn` updating the model / honoring an injected
    optimizer+clip, the reporting helpers, and `BenchmarkConfig` → step-fn
    round-trip) plus the example import smoke test; `make ci` green (~99.8%),
    `make docs` clean. **PR #9 is now complete.**
- **Benchmark eval upgrades (issue #36) — DONE (this PR, branch
  `claude/eval-upgrades`).** Adds offline latent-quality metrics so we can judge
  a model beyond nearest-centroid separability. Three pieces:
  - **Quantization-cost view.** `inference.encode` / `EncodedCells` now also
    return `quantized` (the post-quantization latent, i.e. the codes embedded
    back into latent space; free — same `model.quantize(z)` call). `evaluate_model`
    reports `separability` (continuous `z`), `separability_quantized`, and their
    `separability_gap` — how much biology the RVQ bottleneck discards. Surfaced in
    `results_to_dicts` and the Markdown table (`sep_quant` / `sep_gap` columns).
  - **scIB clustering metrics.** `benchmark/clustering.py::clustering_metrics`
    (NMI / ARI via KMeans + cell-type ASW) using the new **optional `benchmark`
    extra** (`scib-metrics`, `umap-learn`, `matplotlib`; `uv.lock` refreshed).
    Lazy-imported; opt-in via `evaluate_model(..., compute_clustering=True)` /
    `run_benchmark` / `run_suite`. Off by default so the core path never imports
    jax/scib. The `nmi` / `ari` / `ct_asw` table columns appear only when
    computed.
  - **UMAP viz.** `benchmark/viz.py::plot_latent_umap` (+ `compute_umap`) —
    grid of UMAP scatters, rows = `latent` / `quantized`, columns = `labels` /
    `color_by` (batch view). Works on the continuous representations; raw integer
    codes are intentionally not UMAP'd (Euclidean on codebook indices is
    meaningless).
  - New exports on `omvqvae.benchmark`. `examples/latent_quality.py` demonstrates
    `compute_clustering=True` + a saved UMAP. Tests: clustering/viz guarded with
    `importorskip` (run in CI since all extras are installed); Part A tests are
    unconditional. `make ci` green (249 passed, ~99.6%).
  - **Follow-up (issue #37):** scVI baseline behind a `LatentModel` protocol,
    reusing these shared latent metrics.

## Next task — PR #10: v1.0 release

PR #9 (benchmarking & scaling) is **done** across all three slices
(metrics/reporting scaffold, empirical sweeps + report, Census streaming
throughput). The next roadmap item is **PR #10 — v1.0 release** (see
`docs/PROJECT_PLAN.md` → Phase 3):

- **API freeze**: confirm the public surface re-exported from each subpackage
  `__init__` is the intended v1 API; tidy any stragglers.
- **Final docs**: a top-level `README.md` pass (badges, quickstart, the
  encode→codes→decode story) and a docs review (the Sphinx site already builds
  warnings-as-errors in CI).
- **Packaging / PyPI**: verify `pyproject.toml` metadata (classifiers, URLs,
  `version = "1.0.0"`), build the sdist/wheel (`uv build`), and document the
  release flow. Leave the actual PyPI upload (a networked, credentialed step)
  for a human.

This is naturally several slices — a good first chunk is the **README + API
freeze audit** (offline, no new deps), with packaging/PyPI as a follow-up.

### Building blocks already in place

- `omvqvae.benchmark.{run_suite, run_benchmark, BenchmarkConfig,
  format_results_table, results_to_dicts}` — the harness + reporting from
  slice 1.
- `omvqvae.benchmark.{make_benchmark_fixture, default_report_configs,
  generate_report}` — the offline fixture + sweep + report from slice 2;
  `generate_report` regenerates against real data by swapping the fixture's
  training source.
- `omvqvae.benchmark.{measure_stream_throughput, make_train_step_fn,
  benchmark_census_throughput, format_throughput_table}` — the slice-3 streaming
  throughput benchmark (pure core + networked Census shell).
- `omvqvae.train.train` / `TrainConfig` — the source-agnostic loop;
  `build_census_dataloader` provides the streamed `Minibatch` source.

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

- **2026-06-30** — PR #9 slice 3: Census streaming throughput/scaling. Added
  `src/omvqvae/benchmark/throughput.py` — `measure_stream_throughput` (the pure,
  offline-tested timing core over any `Minibatch` stream: cells/s, batches/s,
  time-to-first-batch, with warmup batches excluded from the steady-state window
  and an injectable clock), `make_train_step_fn` (a single-optimizer-step closure
  mirroring the `train` inner loop so end-to-end streaming-plus-training
  throughput is measurable, reusing the `BenchmarkConfig` model contract),
  `throughput_to_dicts` / `format_throughput_table` (CSV rows / Markdown table),
  and `benchmark_census_throughput` (the one networked `# pragma: no cover` shell
  that streams a live Census slice through the offline core). Added
  `examples/06_census_throughput.py` (raw vs end-to-end human streaming profile,
  network-gated, imported in CI). Wired exports into `omvqvae.benchmark` and the
  Sphinx `api.rst`; updated `examples/README.md`. No new deps; `uv.lock`
  unchanged. Offline tests at 100% on `throughput.py`; `make ci` green (~99.8%),
  `make docs` clean. **PR #9 is now complete.** Next is PR #10 (v1.0 release).
- **2026-06-30** — PR #9 slice 2: empirical sweeps + benchmark report. Added
  `src/omvqvae/benchmark/report.py` — `make_benchmark_fixture` (larger
  pure-NumPy synthetic raw-count fixture with latent programs, train/eval split,
  a re-iterable `Minibatch` `DataLoader` train source), `default_report_configs`
  (NB vs ZINB vs Gaussian at a shared `2x64` anchor + `codebook_size` 16/64/256
  and `n_codebooks` 1/2/4 sweeps), and `generate_report` (runs `run_suite` and
  renders a Markdown report with an auto-generated interpretation:
  within-likelihood reconstruction, codebook collapse check, separability
  ranking, NB-vs-Gaussian, batch-effect note). Exported from `omvqvae.benchmark`.
  Added `examples/05_benchmark_report.py` (regenerates the committed
  `docs/benchmark_report.md` offline) and wired it into `examples/README.md` +
  Sphinx `api.rst`. Empirically on the synthetic fixture: NB/ZINB recover the
  programs (separability ~1.0) vs Gaussian/ZINB ~0.45–0.49; oversized codebooks
  under-utilize — **NB stays the v1 default**. No new deps; `uv.lock` unchanged.
  Offline tests at ~100% on `report.py`; `make ci` green (~99.8%), `make docs`
  clean. Slice 3 (Census streaming throughput) is next.
- **2026-06-29** — PR #9 slice 1: benchmarking harness. Added
  `src/omvqvae/benchmark/` — pure metrics (`codebook_usage` perplexity/
  utilization, `separability_score` nearest-centroid latent separability,
  `reconstruction_metrics` NLL/MAE evaluated in `eval`+`no_grad`) plus a thin
  harness (`BenchmarkConfig`/`BenchmarkResult`, `evaluate_model`,
  `run_benchmark`/`run_suite`, `format_results_table`/`results_to_dicts`) that
  trains tiny models under several likelihood/codebook configs over an injected
  re-iterable `Minibatch` source and emits a Markdown comparison table. Reuses
  `omvqvae.train.train`, `inference.encode`, and `OmicsVQVAE.decode_codes`.
  Added `examples/04_benchmark_configs.py` (offline NB-vs-Gaussian + codebook
  sweep, smoke-tested) and documented the module in `examples/README.md` +
  Sphinx `api.rst`. No new deps; `uv.lock` unchanged. Offline tests at 100% on
  the benchmark package; `make ci` green (~99.6%). This is the metrics/reporting
  scaffold; the empirical sweeps + report (slice 2) and Census throughput
  benchmark (slice 3) are next.
- **2026-06-28** — PR #8 slice 2: Sphinx documentation. Added a Sphinx project
  under `docs/source/` (`conf.py` with `autodoc` + `napoleon` for the NumPy
  docstrings, `viewcode`, `intersphinx`, `furo` theme): `index.rst` (landing
  page + toctree), `getting_started.rst` (narrative install → train → encode /
  decode → save walk-through linking the `examples/` scripts), and `api.rst`
  (autodoc of the full public API by implementation module). Wired `make docs` /
  `docs-clean` (`sphinx-build -W --keep-going` → `docs/_build/html`) and a `docs`
  CI job (warnings-as-errors). Added a `docs` extra (`sphinx>=7`, `furo`) and
  refreshed `uv.lock`. Converted the Markdown code fences in two module
  docstrings (`models/vqvae.py`, `inference/codes.py`) to RST literal blocks so
  the build is warning-clean. `make ci` green (~99.7%); `make docs` clean. **PR
  #8 is now complete.** Next is PR #9 (benchmarking & scaling).
- **2026-06-27** — PR #8 slice 1: example scripts. Added `examples/` — three
  runnable, self-contained scripts (`01_train_local_anndata.py`,
  `02_inspect_and_generate_codes.py`, `03_census_streaming.py`), a `README.md`
  overview, and a shared offline `synthetic_data.py` helper (tiny raw-count
  AnnData with latent "programs"). Example 1 trains on a local AnnData and
  `save_pretrained`s; example 2 walks the full `omvqvae.inference` API
  (encode → inspect codebook usage / program structure → decode →
  `decode_to_params` → generate from edited codes, via a
  `save_pretrained`/`load_pretrained` round-trip); example 3 streams a Census
  slice (network-gated). `tests/test_examples.py` smoke-tests the offline
  examples end to end (importlib-driven `main()` on tiny data) and imports the
  Census example (its `main` runs only under `@pytest.mark.network`). Examples
  sit outside the `src` coverage scope, so the gate is unaffected; no new deps;
  `make ci` green (~99.7%). Slice 2 (Sphinx docs) is the next chunk.
- **2026-06-26** — PR #7: discrete-code inference API. Added the
  `omvqvae.inference` package (`inference/codes.py`): `encode` /
  `encode_anndata` turn raw counts / a local AnnData (aligned to the model's
  `GeneVocabulary`) into an `EncodedCells` bundle (`codes`
  `(n_cells, n_codebooks)` int64 + per-cell `size_factors` + continuous
  `latent`); `decode` maps codes → expected counts and `decode_to_params`
  exposes the full head distribution. Inference runs in `eval` + `no_grad`
  (EMA codebooks untouched) with the model's prior mode restored, and is
  batched. Added the inverse path to the layer/model: `VectorQuantizer.lookup`
  / `ResidualVQ.lookup` (indices → summed quantized vector) and
  `OmicsVQVAE.decode_codes` / `codes_to_params`. No new deps; `uv.lock`
  unchanged. Offline tests at 100% coverage on `inference/codes.py`
  (held-out round-trip, batched == unbatched, numpy/sparse, AnnData alignment
  with reordered/missing/extra genes, size-factor scaling, eval-mode
  side-effect freedom, all error paths); `make ci` green (~99.7%). **PR #7 is
  now complete.**
- **2026-06-25** — PR #6: HuggingFace Hub integration. Added top-level
  `hf_utils.py` — `save_pretrained` / `load_pretrained` round-trip a trained
  `OmicsVQVAE` (state dict + codebooks), its architecture config, and the gene
  vocabulary (`organism`, `gene_ids`) through a HuggingFace-style directory
  (`config.json` + `pytorch_model.bin`); `load_pretrained` rebuilds the exact
  model/feature space and returns a `LoadedModel`. Made the model
  **self-describing** (`OmicsVQVAE.get_config()` / `from_config()`) so
  serialization is decoupled from the training CLI. `from_checkpoint` bridges a
  CLI checkpoint bundle into the same directory format; `push_to_hub` /
  `from_pretrained` are thin `huggingface_hub` shells (`# pragma: no cover`)
  reusing the tested pure save/load step. Added direct dep
  `huggingface-hub>=0.20.0` (already transitive via `transformers`); `uv.lock`
  refreshed. Offline tests at 100% coverage on `hf_utils.py` plus
  `get_config`/`from_config` model tests; `make ci` green (~99.7%). **PR #6 is
  now complete.**
- **2026-06-24** — PR #5 slice 2: config-driven training CLI. Added
  `train/cli.py` — an OmegaConf structured-config schema (`ExperimentConfig` +
  `ModelConfig`/`DataConfig`/`TrainingConfig`/`TrackingConfig`) and a
  single-command typer app exposed as the `oqae-train` console script. Pure
  builders (`build_model`, `build_train_config`, `build_tracker_from_config`,
  `build_data`) map config sections to objects; `run_experiment` wires them and
  calls `omvqvae.train.train`. Local-AnnData source derives its `GeneVocabulary`
  from the file's genes; the Census source is the one gated/`# pragma: no cover`
  branch. Optional checkpoint persists `{state_dict, organism, gene_ids,
  config}`. Shipped `configs/train_toy.yaml`; registered `[project.scripts]
  oqae-train`. All CLI deps were already declared → **`uv.lock` unchanged**.
  Offline tests cover config loading/validation, every builder, `run_experiment`
  against a synthetic `.h5ad`, and the Typer `CliRunner` end-to-end; `make ci`
  green (~99% coverage). **PR #5 is now complete.**
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
