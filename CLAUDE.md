# CLAUDE.md — OQAE working guide for AI sessions

> Read this first every session. It is the orientation for daily, mostly-autonomous
> work on OQAE. For *what* to build next, read `docs/STATUS.md` (live state) and
> `docs/PROJECT_PLAN.md` (roadmap + design decisions).

## What this project is

**OQAE (Omics Quantized Auto Encoder)** learns a **discrete, universal latent
space for single-cell omics** with a residual VQ-VAE. Each scRNA-seq cell is
encoded as a set of discrete codes (codebook indices); the generative decoder
turns codes back into expression. We start with scRNA-seq because the CZ
CELLxGENE Census provides it at scale.

The Python package is `omvqvae` (under `src/omvqvae/`); the distribution name is
`oqae`.

## Non-negotiable design decisions (don't re-litigate — see the decisions log in PROJECT_PLAN.md)

- **Data**: stream from the CZ CELLxGENE Census via TileDB-SOMA
  (`cellxgene_census` + `tiledbsoma` + `tiledbsoma_ml`); also support local
  `.h5ad` / `.zarr` AnnData. Not Zarr/Dask-primary.
- **Input**: ingest **raw counts**; reconstruct with **NB / ZINB**; library size
  handled internally (scVI-style). Log-normalized/Gaussian is a pluggable
  alternative only.
- **Organisms**: support **human + mouse**, organism-aware with a per-organism
  gene space; **one model per organism** in v1.
- **v1 model is unconditional** (no batch/covariate input); revisit only if
  benchmarking shows batch effects hurt the latent space.
- **Monitoring = Weights & Biases** (offline-friendly). No bespoke system monitor.

## Where things live

- `docs/PROJECT_PLAN.md` — roadmap (PRs #1–#10), architecture, decisions log.
- `docs/STATUS.md` — current state, what was done last, the next concrete task.
  **Update this at the end of every working session.**
- `src/omvqvae/` — the package. Target structure is in PROJECT_PLAN.md; create
  modules as their PR lands (`data/`, `layers/`, `models/`, `train/`,
  `inference/`, `utils/`).
- `tests/` — pytest; mirror the package; keep tests fast and **offline by
  default** (mark live-Census/network tests as skippable).

## Development workflow (uv-based)

```bash
make setup-env        # uv sync --all-extras  (creates .venv)
make format           # autopep8 + black + isort (auto-fix)
make ci               # format-check + flake8 + mypy + pytest  ← must pass before committing
```

Individual gates: `make lint`, `make type-check` (mypy strict on `src/omvqvae`),
`make test` (pytest with coverage). CI runs the same checks on Python 3.11 and
3.12, plus a bandit/safety security job.

## Conventions

- **Strict typing**: mypy is enabled and CI-enforced. Fully annotate public APIs.
- **Docstrings**: NumPy style for public functions/classes.
- **Formatting**: black + isort (line length 88); flake8 clean.
- **Tests**: add tests with every code change; keep coverage high; default to
  tiny synthetic fixtures so the suite runs offline and fast.
- **Dependencies**: add a new dep only in the PR that first uses it, and refresh
  `uv.lock` (`uv lock`) in the same change.

## Code patterns / idioms

The data layer established these structural conventions; keep following them as
new packages (`layers/`, `models/`, `train/`, …) land.

- **Inject data sources** into loaders/components (pass the iterable or handle
  in) so tests can substitute a plain list — don't construct the source inside
  the class. This is why `CensusMinibatchLoader` is testable offline.
- **Networked/heavy logic = pure core + thin I/O shell.** Keep the real work in
  pure-Python functions tested offline with synthetic fixtures; mark the live
  shell `@pytest.mark.network` (skipped by default) and `# pragma: no cover`.
- **Lazy-import heavy deps** (`torch`, `cellxgene_census`, `anndata`, …) inside
  the function that uses them; put type-only imports under `if TYPE_CHECKING`.
  Keeps `import omvqvae` fast and optional deps optional.
- **`@dataclass` for plain data bundles; a validating class for anything with
  invariants** (raise in `__init__`). `Minibatch` is a dataclass; `GeneVocabulary`
  is a hand-written class that rejects empty/duplicate gene ids.

## Git / PR workflow

- Each automated run works on a fresh branch off the latest `main`
  (e.g. `claude/<topic>`), scoped to roughly **one roadmap PR** (or one coherent
  sub-task) — don't sprawl across multiple PRs in a single run.
- Run `make ci` green, commit with a clear message, push, and **open a PR**.
- Leave merging to a human reviewer unless explicitly told otherwise.
- Co-author trailer on commits:
  `Co-Authored-By: Claude <noreply@anthropic.com>`.

## When to stop and ask vs. proceed

- **Proceed** on well-scoped, decided work (the next task in `docs/STATUS.md`,
  bug fixes, tests, docs).
- **Ask first** (via a notification / leave it for the human) when a choice is
  architecturally significant, contradicts a decision above, requires a new
  heavy dependency not already planned, or is ambiguous. Capture the open
  question in `docs/STATUS.md` so the next session sees it.
