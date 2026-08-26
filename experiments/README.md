# experiments — antibody design benchmarking

This is the working implementation of the `eval/` harness `DESIGN.md` §4/§7 describes as
not-yet-built. It doesn't recompute any confidence metric — everything here summarizes
or reuses what `scripts/run_design.py` (cofolding) and BoltzGen's own campaign pipeline
already produce.

A growing local database of real antibody-antigen structures and affinity data now
lives at `databases/` (top-level, sibling to this folder) — the natural next source for
expanding `reference_targets.py` beyond its one hand-curated entry, though that
expansion hasn't been built yet. See `databases/README.md` for the full index; the
structures + affinity corpus this project started with is `databases/sabdab/`.

## Design-improvement experiments

The files below are this folder's original scope (benchmarking existing campaigns).
Three subfolders hold the active design-improvement work from
[IMPROVE_DESIGN.md](../IMPROVE_DESIGN.md), one per option, built in order:

| Folder | Status |
|---|---|
| `epitope_prediction/` | Done through milestone 7 ([`PLAN.md`](epitope_prediction/PLAN.md) §8-10) — Model A selected, wired into steering, downstream-validated on 8 targets; v2 (gradient-guided steering) reaffirmed no-go. Follow-up pass in progress: expanded training data, calibration, two EpiFormer-inspired architectures — see [`README.md`](epitope_prediction/README.md). |
| `germinal_comparison/` | Paused after milestone 1 ([`PLAN.md`](germinal_comparison/PLAN.md)) — PyRosetta venv done; JAX/ColabDesign setup deprioritized. |
| `finetuning/` | Not started — see `IMPROVE_DESIGN.md` §3 for scope. |

## What's here

| File | Purpose |
|---|---|
| `thresholds.py` | Pass/fail classification for a handful of metrics. Every threshold is labeled `readme` (sourced from README.md's documented guidance) or `heuristic` (invented here because README doesn't cover that column — read the docstring before trusting a heuristic threshold). |
| `aggregate_metrics.py` | Scans every campaign's `all_designs_metrics.csv` under `data/designs/`, tags each with target + structure provenance (`CAMPAIGN_META`), and produces per-campaign and cross-campaign summaries. |
| `report.py` | Renders `aggregate_metrics.py`'s output (plus, optionally, a reference baseline) into a markdown report + JSON sidecar under `reports/`. The JSON exists because `UI_DESIGN.md`'s planned dashboard is meant to render pre-computed output, not recompute it — treat the JSON as the stable contract, the markdown as the human-readable view. |
| `reference_targets.py` | Curated real (experimentally known) antibody-antigen complexes. Currently just `hel` (PDB `1FDL`, the D1.3 anti-HEL Fab) — matched to the existing `hel_antibody` campaign. Add entries only after independently fetch-verifying the PDB id and (critically) the *label* chain ids, not *auth* ids — see the module docstring. |
| `fetch_reference.py` | Fetches and caches the three sequences (antigen, heavy, light) for a curated reference complex under `reference_data/` (gitignored). |
| `score_reference.py` | Runs a reference complex through `scripts/run_design.py` — the exact same cofolding pipeline used to externally validate generated designs — producing a "what does a true positive score here" baseline. |

## Usage

```bash
# Slice 1 — aggregate what's already on disk, no new compute
python3 experiments/report.py

# Slice 2 — also score a known-real complex as a baseline (launches a real cofolding run,
# same runtime cost as scripts/run_design.py: minutes, GPU)
python3 experiments/report.py --with-reference hel
```

## What this benchmark does NOT do (yet)

- **No structural/epitope-overlap check.** `score_reference.py`'s baseline is a
  confidence-scale calibration only: "does this pipeline assign a comparably high ipTM
  to a known true positive as it does to generated candidates." A generated design can
  score well against the baseline while binding a completely different, non-native
  surface patch — this cannot detect that. A cheap follow-on exists: `gemmi` (mmCIF
  parsing) is already installed in every backend venv on this machine, and a
  distance-based residue-contact epitope-overlap Jaccard score between the reference
  complex's real interface and a design's refolded interface is maybe 30 lines. Left out
  of v1 deliberately — not an oversight — because (a) refolded CIFs are only retained on
  disk for the top-`budget` designs (5-10 per campaign, not all 50-1000 rows scored), and
  (b) it would pull `gemmi` into what's otherwise a bare-`python3`, zero-venv
  orchestration layer, matching `scripts/`'s existing design choice to stay stdlib-only.
- **BoltzGen's own "native"/`sequence_recovery` ground-truth RMSD machinery
  (`src/boltzgen/.../analyze.py`) is not used here** — confirmed it can't be pointed at
  an already-finished campaign's outputs; it requires the reference structure to have
  been the literal conditioning input at generation time (same tensor shapes). That's
  why `score_reference.py` takes the "rerun through Skill A" approach instead.
- **No cross-target novelty/redundancy check** against a broader antibody database
  (SAbDab, PDB). BoltzGen has `compute_novelty_foldseek` but it's non-functional out of
  the box (hardcoded author-specific path) — not wired up here.

## Caveats that apply to every number this produces

Same ones README.md already states for the underlying pipeline, worth repeating because
it's easy to forget while staring at a pass-rate table: these are computational
predictions scored by each model's own confidence, not validated binders. Existing
campaigns on disk are 50-1000 designs; BoltzGen's own guidance is that realistic
hit-finding needs 10,000-60,000. Treat every pass rate in `reports/` as demo-scale, not
a representative campaign hit rate.
