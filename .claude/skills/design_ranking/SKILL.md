---
name: design_ranking
description: Rank and down-select a large BoltzGen antibody-design pool (hundreds–60,000 designs) into a diverse shortlist to hand to a wet lab, using the funnel the scoring-calibration arc validated — cluster for diversity, gate on plausibility (ipTM + filter_rmsd), rank within cluster by free plausibility metrics. Use for "rank these designs", "pick the top N from this campaign", "which designs should we order/test", "build the shortlist", "down-select the pool". NOT for scoring a single complex or running a design campaign (see the "design" skill); NOT for epitope-steered generation (see "epitope_guided_design"). This skill selects from an existing pool; it does not generate.
---

## What this is

A down-selection funnel for a finished BoltzGen antibody campaign — turning a pool of
hundreds to tens of thousands of designs into a **diverse shortlist of plausible
candidates**. It is the operational form of what five experiments established in
[`experiments/scoring_calibration/`](../../experiments/scoring_calibration/PLAN.md):
no cheap metric ranks antibody designs by *binding*, but structural plausibility and
diversity are both selectable and cheap, and that is what a hit-finding shortlist
actually needs.

**Read this first — what the funnel does and does not give you.** It enriches for
designs that are *foldable, self-consistent, and diverse across binding modes*. It does
**not** rank by predicted affinity or even predicted binding — there is no validated
metric for that in this project (§"Why no binding ranker" below). Treat the output as an
*existence filter's shortlist*, not a ranked list of likely binders. Hand the wet lab
diversity + plausibility; do not promise them rank-ordered potency.

Everything the funnel uses is already in each campaign's
`all_designs_metrics.csv` (`data/designs/<campaign>/run*/final_ranked_designs/`) — no
new GPU, no FastRelax, no extra scoring model.

## The funnel

### Step 1 — cluster for diversity (MMseqs2)

Cluster the pool so the shortlist spans binding modes rather than 500 near-copies of one
lineage. **Cluster on CDR-H3, not whole sequence.** Antibodies share framework germline,
so whole-VH identity is dominated by framework and collapses everything into one cluster
— the same reason `databases/splits/` clusters *antigens* at 40% but that threshold is
meaningless for antibody frameworks. CDR-H3 is where the binding-mode diversity lives.

- Extract CDR-H3 from each `designed_sequence` (ANARCI/IMGT; the epitope_prediction code
  already has ANARCI numbering — see `src/epiformer` usage) and cluster with MMseqs2
  (`databases/src/_common.py:ensure_mmseqs2()` provides the binary).
- **Richer option**: cluster by *predicted epitope* instead, via the
  `epitope_guided_design` skill (Model A). Epitope-diverse candidates are the diversity
  that matters most for a campaign; use this when epitope predictions are available.
- Keep the clustering; it drives Step 3's cluster-wise selection.

### Step 2 — gate on plausibility (free, measured thresholds)

Discard designs that are not credibly modelled. Two free columns, both validated:

| gate | column | threshold | source |
|---|---|---|---|
| interface confidence | `design_to_target_iptm` | **≥ 0.699** pass, ≥ p25 (~0.58) borderline | measured median real binder, `experiments/thresholds.py` `MEASURED_REFERENCE` (EXPERIMENT_01) |
| structural self-consistency | `filter_rmsd` | **≤ 2.5 Å** | the project's refold-match bar; ipTM's best free partner (EXPERIMENT_05) |

Apply the gate **per cluster**, not globally — keep each cluster's gate-passers rather
than discarding a whole binding mode because its absolute ipTM runs lower. ipTM here is a
*plausibility gate* (can a credible interface be modelled), **not** a binding score (see
below). On EXPERIMENT_05 these two were the best predictors of design self-consistency
(ipTM AUC 0.708, `delta_sasa_refolded` 0.675), and both are free.

### Step 3 — rank within cluster, then sample across clusters

Within each surviving cluster, rank by the **free plausibility metrics**, best first:
`design_to_target_iptm`, then `delta_sasa_refolded` (both are in BoltzGen's own
Algorithm-2 ranking). Then take a **diverse sample across clusters** for the shortlist —
e.g. top-k per cluster, breadth over depth — because diversity (Step 1), not the
within-cluster ranking, is doing the real selection work.

**Do not rank by `packstat`, and do not run FastRelax for ranking** (see below). Skip
Rosetta entirely unless a validated use appears.

## What NOT to do, and why (hard-won — do not relitigate)

- **Do not use `packstat` to rank or filter designs.** It won its one test (23/28 on
  binding-retained-vs-destroyed point mutants of crystal backbones) and then failed
  every other: affinity ranking ρ=+0.058, *below its own null*
  ([EXPERIMENT_04](../../experiments/scoring_calibration/EXPERIMENT_04_packstat_transfer.md));
  and on real designs AUC **0.508 (chance), anti-correlated on a second target**, even
  among already-plausible designs
  ([EXPERIMENT_05](../../experiments/scoring_calibration/EXPERIMENT_05_design_transfer.md)).
  It carries no signal on generated backbones.
- **Do not spend FastRelax (Rosetta) on a 60k pool.** It is ~5–8 min/complex, and on
  EXPERIMENT_05's label the best Rosetta metric did not beat free ipTM. Two expensive
  Rosetta passes in this arc returned negative-or-redundant results.
- **Do not treat ipTM (or any metric here) as a binding/affinity selector.** ipTM is at
  chance (14/28) on the within-complex comparison a campaign makes, and ranks affinity
  near its null. It measures whether a credible structure can be modelled — plausibility,
  not potency (EXPERIMENT_01, confirmed from both sides in _05).
- **Do not swap in a "better" inverse-folding scorer expecting a binding ranker.**
  ProteinMPNN (published ρ≈0.30 on this affinity data) reproduced ESM-IF1's weak +0.11 on
  our structures (+0.097), nowhere near 0.30
  ([EXPERIMENT_06](../../experiments/scoring_calibration/EXPERIMENT_06_proteinmpnn_transfer.md)).
  The metric is not the lever.

## Why no binding ranker — and the one open lead

Six experiments, three benchmarks, four metric families: nothing ranks antibody designs
by binding in a way that survives a change of question (two-benchmark ordering agreement
ρ=+0.087). That is why Step 3 ranks plausibility, and the funnel is an existence filter.

**The one lead worth watching.** EXPERIMENT_06 found that scoring antibodies against
*per-complex* predicted backbones washes out inverse-folding signal, but fixing **one
backbone per target** and threading the sequences makes ProteinMPNN's affinity
correlation real: +0.247 on a shared predicted backbone, and **+0.591 on the real
crystal for 4fqi** — the strongest affinity signal in the arc. So a *fixed-native-
backbone* inverse-folding score genuinely ranks affinity for some targets. **Do not wire
it into this funnel yet, and the reason is now sharper than "unvalidated":** it is
genuinely target-specific — 1n8z (HER2/trastuzumab) *inverts* even on the crystal
(ρ=−0.16), and we cannot yet predict which targets it tracks vs. inverts. A score that
flips sign between targets will actively mis-rank a new one you cannot pre-classify. If a
future Step 3 wants a binding-aware score, this is the candidate — thread designs onto a
single canonical reference (the predicted or crystal target structure), score with
ProteinMPNN — but only once there is a way to tell, per target, whether it will work. The
alternative is a supervised model, blocked on binding data that
[does not exist in `databases/`](../../experiments/finetuning/MILESTONE_01_label_validation.md).
Until one of those lands, this funnel is the honest best.

## Gotchas

- **`filter_rmsd` is structural self-consistency, not binding** — a design can pass it
  and not bind. The gate removes wrecked designs; it does not confirm binders.
- **Only the top-`budget` designs keep a refolded structure on disk** for most campaigns
  (5–10 per campaign). The metrics CSV covers the whole pool, so this funnel runs on the
  full pool from CSV columns alone — but anything needing the *structure* (e.g. a future
  structural ranker) is limited to the retained few. If a campaign should support
  structural re-ranking later, retain structures across the rank range at launch
  (a `--budget`/retention decision — see `scoring_calibration/PLAN.md` §8b).
- **Whole-VH clustering collapses to one cluster** — always cluster on CDR-H3 or epitope.
