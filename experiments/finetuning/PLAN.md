# Fine-tuning — putting binding knowledge into weights

Status: draft v1 · 2026-10-25

Implements **Option 3** of [IMPROVE_DESIGN.md](../../IMPROVE_DESIGN.md) §6.3, and the
first direct attack on §1: *nothing in this stack was ever trained on antibodies.*

Every metric tested so far — ipTM, ipSAE, Rosetta energies, ESM-IF1, AntiFold — is
**unsupervised with respect to binding**. None of them ever saw a binding label. A
model that has is the one thing this project has not tried, and §5's "training branch"
has been open since v2 of the plan.

---

## 1. What AntiFold established, and why it points here

AntiFold is ESM-IF1 fine-tuned on antibodies — same 142M architecture, same task, same
scoring API. It is therefore a controlled A/B on antibody-specific training at the
inverse-folding stage. On the 28-parent paired benchmark:

| | paired | vs null (17/28) |
|---|---|---|
| ESM-IF1 (generic) | 13/28 | below |
| **AntiFold (antibody-tuned)** | **16/28** | **below** |
| packstat (physics, no training) | 23/28 | above |

Antibody fine-tuning moved it in the right direction (13 → 16) but **not past the null
control**. So the cheapest version of the §1 fix — swap in a model that is the generic
one plus antibody training — produced a gain inside the noise.

Two readings, and the second is the one this plan acts on:

- Antibody *structural* specialisation is worth little at this stage. Plausible: the
  CDR-loop sequence statistics AntiFold learned may already be implicit in a backbone.
- **The missing ingredient is not antibody-ness, it is binding.** AntiFold was trained
  on sequence recovery, not on whether anything binds. No model in our stack has ever
  seen a binding label.

Model size is *not* the constraint. AntiFold is 142M, identical to ESM-IF1 and in the
same range as this project's own Model D. The constraint is label diversity, and that
is what §2 is about.

## 2. What training data actually exists — three traps found while checking

`databases/asd/` looked like the answer: 1.2M rows, 9,575 antigens. Three checks
reduced that substantially, and each is worth recording because each would have
produced a confidently wrong result.

**Trap 1 — the `-log KD` label is unreliable per target.** Already documented in
[EXPERIMENT_03](../scoring_calibration/EXPERIMENT_03_within_antigen_ranking.md) §4:
2fjg and 1mlc carry values down to −4.1, which cannot be −log KD, and are almost
certainly ΔΔG-like with inverted sign.

**Trap 2 — graded affinity exists for almost no antigens.**

| source | rows | **distinct antigens** |
|---|---|---|
| `-log KD` (AbBiBench) | 152,401 | **5** |
| `alphaseq` | 198,703 | **3** |
| `ddg` | 670 | 22 |

Fine-tuning on graded affinity means training on five antigens. It would learn those
antigens, and there is no honest held-out set.

**Trap 3 — the structure-bearing rows are all positives.** `bool` has 8,644 antigens,
which looked like the diversity fix. But only 4,040 rows (2%) carry a `target_pdb`, and
**all 4,040 are label 1**. They come from `structures-antibodies` /
`structures-nanobodies` — solved complexes, which are binders by definition, exactly
like this project's own `clean_positive` set. There are **zero structural negatives**.

This killed the originally chosen plan (train a structure-conditioned model on the
structure-bearing `bool` rows). It does not exist as a dataset.

**What is left, after all three checks:**

| source | rows | antigens w/ both classes | structures? |
|---|---|---|---|
| ASD `bool` (covid-19, hiv, met) | **106,197** | **832** | **no — sequence only** |
| SAbDab `train_era` | 8,072 structures | n/a (recovery task) | yes |
| AbBiBench `-log KD` | 152,401 | 5 | predictable |
| abdesign_db / ab_bind ΔΔG | ~1,300 | ~28 parents | predictable |

Negatives come only from covid-19 (27,696), hiv (22,708) and met (2,484). Everything
else — patents, literature, genbank, structures — is binders-only.

## 3. The plan: a sequence-only binding classifier (branch A) — **BLOCKED 2026-10-28**

> Milestone 1 invalidated this section's premise before any training. The "832 antigens
> of genuine diversity" are 6 MMseqs2 clusters, and the dominant label is neutralization
> rather than binding. Kept below as written for the record; see
> [MILESTONE_01_label_validation.md](MILESTONE_01_label_validation.md) for what replaces
> it and §5 for the remaining options, now reordered.

**Chosen because it is the only diverse supervised signal available, not because it is
the ideal architecture.** 832 antigens with both classes is two orders of magnitude
more diversity than AbBiBench's five, and it is the only option where a held-out
antigen split is meaningful.

**What it is**: antibody sequence + antigen sequence → binds / does not bind. An
antibody language model (AbLang2 or ESM2) and an antigen encoder, with a binding head.

**What it is not**: a replacement for BoltzGen's stage 2. It consumes no structure, so
it cannot inverse-fold. It is a **filter** — a post-hoc scorer over generated designs.

That is a smaller claim than "fix the generator", and it is the honest one. It also
happens to target this project's largest open gap: four experiments have established we
do not have a trustworthy selection signal, and every candidate so far was unsupervised.
A supervised one is the obvious untried thing.

### Why a second stage on AbBiBench is still planned

Per the agreed staging: `bool` first as a diverse but weak pretext task, then
AbBiBench's graded `-log KD` as a narrow but strong fine-tune. Standard practice when
one label is diverse-and-noisy and the other is precise-and-narrow. The second stage
must use leave-one-target-out over the five valid targets, and its numbers are
comparable to AbBiBench's published baselines (ProteinMPNN ρ≈0.30, ESM-IF1 ≈0.28).

### Risks, stated before any training

- **Label provenance.** ASD's `bool` aggregates patents, COVID and HIV literature with
  unknown assay quality, and we have already been burned once trusting an ASD label.
  Validate before training: spot-check a sample against primary sources, and check
  whether "non-binder" means *measured not to bind* or merely *not reported to bind*.
- **COVID/HIV dominance.** The 832 both-class antigens are concentrated in two viral
  families whose members are highly related. Effective diversity is well below 832, and
  a held-out split must be by antigen *cluster*, not antigen identity — the same
  MMseqs2 discipline `databases/splits/` already uses.
- **Class imbalance by source.** Patents contribute 138,726 positives and no negatives.
  Training on the full `bool` would learn "is this in the patent set", not binding.
  Restrict to the three sources that have both classes.
- **A supervised model can learn dataset artefacts rather than biophysics.** The
  null-control discipline that changed the conclusion in EXPERIMENT_02, _03 and the
  paired re-run must come along: carry a quantity that cannot work.

## 4. Milestones

1. ✅ **Label validation** (no GPU) →
   [MILESTONE_01_label_validation.md](MILESTONE_01_label_validation.md).
   **Result: branch A is blocked.** The 832 both-class antigens collapse to **6
   clusters** at the 40%-identity threshold `databases/splits/` uses (800 of them are
   HIV Env strain variants), and the largest negative source is labelled for
   **neutralization, not binding** — DeepAAI's "non-neutralizing" includes antibodies
   that do bind. The second-largest cites a method paper with no documented
   negative-generation methodology. Both halves of branch A's premise fail.
2. ⬜ **Leak-free splits by antigen cluster**, holding out whole clusters. Must also
   exclude anything overlapping the scoring-calibration benchmarks, or the evaluation
   is circular.
3. ⬜ **Baselines before training anything**: sequence identity to nearest training
   antibody; an antibody-LM likelihood with no binding supervision. If the supervised
   model cannot beat these, it has learned nothing useful.
4. ⬜ **Stage 1** — train the binder/non-binder classifier on the 106,197-row set.
5. ⬜ **Stage 2** — fine-tune on AbBiBench `-log KD`, leave-one-target-out.
6. ⬜ **Evaluate on this project's own benchmarks**: the 28-parent paired test and the
   6-target within-antigen set, against the existing metric table *and its null
   controls*. This is the only comparison that matters — published numbers on a
   model's own benchmark have repeatedly failed to reproduce here.
7. ⬜ **Decide**: is a sequence-only filter worth wiring into the funnel, given it
   cannot see structure and the funnel's other metrics can?

## 5. Branches not taken (recorded, not closed)

**Branch B — structure-conditioned on whatever has structures.** Usable set is SAbDab
`train_era` (8,072, sequence-recovery only), AbBiBench's 5 antigens, and ~1,300 ΔΔG
mutants. Pipeline-compatible and leak-free, but sequence recovery is what AntiFold
already did, and §1's result suggests the return is small. Revisit if the sequence-only
filter works and a structure-aware version becomes worth the data cost.

**Branch C — supervised structure-conditioned training.** Dead as of this document:
no structural negatives exist. Would become viable only if we generated negatives
ourselves — e.g. by cofolding deliberately mismatched antibody/antigen pairs. That is
cheap to do and worth revisiting, with the caveat that synthetic decoys are much easier
to reject than real non-binders and would likely overstate performance.

**Branch D — fine-tune BoltzGen's own inverse folder.** The only option that puts the
gain in the real pipeline. Blocked on what §6.3 records: all training configs need 8
GPUs even for the 12-block variant, there is no LoRA/PEFT anywhere in the vendored code,
and the data pipeline expects a specific `target_dir`/`msa_dir`/`moldir` layout with
precomputed cluster ids. Unchanged by anything here.

**Full-scale `bool` with predicted structures** — the version that would make branch C
real at 8,644-antigen diversity. ~20,000 GPU-hours, not feasible on this box. Written
up separately in [FULL_SCALE_BOOL.md](FULL_SCALE_BOOL.md) so it is ready if cluster
access ever appears.
