# Full-scale structure-conditioned training on ASD `bool` — a cluster-scale proposal

Status: **not scheduled · written 2026-10-25 · requires compute this project does not have**

This is deliberately not a plan we are executing. It is the version of
[PLAN.md](PLAN.md)'s branch C that would actually work, written up while the reasoning
is fresh so it is ready if cluster access ever appears. Nothing here should be started
on the DGX Spark.

---

## 1. The idea in one paragraph

ASD's `bool` subset holds **257,656 antibody–antigen pairs across 8,644 antigens** with
binder/non-binder labels. That is, by two orders of magnitude, the most diverse binding
signal available to this project. It is unusable today for anything structure-aware
because **only 2% of it has a structure** — and, worse, every structure-bearing row is a
positive (PLAN.md §2, trap 3). Predict the missing complexes and both problems vanish at
once: you get structures *and* you get structural negatives, which currently do not exist
anywhere in our data.

The result would be the first dataset in this project capable of training a
structure-conditioned model on binding itself, rather than on sequence recovery
(AntiFold's objective) or on confidence (everything else we have measured).

## 2. Why it is out of reach here

| | |
|---|---|
| complexes to predict | ~257,656 (or ~106,197 restricting to the 832 both-class antigens) |
| measured cofold cost | ~5 min/complex at ~600 residues, ~6.3 min at ~890 (EXPERIMENT_01/03) |
| **total GPU time** | **~21,000 GPU-hours** (full) / **~8,800** (both-class subset) |
| on one GB10 | ~2.4 years / ~1.0 year |
| on 64 A100s | ~2 weeks / ~6 days |

Those per-complex numbers are measured on this hardware, not estimated: 332 complexes in
16.9 h and 330 in 34.8 h, across two benchmarks.

Storage is the quieter constraint. Our 330-complex ASD run produced ~1 GB of structures
and PAE matrices; linearly that is **~780 GB for the full set**, which is fine on a
cluster and awkward here (3.4 TB free, but shared with everything else).

## 3. What would have to be true before spending the compute

Each of these is cheap to check and would invalidate the whole exercise if it failed.
They are listed in the order that fails fastest.

1. **Do the `bool` negatives mean what we think?** The entire value rests on 52,888
   non-binders, all from covid-19, hiv and met. If "non-binder" means *not reported to
   bind* rather than *measured not to bind*, the label is absence-of-evidence and the
   dataset is worthless for this. **Check before anything else** — it is a literature
   question, not a compute one.
2. **Is the diversity real?** 8,644 antigens, but the both-class ones are concentrated
   in two viral families whose members are highly related. MMseqs2 clustering at the
   40% identity used by `databases/splits/` would give the effective number. If 832
   antigens collapse to ~20 clusters, the diversity argument evaporates and this becomes
   a very expensive version of AbBiBench.
3. **Does predicted-structure quality survive at this scale?** EXPERIMENT_03 found
   ESM-IF1 scoring +0.078 on our predicted backbones against ≈0.28 published on crystal
   structures. If predicted backbones are that much worse, training on 250k of them
   trains on 250k degraded inputs. A 500-complex pilot comparing predicted vs deposited
   structures for the ~4,000 rows that have both would settle it for ~40 GPU-hours.
4. **Is a structure-conditioned model actually better than sequence-only here?** PLAN.md's
   branch A trains a sequence-only classifier on the same labels for essentially no
   compute. **Run that first.** If sequence-only already saturates the signal, the
   21,000 GPU-hours buy nothing, and we will know for the price of a few CPU-days.

Point 4 is the load-bearing one. This document should not be executed until branch A has
reported, because branch A is the cheap experiment that tells you whether the expensive
one is worth running.

## 4. If it were run — sketch

**Data.** Restrict to the 832 both-class antigens (106,197 rows) rather than the full
257,656; the binders-only sources (patents, literature, genbank) contribute no
discriminative signal and would teach the model to recognise the patent set. Split by
antigen *cluster*, not antigen, holding out whole clusters — the same discipline
`databases/src/build_splits.py` already implements. Exclude anything overlapping the
scoring-calibration benchmarks, or evaluation is circular.

**Structures.** Cofold each pair with Boltz-2, no MSA, exactly as `score_sets.py` does,
so the inputs match what the design pipeline produces at inference. Retain PAE matrices:
they cost nothing extra at generation time and we have twice wanted them afterwards.

**Model.** Fine-tune a structure-conditioned encoder — AntiFold or ESM-IF1, both 142M —
with a binding head, rather than training from scratch. Size is not the constraint
(PLAN.md §1); data is. A larger model becomes worth considering only if the 142M one is
demonstrably capacity-limited.

**Evaluation.** This project's own benchmarks, with null controls. Published numbers on a
model's own benchmark have failed to reproduce here twice (ESM-IF1, AntiFold), so an
external figure is not evidence. The specific bar: beat packstat's 23/28 on the paired
test, and beat the antigen-only null on the within-antigen set.

## 5. What makes this worth keeping on file

Three things would be true of this dataset that are not true of anything we have:

- **Structural negatives.** Our only ones are ~256 destroyed point mutants. Point
  mutants are a far subtler perturbation than a genuine non-binder, which is a standing
  limitation on every paired result in this project.
- **Antigen diversity with binding labels.** AbBiBench has 5 antigens; this would have
  hundreds of clusters.
- **Supervision on binding.** Every metric this project has evaluated is unsupervised
  with respect to binding. That is the single largest structural gap in the approach, and
  this is the only dataset that could close it at scale.

Whether those are worth ~9,000–21,000 GPU-hours depends entirely on §3's four checks,
three of which cost nothing.
