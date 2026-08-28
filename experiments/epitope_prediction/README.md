# Epitope prediction — results summary

Full design rationale, literature grounding, and detailed narrative live in
[`PLAN.md`](PLAN.md) — this file is the results-focused reference: what was built, what
each experiment found, and how to reproduce it. Two work phases:

- **v1 (original build)**: interface labeler, training data, Models A/B, steering
  integration, 8-target downstream validation. All done; see [§1](#1-v1-original-build).
- **v2 (this pass)**: expanded training data, fixed a real labeling bug, added features
  and calibration, retrained A/B, and built two new EpiFormer-inspired architectures
  (Models C/D) to see whether a fancier borrowed architecture beats the original simple
  design. **It does**: Model D (a faithful fork of EpiFormer's own encoder, once given a
  training recipe suited to its size) now beats Model A on every metric — see
  [§2](#2-v2-data-expansion-calibration-and-two-new-architectures).

---

## 1. v1 (original build)

Antibody-agnostic epitope prediction: given only an antigen (no candidate antibody yet),
predict which surface residues are likely to be *someone's* epitope, to steer BoltzGen's
`binding_types` conditioning during antibody generation. Two architectures were built and
compared — Model A (geometric-only GNN) and Model B (geometric + ESM2) — both as 5-member
bagging-PU ensembles producing `(propensity, confidence)` per residue.

**Original results (test.txt, 751 structures, AACDB-only training set, 3,520 structures):**

| model | AUC | Brier | confidence-sanity |
|---|---|---|---|
| GBT baseline | 0.574 | 0.087 | n/a |
| **Model A (geometric)** | **0.625** | 0.086 | passes |
| Model B (geometric+ESM2) | 0.582 | 0.088 | **fails** (overfit, non-monotonic) |

Model A selected. Wired into BoltzGen's `binding_types` conditioning and validated on 8
real conditioned-vs-unconditioned campaign pairs (3 `dev.txt` + 5 `test.txt` targets,
50 designs each, real GPU campaigns): conditioning improved mean recall on 5/7 confident
targets, tied on 1, lost on 1 — a real, positive but noisy effect, not a guaranteed win.
Go/no-go on v2 (gradient-guided steered diffusion): **no-go**, since the bottleneck was
diagnosed as prediction accuracy, not the conditioning mechanism itself. Full narrative,
target-by-target breakdown, and two real bugs caught along the way (a SASA NaN-fallback
scale mismatch, an alt-loc atom-duplication bug) are in `PLAN.md` §9–§10.

---

## 2. v2: data expansion, calibration, and two new architectures

Motivated by `PLAN.md` §10's own recommendation: "the better next investment remains
improving or better-calibrating the epitope model." Mid-way, evaluated ideas from a new
paper, **EpiFormer** (arXiv [2606.04154](https://arxiv.org/abs/2606.04154)) — an
antibody-*aware* epitope predictor (needs a real antibody as input, unlike our
antibody-agnostic setup) — leading to two additional model variants and a plan to use
EpiFormer's real, pretrained model as an independent downstream evaluator.

### 2.1 Training-data expansion

AACDB (the original training-label source) only annotates 3,520 of the 8,072
`train_era` structures — 97% of its own ceiling, no more headroom there. The
coordinate-based labeler already built for `dev`/`test` eval
(`data/interface_labels.py`) has no AACDB dependency, so it was extended to cover the
remaining ~4,444 `train_era` structures directly from `databases/sabdab/` coordinates.
Training set grew **3,520 → 6,149 structures** (4.26M residues).

### 2.2 Real bug found and fixed: non-protein antigens scored as epitope residues

`get_chain_atoms()` filtered atoms only by element (`!= "H"`), not by residue identity —
so a SAbDab "antigen chain" that was actually a bound ion, hapten, or sugar (per SAbDab's
own `antigen_type` column) got scored as real epitope residues. Verified concretely:
`pdb_00001a0q`'s "epitope" turned out to be 2 zinc ions + a heparin fragment. Affected
**24% of `train_era`**, ~2% of `dev`/`test` (already-reported v1 numbers were only
mildly affected). Fixed with a `STANDARD_RESIDUES` allowlist in `get_chain_atoms()`
(`data/interface_labels.py`). Verified via the existing AACDB sanity-check: mean Jaccard
agreement improved **0.587 → 0.748**.

A second bug surfaced by the larger, more diverse structure pool: the labeler's dense
pairwise-distance matrix OOM'd (338 GiB) on one large multi-copy structure. Replaced with
a `scipy.spatial.cKDTree` radius query — verified behavior-preserving (identical
sanity-check Jaccard before/after).

### 2.3 Feature and training fixes

- Fixed the SASA NaN-fallback (`model/features.py`): the old fallback divided raw SASA
  by an arbitrary constant, silently mixing relative and absolute scales for ~40% of
  residues. Now uses real per-residue-type theoretical max-ASA values (Tien et al. 2013).
- Added 3 features: hydrophobicity (Kyte-Doolittle), a protrusion/curvature proxy (a
  docstring had claimed this existed; it hadn't), and distance-to-antigen-centroid.
  `GEOMETRIC_DIM` 4 → 7.
- Added class-imbalance handling (`pos_weight` in `BCEWithLogitsLoss`, previously absent
  despite a 6–10% positive rate) — `model/gnn.py`.
- Added post-hoc isotonic calibration (`model/calibration.py`), previously nonexistent
  anywhere in the pipeline.
- Fixed a bug in the eval code itself: the confidence-sanity check was validating
  ensemble confidence against *calibrated* propensity, but confidence is derived from
  raw ensemble disagreement — calibration (fit only on propensity) doesn't preserve
  per-confidence-bin error ordering, so this produced a false-looking failure. Now
  checks against raw propensity, matching what confidence is actually computed from.

### 2.4 Models A & B, retrained

| model | AUC | Brier (raw) | Brier (calibrated) | precision@recall~0.3 | precision@recall~0.5 | confidence-sanity |
|---|---|---|---|---|---|---|
| GBT baseline | 0.714 | 0.076 | — | 0.198 | 0.175 | n/a |
| **Model A** | **0.867** | 0.196 | **0.060** | **0.862** | **0.535** | passes |
| Model B | 0.866 | 0.151 | 0.058 | 0.742 | 0.523 | passes (previously failed) |

Both models improved substantially (AUC +0.24 for A). More interestingly: Model B, which
originally lost outright and failed its confidence-sanity check from overfitting, now
**ties Model A on AUC/Brier** and its confidence signal is reliable too — the original
"ESM2 doesn't transfer, geometric features do" story looks like it was partly a
data-scarcity artifact. Model A still wins clearly on precision-at-fixed-recall over B —
but see §2.6: Model D ends up beating A on every metric here too, so A's status as
"selected model" doesn't survive past this section.

### 2.5 Model C — EpiFormer-inspired encoder, from scratch (`model/egnn_encoder.py`)

EpiFormer's antigen-side encoder (before its cross-attention, which needs an antibody we
don't have) is an E(3)-equivariant multi-relational GNN: 4 edge relations (sequential,
short-range, k-NN, medium-range spatial), per-relation message MLPs, equivariant
coordinate updates. Reimplemented from the paper description (not their code — that's
Model D below), antigen-only, no antibody input, no cross-attention.

**Real bug found and fixed**: the first training run diverged catastrophically (loss
~10²⁰–10²⁷ from epoch 1, val AUC stuck at chance, across all 5 ensemble members). Root
cause: the coordinate update summed over every neighbor (up to ~120/node at `KNN_K=30`
across 4 relations) with no degree normalization and no bound on the learned scalar
magnitude — a classic EGNN runaway-feedback failure. Fixed with degree-normalized
(`mean`) aggregation and a `tanh` bound on the coordinate scalar, matching what the real
EpiFormer code does by default (`coords_agg='mean'`) and had been omitted from this
reimplementation. Verified stable on real data before retraining.

**Result after the fix** — trains stably, but underperforms both A and B on every
metric:

| | AUC | Brier (calibrated) | precision@recall~0.3 | precision@recall~0.5 | confidence-sanity |
|---|---|---|---|---|---|
| Model C | 0.838 | 0.066 | 0.544 | 0.383 | **fails** |

### 2.6 Model D — a faithful fork of EpiFormer's actual code (`model/res_mp_fork.py`)

Went further than Model C: forked EpiFormer's *actual* antigen-branch encoder source
(`src/epiformer/model/res_mp.py`'s `MultiRelationalEGNNLayer`/`ResMP`, the class their
own `EpiformerBlock` instantiates by default), stripped of the antibody branch, ported to
run on plain tensors instead of PyG `HeteroData` and `index_add_` instead of
`torch_scatter` (so it runs in the same aarch64-friendly venv as everything else). Kept
their real hyperparameters (`hidden_dim=128` vs our 64, RBF distance encoding, per-relation
coordinate-update weights) — a faithful port, not another simplified variant. Reuses
Model C's exact data cache and this project's shared ensemble/calibration/eval harness,
so only the encoder architecture differs across A/C/D.

**First attempt** (shared training recipe: `lr=1e-3`, `patience=5`) trained numerically
stably (their own degree normalization carried over correctly) but badly
underperformed — worse than the simple GBT baseline, and with a confidence signal that
was *anti-correlated* with correctness (error rose from 0.52 to 0.66 as confidence
increased):

| | AUC | Brier (calibrated) | precision@recall~0.3 | precision@recall~0.5 | confidence-sanity |
|---|---|---|---|---|---|
| Model D (lr=1e-3, patience=5) | 0.655 | 0.078 | 0.148 | 0.136 | fails badly |

Most ensemble seeds stopped very early (one at epoch 6). Rather than conclude the
architecture doesn't transfer, retried with a recipe suited to its larger parameter
count (`lr=3e-4`, `patience=15`) — and the result flipped completely:

| | AUC | Brier (calibrated) | precision@recall~0.3 | precision@recall~0.5 | confidence-sanity |
|---|---|---|---|---|---|
| **Model D (lr=3e-4, patience=15)** | **0.876** | **0.054** | **0.937** | **0.766** | **passes, cleanly monotonic** |

Model D now **beats Model A on every metric**, including the precision-at-fixed-recall
numbers that made Model A the clear pick over Model B. The original 0.655 result was
undertraining, not an architectural ceiling — most of its ensemble seeds under the old
recipe had stopped by epoch 6–15; under the new one, one seed used the full 60-epoch
budget without fully converging. **Model D is now the strongest candidate model
overall** — worth promoting to the selected model for steering, pending confirmation
that its added size/training cost (1.3M params vs Model A's much smaller SAGEConv net)
is worth it for the accuracy gain.

### 2.7 EpiFormer as an independent downstream evaluator (infrastructure ready, not yet applied)

Separately from Models C/D, the plan is to use EpiFormer's actual pretrained model — with
a real generated antibody as input, which it needs and we have at downstream-evaluation
time — as a second, independently-trained opinion on the 8 already-completed BoltzGen
conditioned-vs-unconditioned campaigns from v1 §10 (`eval/.downstream_runs/`, still on
disk, no new GPU-hours needed).

**aarch64 feasibility, resolved**: EpiFormer needs `torch-scatter`/`torch-sparse`/
`torch-cluster`/`torch-spline-conv`, which have no prebuilt wheels for aarch64 anywhere
(checked PyPI and PyG's own wheel index). Built all four from source against a
sm_121-compatible PyTorch nightly in a new `.venvs/epiformer` — worked cleanly (~15 min
total). Also built HMMER from source (user-local, no sudo) since ANARCI's CDR-detection
needs `hmmscan` and it wasn't installed.

**Two real bugs found and patched in EpiFormer's own vendored code**
(`src/epiformer/inference.py`): a `torch.load(weights_only=...)` incompatibility with
PyTorch 2.6+'s new default, and an ANARCI return-value nesting mismatch (their code
assumed one less level of list-nesting than the installed `anarci` package actually
returns). Also downgraded `transformers` to `<5` for AntiBERTy compatibility.

**5-structure sanity check** (real antibody as input, `epitope-group` checkpoint):

| structure | true epitope size | recall |
|---|---|---|
| pdb_00001a14 | 21 | 0.00 |
| pdb_000013bf | 19 | **0.79** |
| pdb_00008pg0 | 21 | 0.00 (near-abstention, 1 residue predicted) |
| pdb_00008tui | 23 | **1.00** |
| pdb_00009bqw | 21 | 0.00 |

Genuinely mixed (2/5 strong, 3/5 miss) — consistent with EpiFormer's own reported
F1=0.305 on its harder benchmark split, not a broken pipeline. Its downstream verdicts
should be treated as one more signal, not a tie-breaker on their own.

**A third real bug, found scoring actual generated designs** (not just natural
structures, all the 5-structure sanity check above used): one whole target (10/10
designs, both variants) crashed with `TypeError: 'NoneType' object is not
subscriptable` inside ANARCI's numbering path. Root cause: `results[0]` (per-chain
ANARCI output) was a non-`None` list whose first element was itself `None` — a
"detected something domain-like but couldn't number it" failure mode, distinct from
"no domain found" (`results[0] is None`, already handled by the earlier fix). More
likely on designed sequences than natural ones (unusual framework residues near the
CDRs that a natural-sequence-trained HMM alignment doesn't expect). Hardened
`identify_cdr_residues_anarci()` to check every level of nesting explicitly rather than
assume a fixed depth is always populated; falls back to the Chothia heuristic for that
chain instead of crashing.

**Applied to all 8 real downstream targets (`eval/epiformer_downstream_score.py --all`),
baseline vs. Model A's conditioned campaigns** (mean recall across each variant's top-5
ranked designs):

| target | baseline | conditioned (Model A) | Δ | agrees with original geometric-method finding (§1, PLAN.md §10)? |
|---|---|---|---|---|
| pdb_000010gh | 0.000 | 0.000 | — | yes (both zero) |
| pdb_00008pmy | 0.345 | **0.609** | +0.264 | yes (this was the "largest effect" target originally too) |
| pdb_00008tzu | 0.218 | 0.278 | +0.060 | yes |
| pdb_00009cb5 | 0.291 | 0.382 | +0.091 | yes |
| pdb_00009cct | **0.575** | 0.308 | **−0.267** | **no — original found conditioning helped here** |
| pdb_00009me5 | 0.314 | 0.318 | ~0 | partial (original found conditioning clearly hurt; EpiFormer sees a wash) |
| pdb_00009me7 | 0.554 | 0.549 | ~0 | partial (original found a modest win; EpiFormer sees a wash) |
| pdb_00009uvi | 0.494 | 0.518 | +0.024 | yes (both saw a near-wash with a mild edge to conditioning) |

Mostly directionally consistent with the original purely-geometric 5Å-contact method —
useful confirmation that that method's conclusions weren't an artifact of its own
scoring approach. One real disagreement worth flagging rather than averaging away:
**pdb_00009cct**, where EpiFormer says conditioning clearly *hurts* (baseline recall
almost double) while the original method found a (weak) improvement. Neither method is
obviously right here; treat this specific target's conclusion as unresolved rather than
picking a side.

**Model D integration and re-run in progress**: `steering/binding_types_spec.py` now
uses Model D (§2.6) instead of Model A, since it wins the held-out comparison
comprehensively. Re-running the same 8-target conditioned-vs-baseline campaigns with
Model D's predictions (`downstream_eval.py --conditioned-variant conditioned_D`) —
baseline runs are reused unchanged (never depend on the steering model), so only 8 new
conditioned campaigns are needed, each still costing the same real GPU-hours as the
original run (design ~50min–5.6h + refold ~1–9h per target).

**Real gap found integrating Model D, worth flagging plainly**: `binding_types_spec.py`'s
`CONFIDENCE_FLOOR=0.85` was tuned against Model A's own confidence distribution and
turned out far too strict for Model D — its 5-member ensemble disagrees more even on
good calls (bins ~0.29–0.95 vs Model A's ~0.63–0.95), so at 0.85, most targets' top
propensity-ranked residue never cleared it, and one target (`pdb_00008tzu`) produced an
empty selection outright (not a real "no signal" case — its confidence was 0.4–0.8, just
not ≥0.85). Lowered to `0.6`, chosen from Model D's own real confidence-sanity-check
bins (§2.6: bin3 confidence~0.62/error 0.243 is the last bin still meaningfully more
accurate than bin1/2). **Caveat this creates**: targets 1–2 below were already run under
the old 0.85 floor before this was caught; re-running them would cost more GPU-hours for
uncertain benefit, so they're left as-is rather than redone — reported with their actual
floor noted, not silently treated as uniform with targets 4 onward. `pdb_00008tzu`
still produces an empty selection even at 0.6 (its top call's confidence is 0.453) —
a genuine abstention under either floor, skipped for this comparison (matches the
original design's own philosophy: don't force a low-confidence guess).

**A second real gap, found reconciling these numbers**: the `STANDARD_RESIDUES` fix
(§2.2) changed `compute_interface_labels_by_label_seq()`'s ground-truth epitope
definition *after* the original PLAN.md §10 numbers were computed with it — since that
function feeds both the true-epitope set and (via `_antigen_contacts`) the design-contact
recomputation, re-scoring the *same, unchanged* baseline/Model A campaigns with current
code doesn't always reproduce the original numbers. Recomputed all 8 (cheap — no GPU
work, just rescoring already-completed campaigns, via `downstream_eval.py <pdb_id>
--compare`) rather than assume old and new code agree: 6/8 targets reproduce their
original PLAN.md numbers exactly or almost exactly; 2 (`pdb_00009cb5`, `pdb_00009uvi`)
shifted noticeably — consistent with the ~2% of `dev`/`test` structures the original bug
quantifiably affected. The table below uses the current, freshly-recomputed numbers
throughout (baseline and Model A included) for a genuine apples-to-apples footing
against Model D, not the original PLAN.md values.

**Own-metric (5Å geometric contact recomputation, matching §1/PLAN.md §10's original
methodology) mean recall, conditioned vs. baseline — all recomputed with current code:**

| target | baseline | conditioned (Model A) | conditioned (Model D) | notes |
|---|---|---|---|---|
| pdb_000010gh | 0.000 | 0.000 | 0.000 (floor=0.85) | all three agree: complete miss (Model D also made only a 1-residue call here); matches original exactly |
| pdb_00008pmy | 0.000 | 0.255 | **0.155** (floor=0.85, union-of-5 recall 0.818 vs D's 0.773) | D clearly beats baseline too, though less than A; baseline/A match original closely (was 0.000/0.243) |
| pdb_00008tzu | 0.221 | 0.301 | *abstained* (confidence never clears 0.6) | matches original exactly; Model D skipped, see above |
| pdb_00009cb5 | **0.400** | **0.455** | 0.436 (floor=0.6, union-of-5 recall 0.955 vs baseline's 1.000) | **ground truth shifted here** (original: 0.303/0.355) — D still beats baseline, similar relative margin |
| pdb_00009cct | 0.050 | 0.067 | **0.000** (floor=0.6) | conditioning HURTS here for both A (EpiFormer's cross-check agrees) and D (worse than A, own-metric now 0.000 vs baseline's 0.050) — a target where conditioning may genuinely backfire regardless of model |
| pdb_00009me5 | 0.335 | 0.376 | 0.237 (floor=0.6) | **row was transcribed swapped with me7 in an earlier edit of this table, corrected here** — matches original exactly for baseline/A; D underperforms baseline here, retaining ~71% of its recall |
| pdb_00009me7 | 0.026 | 0.010 | *running* (floor=0.6) | **row was transcribed swapped with me5 in an earlier edit of this table, corrected here** — matches original exactly (conditioning underperforms here for A too) |
| pdb_00009uvi | **0.247** | **0.247** | *not started* | **ground truth shifted here** (original: tied at 0.289) — still a tie under the new ground truth too |

**Separately, EpiFormer's independent cross-check (§2.7's method) on the same
conditioned_D campaigns** — note this uses a *different* scoring method (EpiFormer's own
learned call, not our 5Å-contact recomputation) so its absolute numbers aren't directly
comparable to the table above, only to its own baseline/Model A rows in §2.7's table:

| target | baseline (EpiFormer) | conditioned Model A (EpiFormer) | conditioned Model D (EpiFormer) |
|---|---|---|---|
| pdb_000010gh | 0.000 | 0.000 | 0.000 |
| pdb_00008pmy | 0.345 | 0.609 | **0.300** — disagrees with the own-metric table above, which shows D beating baseline; EpiFormer rates D's conditioning *worse* than baseline here |
| pdb_00009cb5 | 0.291 | 0.382 | 0.318 — between baseline and A, same direction (D beats baseline) as the own-metric table |
| pdb_00009cct | 0.575 | 0.308 | 0.500 — EpiFormer ranks baseline best, D better than A but still below baseline; broadly agrees with the own-metric table that conditioning doesn't help here |

Remaining targets' EpiFormer cross-checks will be added as each Model D campaign
completes.

---

## 3. Reproducing

```bash
# Data
python3 data/train_labels.py --expanded --no-cache      # AACDB + coordinate-labeled train set
python3 data/interface_labels.py --sanity-check          # AACDB agreement check

# Datasets (A/B and C/D use different caches — see model/dataset.py)
python3 model/dataset.py build --split train
python3 model/dataset.py build --split dev
python3 model/dataset.py build --split test
python3 model/dataset.py build --split train --feature-set C   # also used by Model D

# Train
python3 model/baseline.py
python3 model/gnn.py train --model A   # or B, C, D

# Evaluate (test.txt is the reported yardstick; iterate on dev.txt only)
python3 eval/classifier_metrics.py --split test
```

## 4. File map (v2 additions)

| file | role |
|---|---|
| `data/train_labels.py` | `build_expanded_train_labels()` etc. — AACDB + coordinate-labeled training set |
| `data/interface_labels.py` | `STANDARD_RESIDUES` filter (bug fix), KD-tree distance computation |
| `model/features.py` | 7-column feature set, `build_multirelational_features()` for Models C/D |
| `model/gnn.py` | `pos_weight`, `feature_set="C"/"D"` branches |
| `model/calibration.py` | isotonic calibration (new) |
| `model/egnn_encoder.py` | Model C — from-scratch EpiFormer-inspired encoder (new) |
| `model/res_mp_fork.py` | Model D — faithful fork of EpiFormer's real encoder (new) |
| `eval/classifier_metrics.py` | raw+calibrated Brier, confidence-sanity fix |
| `src/epiformer/` | third-party clone (patched), for Model D's reference and the planned downstream evaluator |
