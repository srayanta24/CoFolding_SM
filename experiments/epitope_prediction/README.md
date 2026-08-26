# Epitope prediction — results summary

Full design rationale, literature grounding, and detailed narrative live in
[`PLAN.md`](PLAN.md) — this file is the results-focused reference: what was built, what
each experiment found, and how to reproduce it. Two work phases:

- **v1 (original build)**: interface labeler, training data, Models A/B, steering
  integration, 8-target downstream validation. All done; see [§1](#1-v1-original-build).
- **v2 (this pass)**: expanded training data, fixed a real labeling bug, added features
  and calibration, retrained A/B, and built two new EpiFormer-inspired architectures
  (Models C/D) to see whether a fancier borrowed architecture beats the original simple
  design. See [§2](#2-v2-data-expansion-calibration-and-two-new-architectures).

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
data-scarcity artifact. Model A still wins clearly on precision-at-fixed-recall, the
metric that actually matters for selecting a small trustworthy residue set for
`binding_types` conditioning, so **Model A remains the selected model**.

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

Most ensemble seeds stopped very early (one at epoch 6) — before concluding the
architecture itself doesn't transfer, retried with a recipe suited to its larger
parameter count (`lr=3e-4`, `patience=15`), still in progress as of this writing.
Early signal: val AUC climbing to ~0.89 by epoch 20 on the first seed, well past where
the original run had already stopped — **the original result looks like it was
undertraining, not an architectural ceiling.** Final numbers will be added here once
that run completes.

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

**Not yet done**: running this over the real 8 downstream campaign outputs
(`eval/epiformer_downstream_score.py`, planned but not yet written).

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
