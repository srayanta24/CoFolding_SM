# Experiment 06 — Does a stronger inverse-folding scorer recover the published baseline?

Status: **complete · 2026-11-01** · Arm A (per-complex backbone) + native-backbone arm
stage 1 (shared backbone). Headline below updated after stage 1.

> **Update after the native-backbone arm (stage 1).** Arm A's low +0.097 turned out to
> be substantially an artefact of the benchmark's *per-complex* backbones, not the
> metric. Scoring every mutant on ONE shared reference backbone per target lifts the
> pooled correlation to **+0.247** (from +0.080 for the same ProteinMPNN score on
> per-complex backbones) — roughly the published ≈0.30, and the strongest affinity
> signal anywhere in the arc. It is target-heterogeneous (+0.56 on 3gbn, +0.45 on 4fqi,
> but −0.44 on 1n8z), so it is a *lead*, not a validated ranker. Details in §3.1 and §6.

Generated numbers: [`results/asd_comparison.md`](results/asd_comparison.md) /
[`.json`](results/asd_comparison.json), regenerable with `analyze_asd.py`.
Scores produced by `compute_proteinmpnn.py` (per-complex) and
`compute_proteinmpnn_shared.py` (shared backbone).

**One-line answer: the metric is not the lever — the backbone protocol is.** On the
benchmark's per-complex backbones ProteinMPNN ranks ASD affinity at ρ = +0.097, slightly
*below* ESM-IF1's +0.111 and ~3× under the published ≈0.30. But fixing one backbone per
target and threading the mutants lifts it to **+0.247** — ≈ the published number, and
the first real affinity signal in the arc. It is target-heterogeneous (a −0.44 inversion
on 1n8z), so it is a lead to chase, not yet a usable ranker.

---

## 1. What was the experiment

Score all 1,530 complexes of the six-target within-antigen ASD benchmark
([EXPERIMENT_03](EXPERIMENT_03_within_antigen_ranking.md)) with **ProteinMPNN** — the
official `v_48_020` model, vendored at `src/proteinmpnn` — and compare its within-antigen
affinity-ranking correlation to ESM-IF1's and to AbBiBench's published ProteinMPNN
number.

## 2. What were we testing

The scoring-calibration arc concluded that no cheap metric ranks antibody designs by
binding ([§11](PLAN.md#111-the-headline)). But that conclusion had one soft spot,
recorded as §11.5: **AbBiBench reports ProteinMPNN ρ ≈ 0.30 and ESM-IF1 ≈ 0.28 on the
same affinity data where our best inverse-folding column reached only +0.111.** Someone
else got double our best with the same family of method, so our conclusion rested on a
benchmark whose resolving power was borrowed, not demonstrated.

Two readings, pointing opposite ways:

- **Metric was the lever.** If a stronger inverse-folding model (ProteinMPNN) reaches
  ≈0.30 on *our* structures, then ESM-IF1 was simply weak, our structures are fine, and
  we have a usable Step-3 ranker. The arc's conclusion would soften materially.
- **Metric is not the lever.** If ProteinMPNN lands near +0.11 like ESM-IF1, then
  swapping scorers does not help, and whatever separates us from the published number is
  *not* the metric.

This experiment discriminates them. It is the cheapest high-value check in the arc — a
re-score of existing structures, no new cofolding.

Null control: `ProteinMPNN ll_antigen` (ProteinMPNN scoring the antigen chain), reported
alongside `ESM-IF1 ll_antigen`, for the same reason every experiment here carries one.

## 3. What was the result

Pooled Spearman vs −log KD, z-scored within target. Higher = predicted tighter.

| metric | ρ | ρ partial (null removed) | published |
|---|---|---|---|
| pTM | +0.141 | +0.131 | — |
| ipTM | +0.133 | +0.124 | — |
| ESM-IF1 ll_heavy | +0.111 | +0.105 | ≈0.28 |
| **ProteinMPNN ll_heavy** | **+0.097** | **+0.090** | **≈0.30** |
| ProteinMPNN ll_antibody | +0.080 | +0.075 | — |
| *ESM-IF1 ll_antigen (null)* | +0.070 | — | — |
| *ProteinMPNN ll_antigen (null)* | +0.059 | — | — |

**ProteinMPNN did not help.** It is slightly below ESM-IF1, ~3× under the published
0.30, and its antibody score (+0.080) is only ~0.02 above its own null.

Per target it tracks ESM-IF1 almost exactly — the two inverse-folding scorers rise and
fall together, which is itself the point:

| target | n | ProteinMPNN ll_heavy | ESM-IF1 ll_heavy | ipTM | ProteinMPNN null |
|---|---|---|---|---|---|
| 1mhp | 30 | −0.054 | −0.225 | −0.110 | +0.120 |
| 1n8z | 300 | −0.217 | −0.102 | +0.209 | −0.125 |
| 3gbn | 300 | +0.224 | +0.207 | +0.289 | **+0.433** |
| 4fqi | 300 | −0.080 | −0.014 | +0.024 | −0.028 |
| aayl49 | 300 | +0.326 | +0.340 | +0.083 | −0.073 |
| aayl51 | 300 | +0.330 | +0.210 | +0.042 | +0.087 |

The only targets where inverse folding works at all are the two AlphaSeq ones
(aayl49/51, ~0.33 for both scorers). On the three AbBiBench targets where the 0.30 is
claimed, ProteinMPNN is weak-to-negative — and on 3gbn, its +0.224 sits *below* its own
+0.433 null, so even that apparent signal is target-specific leakage, not affinity.

### 3.1 The native-backbone arm, stage 1: one shared backbone per target

The §6.3 follow-up, run. Instead of scoring each mutant on its own per-complex cofold
(Arm A), fix **one reference backbone per target** — the best-modelled cofold (highest
ipTM) — and thread every mutant's heavy+light onto it (`compute_proteinmpnn_shared.py`).
Zero alignment risk: every cofold of a target shares the manifest's exact chain lengths,
so threading is a sequence swap with coordinates untouched. This isolates the
*per-complex-backbone washout* — scoring a sequence against the backbone folded from it
is partly circular and flattens the mutation signal.

| | pooled ρ | partial vs null |
|---|---|---|
| ProteinMPNN, **per-complex** backbone (Arm A) | +0.080 | +0.075 |
| ProteinMPNN, **shared** backbone | **+0.247** | **+0.242** |
| *(published AbBiBench ProteinMPNN)* | *≈0.30* | — |

**Fixing the backbone roughly tripled the correlation** and brought it to ≈ the
published baseline — the strongest affinity signal in the whole arc, and the partial
(+0.242 ≈ raw) says it is not null leakage. So Arm A's low number was substantially a
*benchmark-construction artefact*, not a metric or absolute-structure failure.

But it is **not a clean universal ranker** — it is strongly target-heterogeneous:

| target | shared-backbone ρ | per-complex (Arm A) ρ |
|---|---|---|
| 3gbn | **+0.562** | +0.190 |
| 4fqi | **+0.445** | −0.079 |
| aayl49 | +0.400 | +0.282 |
| aayl51 | +0.388 | +0.274 |
| 1mhp (n=30) | −0.186 | −0.037 |
| 1n8z | **−0.441** | −0.200 |

Four targets jump strongly positive; 1n8z (HER2/trastuzumab) is strongly *inverted*, and
1mhp (n=30, underpowered) mildly so. The pooled +0.247 is four strong positives dragged
by two negatives, not a uniform effect.

## 4. What dataset did we use

Unchanged from EXPERIMENT_03/04: the six-target ASD within-antigen benchmark, 1,530
complexes, each one antigen with many heavy-chain-varying antibodies, label −log KD.
2fjg and 1mlc excluded (mislabelled −log KD, EXPERIMENT_03 §4).

Structures: the **predicted** per-complex backbones already on disk from the ASD
cofolding run (`.scores/<entry_id>/.../*_model_0.cif`) — the same structures ESM-IF1 and
Rosetta scored, so this is a controlled metric swap, not a new benchmark.

## 5. How did we test the hypothesis

1. **Vendored official ProteinMPNN** (`dauparas/ProteinMPNN`, pinned, `v_48_020`
   weights) into `src/proteinmpnn`, registered in `setup_env.py`.
2. **Replicated ProteinMPNN's own `score_only` path** (`compute_proteinmpnn.py`): for
   each complex, convert the predicted CIF → PDB (gemmi), score the antibody heavy+light
   chains' log-likelihood *conditioned on the antigen backbone*, averaging over 8 random
   decoding orders (fixed seed) to damp autoregressive-order variance. Also scored heavy
   alone and the antigen (null).
3. **Joined into `analyze_asd.py`** beside ESM-IF1, under `mpnn_` names to avoid a
   column collision (both scorers emit `ll_heavy`/`ll_antigen`), and computed the same
   within-target pooled ρ, partial-vs-null, and per-target table used for every other
   metric.
4. **Carried both nulls through.** ProteinMPNN gets its own `ll_antigen` null, so the
   comparison to the published number is read against the right chance baseline.

All reproducible from the TSVs; nothing hand-transcribed.

## 6. What were the findings

**1. The metric was not the lever — the backbone protocol was.** On *per-complex*
backbones (Arm A) a stronger, differently-trained inverse-folding model reproduced
ESM-IF1's weak result almost exactly (+0.097 vs +0.111). But §3.1 shows that was a
benchmark-construction artefact: holding ONE backbone fixed per target and threading the
mutants lifts ProteinMPNN to **+0.247**, roughly the published ≈0.30. The scorer was
never the problem; scoring each sequence against the backbone *folded from it* was. This
revises the earlier reading — it is not "no inverse-folding signal," it is "the signal is
there but the per-complex-cofold benchmark washed it out."

**2. It is a lead, not a validated ranker — and the heterogeneity is the catch.** The
+0.247 is four strongly-positive targets (3gbn +0.56, 4fqi +0.45, aayl49/51 ≈+0.39)
dragged by two negatives, including a strong **inversion on 1n8z** (HER2/trastuzumab,
−0.44). A ranker that is +0.56 on one target and −0.44 on another cannot be trusted on a
new target without knowing which regime it is in. So this reopens inverse folding as a
*candidate* binding signal — the first in the whole arc worth chasing — without yet
delivering a usable one.

**3. Why this is still not AbBiBench's protocol, and stage 2.** The shared backbone here
is an *arbitrary predicted* reference (the best-ipTM cofold), not a crystal. The
inversions (1n8z, 1mhp) could be reference-backbone pathologies as easily as real
biology. Stage 2 — the true native arm — fetches the crystal complex for the four
PDB-backed targets (all present in `databases/sabdab/`), threads the mutant sequences
onto the *crystal* backbone, and re-scores. That both (a) removes the arbitrary-reference
concern and (b) verifies AbBiBench's absolute number in our hands. It carries real
mis-registration risk (1n8z and 4fqi crystals are Fabs — VH+CH1 — so the manifest's
VH-only mutants need verified VH-region alignment, unlike stage 1's exact-length
threading), which is why it was staged second and is not yet run.

**4. For the funnel (the practical consequence).** Stage 1 changes the Step-3 picture
from "no binding ranker is possible" to "a *fixed-reference-backbone* inverse-folding
score is a real but unreliable lead." It is **not** yet recommended for the production
funnel — the 1n8z inversion means it could actively mis-rank a new target — so the
funnel recommendation is unchanged for now (gate on ipTM + filter_rmsd, diversify by
cluster, rank by free plausibility metrics). But it is the one avenue that earned a
follow-up, and `design_ranking`'s "why no binding ranker" note records it as the open
lead rather than a closed door.

### Limitations

- **The shared-backbone reference is an arbitrary predicted cofold** (finding 3) — the
  central caveat on the +0.247, and the reason stage 2 (crystal) is needed before this
  is more than a lead.
- **Strong target heterogeneity** including a −0.44 inversion — the pooled number hides
  it, and a funnel cannot.
- **We have not reproduced AbBiBench's absolute number on a native backbone** — "0.30"
  is still their figure; stage 1's +0.247 is suggestive agreement, not verification.
- **Six targets, two of them AlphaSeq** (aayl49/51, a different assay) — two of the four
  positive targets. Worth remembering before leaning on the aggregate.
