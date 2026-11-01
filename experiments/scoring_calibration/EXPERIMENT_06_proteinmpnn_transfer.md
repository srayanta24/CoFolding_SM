# Experiment 06 — Does a stronger inverse-folding scorer recover the published baseline?

Status: **complete · 2026-11-01** · Arm A (per-complex backbone) + native-backbone arm
stage 1 (shared backbone). Headline below updated after stage 1.

> **Update after the native-backbone arm (stages 1 and 2).** Arm A's low +0.097 was
> substantially an artefact of the benchmark's *per-complex* backbones, not the metric.
> On ONE shared predicted backbone per target the pooled correlation rose to **+0.247**
> (stage 1); on the real **crystal** backbone, 4fqi reached **ρ=+0.591** — the strongest
> affinity signal in the arc (stage 2). So a fixed native backbone + ProteinMPNN *does*
> rank affinity. But it is target-specific in a way structure quality cannot fix: 1n8z
> (HER2/trastuzumab) inverts on the crystal too (−0.16). A real lead, not a universal
> ranker. Details in §3.1–3.2 and §6.

Generated numbers: [`results/asd_comparison.md`](results/asd_comparison.md) /
[`.json`](results/asd_comparison.json), regenerable with `analyze_asd.py`.
Scores produced by `compute_proteinmpnn.py` (per-complex) and
`compute_proteinmpnn_shared.py` (shared backbone).

**One-line answer: the metric is not the lever — the backbone *protocol* is.** On the
benchmark's per-complex backbones ProteinMPNN ranks ASD affinity at ρ = +0.097, ~3× under
the published ≈0.30. Fix one backbone per target and thread the mutants and it rises to
**+0.247** (shared predicted) and **+0.591 on a real crystal for 4fqi** — the strongest
affinity signal in the arc. So inverse folding *does* rank affinity on a fixed native
backbone. But it is genuinely target-specific (1n8z/HER2 inverts even on the crystal):
a real lead to chase, not a universal ranker.

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

### 3.2 The native-backbone arm, stage 2: the real crystal

Stage 1's reference was an arbitrary *predicted* cofold, so its inversions could have
been reference pathologies. Stage 2 (`compute_proteinmpnn_native.py`) replaces it with
the actual **SAbDab crystal complex** per target — threading mutant sequences onto the
crystal variable-domain backbone, conditioned on the crystal antigen. An **alignment
gate** (manifest vs crystal resolved sequence, ≥0.85 identity, contained, gap-free)
protects against mis-registration; it passed 1mhp/1n8z/4fqi and **excluded 3gbn** (light
chain only 0.74 — a gap or divergent chain, refused rather than threaded on a guess).

| target | n | **crystal ρ** | shared (predicted) | per-complex (Arm A) |
|---|---|---|---|---|
| 4fqi | 300 | **+0.591** | +0.445 | −0.079 |
| 1n8z | 300 | −0.158 | −0.441 | −0.200 |
| 1mhp | 30 | −0.075 | −0.186 | −0.037 |

Two clean conclusions:

- **4fqi: ρ=+0.591 on the real crystal — the strongest affinity signal anywhere in the
  arc, and *higher* than the predicted shared backbone.** When the structure is a real
  crystal and the target cooperates, ProteinMPNN inverse-folding likelihood genuinely
  ranks affinity. The lead is real, not an artifact of any predicted reference.
- **1n8z's inversion is real, not a reference artifact.** It stays negative on the
  crystal (−0.158, milder than shared's −0.44 but the same sign). HER2/trastuzumab
  affinity genuinely does *not* track native-likeness — a target/assay property no amount
  of structure quality fixes.

(1mhp is n=30 and underpowered; ignore its sign.) A caveat on the gate: a stricter gate
would have threaded a native backbone for the two influenza-HA targets where it works
best (3gbn, 4fqi), and 3gbn — stage 1's strongest predicted target (+0.56) — is exactly
the one excluded for a light-chain gap, so the clean positive here rests on 4fqi alone.

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
5. **Shared-backbone arm** (`compute_proteinmpnn_shared.py`): best-ipTM cofold per target
   as the fixed reference, mutants threaded by sequence swap (exact-length, zero
   alignment risk).
6. **Crystal arm** (`compute_proteinmpnn_native.py`): SAbDab crystal per target, mutant
   VH/VL threaded onto the crystal variable-domain backbone behind an **alignment gate**
   (≥0.85 identity, contained, gap-free) that skips a target rather than mis-thread.
   One parsing trap caught here and worth recording: ProteinMPNN's own `parse_PDB`
   inserts `-` placeholders for unresolved residues and insertion codes, which
   frame-shifted the alignment and made the gate reject all four clean targets at id≈0.4.
   The fix was to extract a *gap-free* resolved backbone straight from gemmi. The gate
   firing on the buggy parse — rather than threading onto a gapped sequence — is exactly
   what it is for.

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

**3. The crystal confirms it, and rules out "just a predicted-reference artifact."**
Stage 2 (§3.2) replaced the arbitrary predicted reference with the real SAbDab crystal.
On 4fqi the crystal gives **ρ=+0.591** — higher than the predicted backbone and the
strongest affinity signal in the arc — so a real native backbone + ProteinMPNN genuinely
ranks affinity. And 1n8z's inversion *persists on the crystal* (−0.16), proving it is a
real target property, not a reference pathology. The signal is real; so is its
target-specificity. (3gbn, stage 1's strongest predicted target, was excluded by the
alignment gate for a light-chain gap, so the clean crystal positive rests on 4fqi.)

**4. For the funnel (the practical consequence).** The arc's Step-3 picture moves from
"no binding ranker is possible" to "a *fixed-native-backbone* inverse-folding score
ranks affinity for *some* targets (4fqi +0.59) and inverts for others (1n8z)." It is
**still not deployable** as a production ranker — a score that flips sign between targets
will actively mis-rank a new one you cannot pre-classify — so the `design_ranking` funnel
recommendation is unchanged (gate on ipTM + filter_rmsd, diversify by cluster, rank by
free plausibility metrics). What stage 2 adds is *why*: the limit is no longer "no signal
exists" but "we cannot yet predict which targets the signal works on." That is the real
open problem, and it is a better-posed one than where the arc started.

### Limitations

- **The crystal positive rests on one well-powered target.** Of the three that passed
  the gate, 4fqi (n=300) is the clean strong positive; 1n8z inverts; 1mhp is n=30. 3gbn
  — stage 1's strongest — was gate-excluded. So "crystal backbone + ProteinMPNN ranks
  affinity" is demonstrated decisively on 4fqi and is suggestive, not proven, in general.
- **Strong, real target heterogeneity** including the 1n8z inversion that survives to the
  crystal — a funnel cannot absorb a metric that flips sign between targets.
- **We still have not reproduced AbBiBench's *pooled* absolute number** — the arc shows
  the signal is real per-target (4fqi +0.59) but its spread across targets is wider than
  a single published ≈0.30 conveys.
- **Six targets, two of them AlphaSeq** (aayl49/51, a different assay); and the crystal
  arm covers only 3 of 6. Worth remembering before leaning on any aggregate.
- **The open problem is now target-selection, not metric choice.** We cannot yet predict,
  for a new target, whether inverse-folding likelihood will track or invert against
  affinity — that is what a usable Step-3 ranker would require.
