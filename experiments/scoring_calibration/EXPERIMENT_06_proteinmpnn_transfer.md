# Experiment 06 — Does a stronger inverse-folding scorer recover the published baseline?

Status: **complete · 2026-11-01**

Generated numbers: [`results/asd_comparison.md`](results/asd_comparison.md) /
[`.json`](results/asd_comparison.json), regenerable with `analyze_asd.py`.
Scores produced by `compute_proteinmpnn.py`.

**One-line answer: no.** ProteinMPNN ranks ASD affinity at ρ = +0.097 pooled — slightly
*below* ESM-IF1's +0.111, ~3× under the published ≈0.30, and barely above its own null.
The metric is not the lever; §11.1's conclusion stands.

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

**1. The metric is not the lever.** A stronger, differently-trained inverse-folding
model reproduces ESM-IF1's weak result almost exactly (+0.097 vs +0.111), nowhere near
the published 0.30. Of the two candidate explanations for the gap to AbBiBench — the
scorer or the structures — **the scorer is now ruled out.** §11.1 stands: swapping in a
better inverse-folding model buys the funnel nothing, and there is still no validated
binding ranker.

**2. What this does not settle.** We scored each antibody against *its own per-complex
predicted backbone*, which is directly comparable to ESM-IF1 but is **not** AbBiBench's
protocol (one native crystal backbone per target, threaded mutants). So the low result
has two non-exclusive causes this arm cannot separate: (a) our predicted structures are
too noisy to rank from, or (b) scoring a sequence against the backbone folded from it is
partly circular and washes out the mutation signal. Both are *structure/protocol*
problems, not metric problems.

**3. The clean follow-up is a native-backbone arm.** Fetch the crystal complex for the
four PDB-backed targets, thread the mutant antibody sequences onto it, re-score with
ProteinMPNN. If that reaches ≈0.30, the bottleneck is our predicted structures and the
programme should target structure quality; if it stays ≈0.10, the published baseline
simply does not transfer to this benchmark. Not run here — it needs a crystal fetch and
sequence threading, i.e. a genuinely new benchmark rather than a re-score.

**4. For the funnel (the practical consequence).** The ProteinMPNN check was the lever
that could have added a binding-aware Step-3 ranker. It did not. So the funnel
recommendation is unchanged: gate on ipTM + filter_rmsd (plausibility), diversify by
cluster, and rank within cluster by the free plausibility metrics — no inverse-folding
scorer, however strong, earns a place as a binding ranker on current evidence.

### Limitations

- **Predicted, per-complex backbones** (see finding 2) — the central confound, and the
  reason a *low* result is less conclusive than a high one would have been.
- **We did not reproduce AbBiBench's absolute number** on *any* backbone, so "0.30" is
  taken from their paper, not verified in our hands. The native-backbone arm would also
  serve as that verification.
- **Six targets, two of them AlphaSeq.** The only targets where inverse folding worked
  are the two whose labels are AlphaSeq (a different assay), which is worth remembering
  before leaning on them.
