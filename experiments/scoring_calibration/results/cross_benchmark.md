# Cross-benchmark agreement (generated)

`analyze_crossbench.py` over the two generated benchmark JSONs. Narrative: [PLAN.md](../PLAN.md) §11.

Each experiment asked *does metric X work on benchmark Y*. This asks the question that decides what the arc concluded: **is a metric that wins on one benchmark more likely to win on the other?**

| metric | paired | p | beats null | ASD ρ | beats null |
|---|---|---|---|---|---|
| packstat | 23/28 | 0.0005 | ✓ | +0.058 | ✗ |
| -dG/dSASA | 18/28 | 0.0925 | ✓ | +0.100 | ✓ |
| -dG_separated | 18/28 | 0.0925 | ✓ | +0.125 | ✓ |
| *ESM-IF1 ll_antigen* **(null)** | 17/28 | 0.1725 | — | +0.070 | — |
| shape complementarity | 16/28 | 0.2858 | ✗ | +0.112 | ✓ |
| buried SASA | 16/28 | 0.2858 | ✗ | +0.065 | ✗ |
| ipTM | 14/28 | 0.5747 | ✗ | +0.133 | ✓ |
| interface H-bonds | 13/28 | 0.7142 | ✗ | +0.039 | ✗ |
| ESM-IF1 ll_antibody | 13/28 | 0.7142 | ✗ | +0.082 | ✓ |

Null control: **0.61** win rate on paired, **+0.070** on ASD. `✓` means the metric beat the null *on that benchmark*.

## Do the two orderings agree?

Rank correlation between the two columns, over the 8 non-null shared metrics: **ρ = +0.087**.

The best metric on one benchmark is near-worst on the other, in both directions: `packstat` tops the paired test and sits below the null on ASD, while `ipTM` is at chance on the paired test and near the top on ASD.

## Is the overlap more than chance?

3 of 8 metrics beat the paired null; 5 of 8 beat the ASD null; **2 beat both** (-dG/dSASA, -dG_separated).

Counting survivors means nothing without knowing how many luck produces. Under independence the overlap is hypergeometric:

| overlap | P |
|---|---|
| 0 | 0.018 |
| 1 | 0.268 |
| 2 ← observed | 0.536 |
| 3 | 0.179 |

- expected overlap by chance: **1.88**
- observed: **2**
- P(overlap ≥ observed) = **0.714**

So **no metric has demonstrated cross-benchmark signal**. The two metrics that survive both are precisely what a no-signal world produces.

> This is the null-control discipline applied one level up. Within a benchmark, a metric is compared against a quantity that cannot carry the signal. Across benchmarks, a *count of survivors* is compared against the count chance produces. Both answer the same question: is this more than nothing?

## What this does and does not establish

**Does**: no metric among those tested ranks designs in a way that survives a change of question. The remedy of picking a better existing scalar is closed, because the shortlist has been enumerated and none of it transfers.

**Does not**: prove no cheap metric exists. Four families were tested (BoltzGen confidences, ipSAE, Rosetta, inverse-folding likelihoods), not all possible ones. And a metric *may* legitimately differ between these two benchmarks, since one contrasts intact against wrecked interfaces and the other ranks functional binders — the chance calculation is what rules out reading the survivors as evidence, not the disagreement by itself.

