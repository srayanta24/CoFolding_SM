# EXPERIMENT_03 — within-antigen affinity ranking (generated)

`analyze_asd.py` over 330 entries. Narrative: [EXPERIMENT_03](../EXPERIMENT_03_within_antigen_ranking.md).

Each target is **one antigen, many antibodies, heavy chain only varying**. Affinity is −log KD, higher = tighter, so **every expected correlation is positive**.

## Per target

| target | n | ipTM | pTM | complex pLDDT | confidence_score | ESM-IF1 ll_heavy | ESM-IF1 ll_light | ESM-IF1 ll_antibody | ESM-IF1 ll_antigen |
|---|---|---|---|---|---|---|---|---|---|
| 1n8z | 110 | +0.208 | +0.211 | -0.291 | +0.092 | -0.067 | +0.198 | +0.013 | -0.056 |
| 3gbn | 110 | +0.273 | +0.349 | +0.554 | +0.392 | +0.194 | -0.184 | +0.141 | +0.284 |
| 4fqi | 110 | -0.009 | +0.000 | -0.117 | -0.031 | +0.114 | +0.096 | +0.132 | +0.079 |

## Pooled (z-scored within target)

| metric | ρ | n |
|---|---|---|
| ipTM | +0.194 | 330 |
| pTM | +0.204 | 330 |
| complex pLDDT | +0.060 | 330 |
| confidence_score | +0.168 | 330 |
| ESM-IF1 ll_heavy | +0.078 | 330 |
| ESM-IF1 ll_light | +0.018 | 330 |
| ESM-IF1 ll_antibody | +0.083 | 330 |
| *ESM-IF1 ll_antigen* **(null control)** | +0.105 | 330 |

> **The null control scores +0.105 pooled.** The antigen is identical across every entry within a target, so it cannot carry affinity information. Where it correlates, something target-specific is leaking into every metric that reads the structure, and those columns cannot be interpreted at face value.

AbBiBench's published baselines on this same data: ProteinMPNN ≈0.3, ESM-IF1 ≈0.28.

## Heavy-chain diversity

| target | length | variable positions | mean pairwise identity |
|---|---|---|---|
| 1n8z | 121 | 21 | 0.9307 |
| 3gbn | 121 | 11 | 0.9563 |
| 4fqi | 121 | 16 | 0.9335 |

Included to test the hypothesis that a null target's variants are simply too similar to resolve. It **refutes** it: all three targets are comparably diverse, and the target with the fewest variable positions shows the strongest apparent signal.

