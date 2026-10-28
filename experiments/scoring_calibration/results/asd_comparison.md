# EXPERIMENT_03 — within-antigen affinity ranking (generated)

`analyze_asd.py` over 1530 entries. Narrative: [EXPERIMENT_03](../EXPERIMENT_03_within_antigen_ranking.md).

Each target is **one antigen, many antibodies, heavy chain only varying**. Affinity is −log KD, higher = tighter, so **every expected correlation is positive**.

## Per target

| target | n | ipTM | pTM | complex pLDDT | confidence_score | ESM-IF1 ll_heavy | ESM-IF1 ll_light | ESM-IF1 ll_antibody | ESM-IF1 ll_antigen |
|---|---|---|---|---|---|---|---|---|---|
| 1mhp | 30 | -0.110 | -0.106 | -0.083 | -0.113 | -0.225 | -0.013 | -0.136 | +0.288 |
| 1n8z | 300 | +0.209 | +0.155 | -0.208 | +0.112 | -0.102 | +0.072 | -0.068 | -0.094 |
| 3gbn | 300 | +0.289 | +0.367 | +0.554 | +0.399 | +0.207 | -0.201 | +0.144 | +0.241 |
| 4fqi | 300 | +0.024 | +0.038 | -0.053 | +0.011 | -0.014 | +0.061 | +0.010 | +0.051 |
| aayl49 | 300 | +0.083 | +0.078 | +0.051 | +0.053 | +0.340 | -0.162 | +0.213 | +0.052 |
| aayl51 | 300 | +0.042 | +0.055 | +0.045 | +0.053 | +0.210 | +0.084 | +0.206 | +0.100 |

## Pooled (z-scored within target)

| metric | ρ | n |
|---|---|---|
| ipTM | +0.133 | 1530 |
| pTM | +0.141 | 1530 |
| complex pLDDT | +0.064 | 1530 |
| confidence_score | +0.113 | 1530 |
| ESM-IF1 ll_heavy | +0.111 | 1530 |
| ESM-IF1 ll_light | -0.035 | 1530 |
| ESM-IF1 ll_antibody | +0.082 | 1530 |
| *ESM-IF1 ll_antigen* **(null control)** | +0.070 | 1530 |

> **The null control scores +0.070 pooled.** The antigen is identical across every entry within a target, so it cannot carry affinity information. Where it correlates, something target-specific is leaking into every metric that reads the structure, and those columns cannot be interpreted at face value.

AbBiBench's published baselines on this same data: ProteinMPNN ≈0.3, ESM-IF1 ≈0.28.

## Heavy-chain diversity

| target | length | variable positions | mean pairwise identity |
|---|---|---|---|
| 1mhp | 118 | 14 | 0.9793 |
| 1n8z | 121 | 23 | 0.9336 |
| 3gbn | 121 | 11 | 0.9556 |
| 4fqi | 121 | 16 | 0.9348 |
| aayl49 | 118 | 33 | 0.9402 |
| aayl51 | 119 | 36 | 0.9562 |

Included to test the hypothesis that a null target's variants are simply too similar to resolve. It **refutes** it: all three targets are comparably diverse, and the target with the fewest variable positions shows the strongest apparent signal.

