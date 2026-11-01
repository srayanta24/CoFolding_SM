# EXPERIMENT_03 — within-antigen affinity ranking (generated)

`analyze_asd.py` over 1530 entries. Narrative: [EXPERIMENT_03](../EXPERIMENT_03_within_antigen_ranking.md).

Each target is **one antigen, many antibodies, heavy chain only varying**. Affinity is −log KD, higher = tighter, so **every expected correlation is positive**.

## Per target

| target | n | ipTM | pTM | complex pLDDT | confidence_score | ESM-IF1 ll_heavy | ESM-IF1 ll_light | ESM-IF1 ll_antibody | packstat | -dG/dSASA | -dG_separated | shape complementarity | buried SASA | interface H-bonds | ProteinMPNN ll_heavy | ProteinMPNN ll_antibody | ESM-IF1 ll_antigen | ProteinMPNN ll_antigen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1mhp | 30 | -0.110 | -0.106 | -0.083 | -0.113 | -0.225 | -0.013 | -0.136 | -0.055 | -0.097 | -0.091 | -0.097 | +0.026 | +0.047 | -0.054 | -0.037 | +0.288 | +0.120 |
| 1n8z | 300 | +0.209 | +0.155 | -0.208 | +0.112 | -0.102 | +0.072 | -0.068 | -0.013 | -0.171 | +0.009 | -0.069 | +0.149 | +0.069 | -0.217 | -0.200 | -0.094 | -0.125 |
| 3gbn | 300 | +0.289 | +0.367 | +0.554 | +0.399 | +0.207 | -0.201 | +0.144 | +0.108 | +0.521 | +0.525 | +0.401 | +0.337 | +0.203 | +0.224 | +0.190 | +0.241 | +0.433 |
| 4fqi | 300 | +0.024 | +0.038 | -0.053 | +0.011 | -0.014 | +0.061 | +0.010 | +0.182 | +0.219 | +0.149 | +0.117 | +0.004 | -0.143 | -0.080 | -0.079 | +0.051 | -0.028 |
| aayl49 | 300 | +0.083 | +0.078 | +0.051 | +0.053 | +0.340 | -0.162 | +0.213 | -0.008 | -0.006 | -0.013 | +0.085 | -0.014 | +0.014 | +0.326 | +0.282 | +0.052 | -0.073 |
| aayl51 | 300 | +0.042 | +0.055 | +0.045 | +0.053 | +0.210 | +0.084 | +0.206 | +0.050 | +0.032 | +0.002 | +0.075 | -0.049 | +0.059 | +0.330 | +0.274 | +0.100 | +0.087 |

## Pooled (z-scored within target)

`ρ partial` removes the null control from both sides: it is what remains after the target-specific channel the null reads is taken out. A metric whose raw ρ is no better than the null's has not been shown to read affinity at all.

| metric | ρ | ρ partial (null removed) | n |
|---|---|---|---|
| ipTM | +0.133 | +0.124 | 1530 |
| pTM | +0.141 | +0.131 | 1530 |
| complex pLDDT — **below null** | +0.064 | +0.056 | 1530 |
| confidence_score | +0.113 | +0.105 | 1530 |
| ESM-IF1 ll_heavy | +0.111 | +0.105 | 1530 |
| ESM-IF1 ll_light — **below null** | -0.035 | -0.040 | 1530 |
| ESM-IF1 ll_antibody | +0.082 | +0.075 | 1530 |
| packstat — **below null** | +0.058 | +0.054 | 1530 |
| -dG/dSASA | +0.100 | +0.095 | 1530 |
| -dG_separated | +0.125 | +0.119 | 1530 |
| shape complementarity | +0.112 | +0.104 | 1530 |
| buried SASA — **below null** | +0.065 | +0.062 | 1530 |
| interface H-bonds — **below null** | +0.039 | +0.035 | 1530 |
| ProteinMPNN ll_heavy | +0.097 | +0.090 | 1530 |
| ProteinMPNN ll_antibody | +0.080 | +0.075 | 1530 |
| *ESM-IF1 ll_antigen* **(null control)** | +0.070 | — | 1530 |
| *ProteinMPNN ll_antigen* **(null control)** | +0.059 | — | 1530 |

> **The null control scores +0.070 pooled.** The antigen is identical across every entry within a target, so it cannot carry affinity information. Where it correlates, something target-specific is leaking into every metric that reads the structure, and those columns cannot be interpreted at face value.

AbBiBench's published baselines on this same data: ProteinMPNN ≈0.3, ESM-IF1 ≈0.28.

## Does packstat recover on the extremes?

packstat wins the paired benchmark 23/28 (p=0.0005) but lands **below the null control** above. The natural rescue: the paired test contrasts binding-*retained* against binding-*destroyed* mutants, whereas every entry here is a functional binder differing only in affinity. If packstat detects gross interface damage rather than affinity gradation, it should recover when only the extremes are compared.

Scored exactly as the paired benchmark is — a sign test over groups — at three cut depths. Deltas are in within-target standard deviations.

**Top vs bottom 10% of affinity**

| metric | 1mhp | 1n8z | 3gbn | 4fqi | aayl49 | aayl51 | wins | p |
|---|---|---|---|---|---|---|---|---|
| ipTM | -0.36 | +0.52 | +1.05 | +0.43 | +0.49 | +0.10 | 5/6 | 0.109 |
| pTM | -0.24 | +0.41 | +1.24 | +0.52 | +0.51 | +0.14 | 5/6 | 0.109 |
| complex pLDDT | -0.79 | -0.41 | +1.61 | -0.18 | -0.15 | +0.12 | 2/6 | 0.891 |
| confidence_score | -0.58 | +0.26 | +1.27 | +0.32 | -0.08 | +0.13 | 4/6 | 0.344 |
| ESM-IF1 ll_heavy | +0.72 | -0.37 | +0.89 | +0.13 | +1.08 | +0.59 | 5/6 | 0.109 |
| ESM-IF1 ll_light | +0.16 | +0.16 | -0.79 | +0.22 | -0.59 | +0.18 | 4/6 | 0.344 |
| ESM-IF1 ll_antibody | +0.64 | -0.28 | +0.46 | +0.18 | +0.56 | +0.55 | 5/6 | 0.109 |
| packstat | -0.26 | +0.04 | +0.21 | +0.60 | +0.17 | +0.18 | 5/6 | 0.109 |
| -dG/dSASA | +0.59 | -0.36 | +1.57 | +0.79 | +0.06 | +0.15 | 5/6 | 0.109 |
| -dG_separated | +0.18 | +0.15 | +1.57 | +0.60 | -0.13 | -0.11 | 4/6 | 0.344 |
| shape complementarity | +0.83 | +0.06 | +1.17 | +0.14 | +0.28 | +0.21 | 6/6 | 0.016 |
| buried SASA | -0.62 | +0.43 | +0.15 | +0.10 | -0.28 | -0.33 | 3/6 | 0.656 |
| interface H-bonds | +0.14 | +0.39 | +0.34 | -0.51 | -0.05 | +0.22 | 4/6 | 0.344 |
| ProteinMPNN ll_heavy | +0.09 | -0.65 | +0.72 | -0.12 | +0.81 | +0.89 | 4/6 | 0.344 |
| ProteinMPNN ll_antibody | +0.47 | -0.51 | +0.40 | -0.01 | +0.61 | +0.68 | 4/6 | 0.344 |
| *ESM-IF1 ll_antigen* **(null control)** | +0.87 | -0.24 | +0.62 | +0.33 | +0.36 | +0.11 | 5/6 | 0.109 |
| *ProteinMPNN ll_antigen* **(null control)** | +0.50 | -0.56 | +1.29 | +0.29 | -0.00 | +0.24 | 4/6 | 0.344 |

**Top vs bottom 25% of affinity**

| metric | 1mhp | 1n8z | 3gbn | 4fqi | aayl49 | aayl51 | wins | p |
|---|---|---|---|---|---|---|---|---|
| ipTM | +0.04 | +0.48 | +0.72 | +0.12 | +0.25 | +0.16 | 6/6 | 0.016 |
| pTM | +0.07 | +0.39 | +0.85 | +0.14 | +0.26 | +0.20 | 6/6 | 0.016 |
| complex pLDDT | -0.38 | -0.43 | +1.25 | -0.11 | +0.26 | +0.10 | 3/6 | 0.656 |
| confidence_score | -0.14 | +0.22 | +0.91 | +0.08 | +0.28 | +0.12 | 5/6 | 0.109 |
| ESM-IF1 ll_heavy | -0.05 | -0.25 | +0.55 | -0.10 | +0.95 | +0.54 | 3/6 | 0.656 |
| ESM-IF1 ll_light | +0.40 | +0.31 | -0.50 | +0.16 | -0.53 | +0.18 | 4/6 | 0.344 |
| ESM-IF1 ll_antibody | +0.14 | -0.12 | +0.28 | -0.03 | +0.49 | +0.51 | 4/6 | 0.344 |
| packstat | -0.53 | -0.06 | +0.27 | +0.41 | +0.01 | +0.16 | 4/6 | 0.344 |
| -dG/dSASA | -0.28 | -0.38 | +1.07 | +0.51 | -0.02 | +0.12 | 3/6 | 0.656 |
| -dG_separated | -0.46 | +0.04 | +1.20 | +0.26 | -0.10 | +0.04 | 4/6 | 0.344 |
| shape complementarity | -0.65 | -0.06 | +0.97 | +0.23 | +0.27 | +0.28 | 4/6 | 0.344 |
| buried SASA | -0.51 | +0.35 | +0.40 | -0.12 | -0.15 | -0.06 | 2/6 | 0.891 |
| interface H-bonds | +0.12 | +0.25 | +0.28 | -0.46 | +0.03 | +0.16 | 5/6 | 0.109 |
| ProteinMPNN ll_heavy | +0.05 | -0.57 | +0.42 | -0.21 | +0.88 | +0.75 | 4/6 | 0.344 |
| ProteinMPNN ll_antibody | +0.20 | -0.47 | +0.37 | -0.17 | +0.75 | +0.66 | 4/6 | 0.344 |
| *ESM-IF1 ll_antigen* **(null control)** | +0.81 | -0.30 | +0.62 | +0.20 | +0.05 | +0.19 | 5/6 | 0.109 |
| *ProteinMPNN ll_antigen* **(null control)** | +0.62 | -0.21 | +1.06 | +0.05 | -0.20 | +0.13 | 4/6 | 0.344 |

**Top vs bottom 50% of affinity**

| metric | 1mhp | 1n8z | 3gbn | 4fqi | aayl49 | aayl51 | wins | p |
|---|---|---|---|---|---|---|---|---|
| ipTM | -0.49 | +0.29 | +0.50 | +0.17 | +0.15 | +0.02 | 5/6 | 0.109 |
| pTM | -0.54 | +0.24 | +0.58 | +0.17 | +0.16 | +0.05 | 5/6 | 0.109 |
| complex pLDDT | -0.37 | -0.33 | +0.82 | +0.04 | +0.17 | +0.14 | 4/6 | 0.344 |
| confidence_score | -0.48 | +0.10 | +0.61 | +0.15 | +0.18 | +0.13 | 5/6 | 0.109 |
| ESM-IF1 ll_heavy | -0.87 | -0.02 | +0.38 | -0.03 | +0.50 | +0.48 | 3/6 | 0.656 |
| ESM-IF1 ll_light | -0.26 | +0.08 | -0.26 | +0.05 | -0.29 | +0.07 | 3/6 | 0.656 |
| ESM-IF1 ll_antibody | -0.80 | +0.01 | +0.23 | -0.01 | +0.25 | +0.40 | 4/6 | 0.344 |
| packstat | +0.08 | -0.01 | +0.10 | +0.19 | -0.02 | +0.07 | 4/6 | 0.344 |
| -dG/dSASA | -0.40 | -0.27 | +0.63 | +0.45 | -0.13 | +0.16 | 3/6 | 0.656 |
| -dG_separated | -0.17 | -0.03 | +0.78 | +0.37 | -0.19 | +0.04 | 3/6 | 0.656 |
| shape complementarity | -0.24 | -0.17 | +0.59 | +0.28 | +0.12 | +0.06 | 4/6 | 0.344 |
| buried SASA | +0.32 | +0.16 | +0.40 | +0.07 | -0.11 | -0.10 | 4/6 | 0.344 |
| interface H-bonds | -0.19 | +0.04 | +0.33 | -0.16 | -0.06 | +0.14 | 3/6 | 0.656 |
| ProteinMPNN ll_heavy | -0.52 | -0.36 | +0.35 | -0.07 | +0.52 | +0.72 | 3/6 | 0.656 |
| ProteinMPNN ll_antibody | -0.47 | -0.32 | +0.32 | -0.08 | +0.44 | +0.59 | 3/6 | 0.656 |
| *ESM-IF1 ll_antigen* **(null control)** | +0.39 | -0.21 | +0.38 | +0.18 | +0.12 | +0.27 | 5/6 | 0.109 |
| *ProteinMPNN ll_antigen* **(null control)** | +0.04 | -0.23 | +0.65 | -0.01 | -0.09 | +0.17 | 3/6 | 0.656 |

**It does not rescue packstat, and the reason is the null control.** With only 6 targets the sign test has so little power that the null passes it too — at the decile and quartile cuts the null wins 5/6, with per-target effect sizes as large as the real metrics. No column in this table can be read as evidence. This is the same lesson the paired benchmark taught when 13/17 collapsed at 28 parents: a sign test over a handful of groups is not a finding, and the null control is what reveals it.

The honest statement is therefore **not** "packstat only detects gross damage" — that remains the most plausible mechanism, but this benchmark cannot establish it. What it does establish is the negative: packstat's 23/28 does not transfer to graded affinity ranking.

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

