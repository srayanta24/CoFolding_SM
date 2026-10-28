# Paired within-parent test (generated)

`analyze_paired.py` over 247 mutant complexes on **28 parent complexes**. Narrative: [EXPERIMENT_02](../EXPERIMENT_02_alternative_metrics.md).

For each parent holding both a binding-retained and a binding-destroyed point mutant, does the metric score the retained one higher? Sign test over parents, exact one-sided binomial p. This controls for complex identity completely and is the only column EXPERIMENT_02 found trustworthy.

Bonferroni α over 11 metrics = **0.0045**.

| metric | all | p | original 17 | **new 11 only** | p |
|---|---|---|---|---|---|
| **packstat** | 23/28 | 0.0005 | 14/17 | 8/11 | 0.113 |
| -dG/dSASA | 18/28 | 0.0925 | 12/17 | 6/11 | 0.500 |
| -dG_separated | 18/28 | 0.0925 | 12/17 | 6/11 | 0.500 |
| shape complementarity | 16/28 | 0.2858 | 13/17 | 5/11 | 0.726 |
| buried SASA | 16/28 | 0.2858 | 13/17 | 4/11 | 0.887 |
| interface H-bonds | 13/28 | 0.7142 | 10/17 | 5/11 | 0.726 |
| ipTM | 14/28 | 0.5747 | 9/17 | 5/11 | 0.726 |
| ipSAE_min | 7/28 | 0.9981 | 4/17 | 3/11 | 0.967 |
| ESM-IF1 ll_antibody | 13/28 | 0.7142 | 8/17 | 4/11 | 0.887 |
| AntiFold ll (H+L) | 16/28 | 0.2858 | 8/17 | 8/11 | 0.113 |
| AntiFold ll (+antigen) | 16/28 | 0.2858 | 9/17 | 7/11 | 0.274 |
| *ESM-IF1 ll_antigen* (null control) | 17/28 | 0.1725 | 9/17 | 7/11 | 0.274 |

**Survives Bonferroni: packstat.**

The `all` column is **not** an independent replication of `original 17` — those 17 parents are inside the 28. The `new 11 only` column is the independent test; it is small, so a real metric should agree in *direction* there even where it misses significance.

> The null control must land at chance. Within a parent the antigen is identical between the retained and destroyed mutant (the mutation is in the heavy or light chain), so it cannot discriminate. It changed the conclusion in both EXPERIMENT_02 and EXPERIMENT_03.

## Why the Bonferroni column matters here

This benchmark taught the lesson directly. At 17 parents, shape complementarity and buried SASA were both 13/17 (p=0.025) and were reported as significant findings. At 28 parents both fell to 16/28, scoring 5/11 and 4/11 on the new parents — chance. An uncorrected p≈0.025 across seven metrics was never a finding.

