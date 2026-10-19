# EXPERIMENT_02 — metric comparison (generated)

`analyze_metrics.py` over 332 entries joined across all four result tables. Narrative and interpretation: [EXPERIMENT_02](../EXPERIMENT_02_alternative_metrics.md).

All metrics oriented so **higher = predicted better binder** (Rosetta energy terms are negated).

## Metric comparison

| metric | cross-complex ρ | unpaired AUC | **paired** | p |
|---|---|---|---|---|
| ipTM (baseline) | +0.134 | 0.634 | 9/17 | 0.500 |
| ipSAE_min | +0.198 | 0.626 | 8/17 | 0.685 |
| -dG/dSASA (Rosetta) | +0.179 | 0.634 | 12/17 | 0.072 |
| -dG_separated | +0.145 | 0.644 | 12/17 | 0.072 |
| shape complementarity | +0.054 | 0.541 | 13/17 | 0.025 **←** |
| packstat | +0.003 | 0.624 | 14/17 | 0.006 **←** |
| buried SASA | +0.048 | 0.587 | 13/17 | 0.025 **←** |
| interface H-bonds | +0.049 | 0.525 | 10/17 | 0.315 |
| -unsat H-bonds | +0.131 | 0.447 | 6/17 | 0.928 |
| ESM-IF1 ll_antibody | -0.037 | 0.634 | 8/17 | 0.685 |
| ESM-IF1 ll_heavy | -0.014 | 0.602 | 8/17 | 0.685 |
| ESM-IF1 ll_light | -0.061 | 0.606 | 9/17 | 0.500 |
| *ESM-IF1 ll_antigen* (null control) | +0.341 | 0.396 | 9/17 | 0.500 |

> **Only the paired column is valid.** The first two are confounded — see diagnostics. The null control landing at chance in the paired column is what makes that column believable.

## Diagnostics — why two columns were discarded

**1. Unpaired AUC is confounded by arm composition.** `matched_positive` draws on 27 parent complexes, `hard_negative` on 24, and only **17** are shared — 17 parents appear in just one arm. The comparison therefore partly measures which complexes landed where.

**2. Cross-complex ρ is confounded by antigen identity.** 100 entries span 89 distinct antigens. log10(Kd) spread *within* a repeated antigen is **1.731** against a total spread of **7.974**, so roughly 78% of affinity variance is between antigens. Any antigen-only feature correlates with Kd spuriously — which is exactly what the null control's positive ρ demonstrates.

**3. Complex size was checked and ruled out.** log10(Kd) vs total residues ρ=+0.074, vs antigen residues +0.097, vs buried SASA +0.048. All negligible. EXPERIMENT_01 flagged size as the likely confound in its limitations; it is not. Antigen identity is.

## Cost (measured)

| tier | n | mean | total |
|---|---|---|---|
| Rosetta (CPU, 16 workers) | 332 | 22.7 min | 125.5 CPU-h |
| ESM-IF1 (CPU) | 332 | 7.6 s | 0.70 h |
| ipSAE | 332 | ~0 | re-reads saved PAE matrices |

For scale: generating one design costs ~3 min of GPU, so Rosetta rescoring is roughly **5× more expensive than generation** — top-N only, never funnel-wide.

