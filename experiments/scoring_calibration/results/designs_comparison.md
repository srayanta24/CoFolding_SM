# EXPERIMENT_05 — packstat on real designs (generated)

`analyze_designs.py`. Narrative: [EXPERIMENT_05](../EXPERIMENT_05_design_transfer.md).

Within each target, does the metric separate designs whose refold matches the designed backbone (`filter_rmsd` ≤ 2.5) from those whose refold does not? Reported as AUC — P(random passer outscores a random non-passer). 0.5 is chance.

Pre-registered: **PASS** = AUC ≥ 0.65 on every target *and* above the null on every target; **WEAK** = ≥ 0.6 on the same terms; **FAIL** otherwise, including "works on one target only".

> **1 further target withheld.** One design pool is gitignored along with its campaign at the user's request, so its numbers are not in this report. **The verdict column below does incorporate it**, which means the verdicts rest partly on data that is not in this repository and cannot be checked from it. Stated here rather than buried.

## ctrop2 — 655 designs (165 pass / 490 fail)

| metric | AUC | p | median pass | median fail | ρ vs −rmsd |
|---|---|---|---|---|---|
| ipTM (free) | 0.708 | 0.00e+00 | 0.3893 | 0.3352 | +0.398 |
| delta_sasa_refolded (free) | 0.675 | 0.00e+00 | 1207.5527 | 1087.9507 | +0.291 |
| -dG_separated | 0.666 | 0.00e+00 | 60.1766 | 54.1192 | +0.274 |
| buried SASA | 0.646 | 0.00e+00 | 2631.9 | 2405.53 | +0.223 |
| interface H-bonds | 0.615 | 9.00e-06 | 14.0 | 12.0 | +0.197 |
| -dG/dSASA | 0.584 | 1.30e-03 | 2.3192 | 2.2314 | +0.146 |
| ESM-IF1 ll_antibody | 0.551 | 4.90e-02 | -1.3232 | -1.3381 | +0.066 |
| ESM-IF1 ll_heavy | 0.548 | 6.72e-02 | -1.3122 | -1.3426 | +0.069 |
| *ESM-IF1 ll_antigen* **(null)** | 0.512 | 6.35e-01 | -1.4794 | -1.4772 | +0.040 |
| packstat | 0.508 | 7.58e-01 | 0.6281 | 0.6265 | -0.018 |
| shape complementarity | 0.495 | 8.44e-01 | 0.5874 | 0.5865 | +0.007 |
| pTM (free) | 0.489 | 6.74e-01 | 0.705 | 0.705 | -0.052 |

## Verdicts

Verdict and AUC below are computed on the **target** only, so every number here is checkable from this repository. The withheld target contributes one bit — *did it land on the same side of chance* — and nothing else; its magnitudes are in the gitignored `designs_*` TSVs. The stronger pre-registered two-target verdict (which uses those magnitudes) is in [EXPERIMENT_05](../EXPERIMENT_05_design_transfer.md) §3.
| metric | verdict (tracked) | AUC | beats null | withheld agrees? |
|---|---|---|---|---|
| ipTM (free) | **PASS** | 0.708 | yes | yes |
| delta_sasa_refolded (free) | **PASS** | 0.675 | yes | yes |
| -dG_separated | **PASS** | 0.666 | yes | yes |
| buried SASA | **WEAK** | 0.646 | yes | yes |
| interface H-bonds | **WEAK** | 0.615 | yes | yes |
| -dG/dSASA | **FAIL** | 0.584 | yes | yes |
| ESM-IF1 ll_antibody | **FAIL** | 0.551 | yes | no |
| ESM-IF1 ll_heavy | **FAIL** | 0.548 | yes | no |
| packstat | **FAIL** | 0.508 | no | no |
| shape complementarity | **FAIL** | 0.495 | no | yes |
| pTM (free) | **FAIL** | 0.489 | no | yes |

`ρ vs −rmsd` is **secondary and biased**: negatives were sampled stratified by `filter_rmsd`, which distorts its marginal distribution. The AUC column is the pre-registered test.

> `filter_rmsd` is **structural self-consistency, not binding.** Nothing here shows a metric picks binders — there are no binding labels for designs anywhere in this project. See [PLAN.md](../PLAN.md) §9.

