# Antibody design benchmark report — 2026-08-15

Aggregates the per-design metrics BoltzGen already computes for each campaign (`data/designs/*/run*/final_ranked_designs/all_designs_metrics.csv`) — no confidence metric here is recomputed, only summarized. Thresholds and their provenance (README-sourced vs. heuristic) are documented in `experiments/thresholds.py`.

## Per-campaign summary

### `ctrop2_antibody` — TROP2 (predicted (boltz2))

1000 designs generated, 146 passed BoltzGen's own filters (14.6%).

| metric | mean | median | min | max | pass rate | source |
|---|---|---|---|---|---|---|
| `design_to_target_iptm` | 0.344 | 0.339 | 0.157 | 0.692 | 0.0% | readme |
| `filter_rmsd` | 9.454 | 8.031 | 1.067 | 21.430 | 23.0% | readme |
| `complex_plddt` | 0.705 | 0.705 | 0.648 | 0.772 | 0.0% | heuristic |
| `liability_num_violations` | 16.707 | 16.000 | 7.000 | 32.000 | 72.2% | heuristic |

### `hel_antibody` — HEL (hen egg lysozyme) (real_pdb (1dpx, apo))

50 designs generated, 2 passed BoltzGen's own filters (4.0%).

| metric | mean | median | min | max | pass rate | source |
|---|---|---|---|---|---|---|
| `design_to_target_iptm` | 0.304 | 0.284 | 0.168 | 0.572 | 0.0% | readme |
| `filter_rmsd` | 11.487 | 12.083 | 1.133 | 17.440 | 4.0% | readme |
| `complex_plddt` | 0.754 | 0.754 | 0.704 | 0.800 | 0.0% | heuristic |
| `liability_num_violations` | 14.460 | 13.000 | 8.000 | 26.000 | 88.0% | heuristic |

### `trop2_antibody` — TROP2 (real_pdb (7e5n))

50 designs generated, 0 passed BoltzGen's own filters (0.0%).

| metric | mean | median | min | max | pass rate | source |
|---|---|---|---|---|---|---|
| `design_to_target_iptm` | 0.282 | 0.258 | 0.188 | 0.508 | 0.0% | readme |
| `filter_rmsd` | 13.915 | 16.304 | 2.600 | 22.974 | 4.0% | readme |
| `complex_plddt` | 0.702 | 0.699 | 0.675 | 0.748 | 0.0% | heuristic |
| `liability_num_violations` | 15.820 | 14.000 | 7.000 | 26.000 | 76.0% | heuristic |

## Cross-campaign summary

Across 3 campaigns:

| metric | mean of campaign means | mean pass rate |
|---|---|---|
| `design_to_target_iptm` | 0.310 | 0.0% |
| `filter_rmsd` | 11.618 | 10.3% |
| `complex_plddt` | 0.720 | 0.0% |
| `liability_num_violations` | 15.662 | 78.7% |

## Known-binder reference baselines

Real, experimentally known antibody-antigen complexes scored through the same cofolding pipeline (`scripts/run_design.py`) used to externally validate designs — calibrates what a genuine true positive scores on this pipeline, since BoltzGen's own metrics are self-reported and generic literature ipTM thresholds may not transfer directly. This is NOT a structural/epitope-overlap check (see experiments/README.md for that caveat) — a design can score well here while binding a different surface than the reference complex.

### `hel`

- **boltz2**: [{'confidence_score': 0.8766084909439087, 'ptm': 0.7380221486091614, 'iptm': 0.6629744172096252}]
- **openfold3**: [{'ptm': 0.718779, 'iptm': 0.624425, 'avg_plddt': 91.12455, 'sample_ranking_score': 0.643296}, {'ptm': 0.718801, 'iptm': 0.626183, 'avg_plddt': 91.085014, 'sample_ranking_score': 0.644706}, {'ptm': 0.642278, 'iptm': 0.581922, 'avg_plddt': 90.660332, 'sample_ranking_score': 0.593993}, {'ptm': 0.735088, 'iptm': 0.663813, 'avg_plddt': 91.264633, 'sample_ranking_score': 0.678068}, {'ptm': 0.740372, 'iptm': 0.664136, 'avg_plddt': 91.374405, 'sample_ranking_score': 0.679383}]

---

_Caveat (README.md): these are computational predictions scored by each model's own confidence, not validated binders. Campaign scale here (50-1000 designs) is far below BoltzGen's own guidance for realistic hit-finding (10,000-60,000) — treat pass rates as demo-scale, not representative campaign hit rates._