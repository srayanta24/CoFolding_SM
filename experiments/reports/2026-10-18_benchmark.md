# Antibody design benchmark report — 2026-10-18

Aggregates the per-design metrics BoltzGen already computes for each campaign (`data/designs/*/run*/final_ranked_designs/all_designs_metrics.csv`) — no confidence metric here is recomputed, only summarized. Thresholds and their provenance are documented in `experiments/thresholds.py`; the `source` column below says which kind each is — `boltzgen` (read off the tool's own code), `literature` (a published, validated operating point), `readme` (this project's own guidance, written for cofolding) or `heuristic` (invented here, treat as a placeholder).

## Per-campaign summary

### `ctrop2_antibody` — TROP2 (predicted (boltz2))

1000 designs generated, 146 passed BoltzGen's own filters (14.6%).

| metric | mean | median | min | max | pass rate | source |
|---|---|---|---|---|---|---|
| `design_to_target_iptm` | 0.344 | 0.339 | 0.157 | 0.692 | 0.0% | measured |
| `filter_rmsd` | 9.454 | 8.031 | 1.067 | 21.430 | 16.5% | boltzgen |
| `complex_plddt` | 0.705 | 0.705 | 0.648 | 0.772 | 0.0% | measured |
| `liability_num_violations` | 16.707 | 16.000 | 7.000 | 32.000 | 72.2% | heuristic |

### `hel_antibody` — HEL (hen egg lysozyme) (real_pdb (1dpx, apo))

50 designs generated, 2 passed BoltzGen's own filters (4.0%).

| metric | mean | median | min | max | pass rate | source |
|---|---|---|---|---|---|---|
| `design_to_target_iptm` | 0.304 | 0.284 | 0.168 | 0.572 | 0.0% | measured |
| `filter_rmsd` | 11.487 | 12.083 | 1.133 | 17.440 | 4.0% | boltzgen |
| `complex_plddt` | 0.754 | 0.754 | 0.704 | 0.800 | 0.0% | measured |
| `liability_num_violations` | 14.460 | 13.000 | 8.000 | 26.000 | 88.0% | heuristic |

### `trop2_antibody` — TROP2 (real_pdb (7e5n))

50 designs generated, 0 passed BoltzGen's own filters (0.0%).

| metric | mean | median | min | max | pass rate | source |
|---|---|---|---|---|---|---|
| `design_to_target_iptm` | 0.282 | 0.258 | 0.188 | 0.508 | 0.0% | measured |
| `filter_rmsd` | 13.915 | 16.304 | 2.600 | 22.974 | 0.0% | boltzgen |
| `complex_plddt` | 0.702 | 0.699 | 0.675 | 0.748 | 0.0% | measured |
| `liability_num_violations` | 15.820 | 14.000 | 7.000 | 26.000 | 76.0% | heuristic |

## Cross-campaign summary

Across 3 campaigns:

| metric | mean of campaign means | mean pass rate |
|---|---|---|
| `design_to_target_iptm` | 0.310 | 0.0% |
| `filter_rmsd` | 11.618 | 6.8% |
| `complex_plddt` | 0.720 | 0.0% |
| `liability_num_violations` | 15.662 | 78.7% |

---

_Caveat (README.md): these are computational predictions scored by each model's own confidence, not validated binders. Campaign scale here (50-1000 designs) is far below BoltzGen's own guidance for realistic hit-finding (10,000-60,000) — treat pass rates as demo-scale, not representative campaign hit rates._