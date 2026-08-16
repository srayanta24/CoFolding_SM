# abdesign_db — AbDesign DB

Source: [naturalantibody.com/ab-design/](https://naturalantibody.com/ab-design/),
Google Drive folder, fetched via `gdown` (see `databases/src/_common.py`'s
`ensure_gdown()`/`gdown_folder()`). **License: CC BY-NC 4.0 (non-commercial use only)**
— see `LICENSE.txt`, fetched alongside the data.

**Important scope note, verified 2026-08-15**: this bundle is not just AbDesign's own
data — it reprocesses **three** source datasets into one IMGT-numbered, structurally
consistent format:

| `dataset` value | mutant rows |
|---|---|
| `AbDesign` | 658 |
| `SKEMPIv2` | 377 |
| `AB-Bind` | 268 |
| **total** | **1,303** (+ 57 wildtype complexes) |

The `AB-Bind` rows here are a *reprocessed* version of `databases/ab_bind/`'s data
(IMGT-renumbered, modeled mutant structures via ABodyBuilder2) — keep both: use
`databases/ab_bind/` for the original citable source, this folder's `dataset ==
'AB-Bind'` subset when you specifically want the reprocessed/modeled version.

## Why this matters for affinity prediction specifically

AbDesign's own 658 mutants (14 antibodies × 7 antigens, all measured under one
consistent ELISA protocol) are a deliberately hard out-of-distribution test: the
companion paper found affinity predictors trained/tuned on SKEMPI/AB-Bind achieve
Spearman ρ≈0.4–0.7 on their native data but collapse to ρ≈0.0–0.1 on this
non-overlapping set. **Use `dataset == 'AbDesign'` as a held-out generalization check
for any affinity model built from this project's other data** (ANDD, AB-Bind, SAbDab) —
not as additional training volume, since mixing it into training defeats the point of
having an independent OOD eval set.

## What's here

- `datasets_mut.csv` — 1,303 mutant rows: `dataset`, `pdb_name`, `method` (ELISA/SPR/
  ITC/etc.), `affinity`, `affinity_type` (`elisa_mut_to_wt_ratio` or `ddg` depending on
  source), mutation position (raw + IMGT), full heavy/light/antigen sequences, CDR
  sequences, IMGT position mappings.
- `datasets_wt.csv` — 57 wildtype complex rows, same schema minus mutation fields.
- `AbDesign_structures/abdesign/{AbDesign,AB-Bind,SKEMPIv2}/` — extracted from the
  bundle's `abdesign.tar.gz`: crystal structures, ABodyBuilder2-modeled mutant/WT
  structures (raw, refined, variable-region-trimmed variants), per source.
- `UPSTREAM_README.md` — the dataset authors' own README (renamed from `README.md` to
  avoid colliding with this file), full directory-structure documentation.
- `LICENSE.txt` — CC BY-NC 4.0 full text.

## Reproducing

```bash
python3 databases/src/abdesign_db.py
```
