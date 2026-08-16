# ab_bind — antibody binding mutational database (AB-Bind)

Source: [AB-Bind](https://github.com/sarahsirin/AB-Bind-Database) (Sirin et al. 2016,
*Protein Science*, "AB-Bind: Antibody binding mutational database for computational
affinity predictions"). Free, public GitHub repo, no license gate. Fetched as a full
repo archive and flattened (the zip extracts into a single `AB-Bind-Database-master/`
subdirectory, moved up into this folder).

Verified 2026-08-15: **1,101 point-mutation rows across 32 unique complexes**
(`AB-Bind_experimental_data.csv` — note it needs `encoding="latin-1"` to read cleanly,
not UTF-8, due to a non-ASCII byte in the source data).

## What's here

- `AB-Bind_experimental_data.csv` — the main dataset: `#PDB`, `Partners(A_B)`,
  `Protein-1`, `Protein-2`, `Mutation`, **`ddG(kcal/mol)`** (binding free energy change
  upon mutation — the actual training/eval target), plus experimental provenance
  (resolution, R-value, pH, temperature, assay method, literature DOI).
- `<PDB>.pdb` / `HM_<PDB>.pdb` — wildtype structures for the 32 complexes (`HM_*`
  prefix = homology model, used where no wildtype crystal structure exists).
- `ABbind_compt_data.zip` — precomputed structural features from the original paper.
- `UPSTREAM_README.md` — the repo's own (minimal) README, renamed from `README.md` to
  avoid colliding with this file.

This is the *original, citable* source. `databases/abdesign_db/` separately bundles a
reprocessed copy of this same data (IMGT-renumbered, with modeled mutant structures) —
keep both: this folder for citation/provenance, that one for the ML-ready version.

## Reproducing

```bash
python3 databases/src/ab_bind.py
```
