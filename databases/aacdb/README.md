# aacdb — curated paratope/epitope annotations (AACDB)

Source: [AACDB](https://i.uestc.edu.cn/AACDB/) (Antigen-Antibody Complex Database),
University of Electronic Science and Technology of China, v1.0, 7,498 manually-processed
antibody-antigen complex rows / 3,674 unique PDB ids. Free, direct HTTP download, no
license gate. Contact for questions: hj@uestc.edu.cn (listed on their site).

**This folder deliberately does NOT include AACDB's own structure/fasta files.**
Verified: **3,651 of AACDB's 3,674 unique PDB ids (99.4%) already exist in
`databases/sabdab/structures/`** (reproducible — see the check at the bottom of this
file). AACDB's real value for this project isn't structures, it's the annotation layer:

- `protein_table.txt` — 7,498 rows, curated/corrected metadata per complex: `pdb`,
  `chains`, `antibody` name, `INN(clinical_trial)` (therapeutic cross-reference),
  `ab_mutation`/`mutation`, `protein` (antigen name), `targets` (antigen UniProt id),
  `ag_mutation`, `organism`, `method`, `resolution`, literature `reference` DOI.
- `revised_entries.txt` — AACDB's own notes on PDB annotation errors they corrected.
- `interacting_res_distance/` and `interacting_res_SASA/` — one file per complex
  (`<pdb>_<chains>_interacting_residues_{distance,SASA}.txt`), paratope/epitope residue
  assignments by two independent methods (atom-distance cutoff, and SASA-burial upon
  complex formation). Not something SAbDab's own summary provides.

## Join key

`protein_table.txt`'s `pdb` column is a plain 4-character PDB id (e.g. `1A14`) —
lowercase it and prefix `pdb_0000` to match `databases/sabdab/summary.csv`'s `PDB`
column, same pattern as `databases/sabdab/affinity/`.

## Reproducing

```bash
python3 databases/src/aacdb.py
```

To reproduce the overlap check:
```python
import csv
aacdb_pdbs = {r['pdb'].lower() for r in csv.DictReader(open('databases/aacdb/protein_table.txt'), delimiter='\t')}
sabdab_pdbs = {r['PDB'].replace('pdb_0000', '').lower() for r in csv.DictReader(open('databases/sabdab/summary.csv'))}
print(len(aacdb_pdbs & sabdab_pdbs), '/', len(aacdb_pdbs))   # -> 3651 / 3674
```
