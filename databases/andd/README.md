# andd — Antibody and Nanobody Design Dataset (ANDD)

Source: [Zenodo record 18151718](https://zenodo.org/records/18151718), v2 (2026-01-05),
DOI `10.5281/zenodo.18151718`, CC BY 4.0. Companion paper: "A Unified Dataset for
Antibody and Nanobody Design Including Sequence, Structure, and Binding Affinity Data"
(*Scientific Data*). Unifies 15 source databases into one sheet. Free, no license gate.

## What's here

- `ANDD_v2.xlsx` (13.3MB, fetched by default) — the main data table, **48,800 rows**
  (verified: 48,801 rows incl. header). Key columns: `Source`, `PDB_ID`,
  `Experimental_Method`, `Ab_or_Nano` (**18,464 Antibody / 30,119 Nanobody-VHH / 121
  scFv / 96 other**), heavy/light/antigen chain ids + accessions, `Ag_Seq`, CDR sequences
  (H1-H3, L1-L3), `Affinity_Kd(M)`, `∆Gbinding(kJ/mol)`, `Affinity_Method`,
  **`Predicted_or_Not`**.
- `Data_dictionary.csv` — column definitions/controlled vocabulary.
- `Data_quality_control_report.pdf` — the authors' own QC documentation.
- `structures/` (only if fetched with `--structures`, see below) — one PDB file per
  structural entry, from `ANDD_pdb.zip`.

## Affinity coverage — verified counts (not just the paper's headline numbers)

Parsed `ANDD_v2.xlsx` directly (stdlib `zipfile` + `xml.etree`, no `openpyxl`/`pandas`
needed) rather than trusting the abstract's numbers blindly. The `Affinity_Kd` column is
non-empty for every row, but most of those are a `\` placeholder, not a real value.
Filtering by `Predicted_or_Not`:

| `Predicted_or_Not` | rows |
|---|---|
| `real` (experimental) | 7,294 |
| `predicted` (ANTIPASTI model) | 2,271 |
| no value / not applicable (`\`) | 39,235 |

**9,565 rows have an actual affinity value** (close to the paper's stated 9,557 — small
version delta between what's cited and this v2 download, not a data problem). **If you
use this for affinity-prediction training or eval, filter on `Predicted_or_Not == 'real'`
unless you deliberately want the model-predicted rows too** — blending them in as if
measured would let training data quietly imitate ANTIPASTI's own errors.

## Structures — verified redundancy, same pattern as AACDB

**7,630 of ANDD's 7,782 unique `PDB_ID` values (98.0%) already exist in
`databases/sabdab/structures/`** — verified the same way as the AACDB check (join by
lowercased PDB id). Only 152 PDB ids are ANDD-only. Given this, **`--structures` (the
2.2GB `ANDD_pdb.zip`) defaults to off** — fetch it only if you specifically need ANDD's
own processing of those 152 non-overlapping entries, or its particular file layout.

To reproduce the check:
```python
import zipfile, xml.etree.ElementTree as ET, csv
NS = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile('databases/andd/ANDD_v2.xlsx') as z:
    shared = ET.fromstring(z.read('xl/sharedStrings.xml'))
    strings = [(si.find('a:t', NS).text or '') if si.find('a:t', NS) is not None else
               ''.join((r.find('a:t', NS).text or '') for r in si.findall('a:r', NS))
               for si in shared.findall('a:si', NS)]
    sheet = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    rows = sheet.findall('.//a:sheetData/a:row', NS)
    def cell_val(c):
        v = c.find('a:v', NS)
        return (strings[int(v.text)] if c.get('t') == 's' else v.text) if v is not None else ''
    header = [cell_val(c) for c in rows[0].findall('a:c', NS)]
    pdb_col = header.index('PDB_ID')
    andd_pdbs = {cell_val(row.findall('a:c', NS)[pdb_col]).strip().lower()
                 for row in rows[1:] if pdb_col < len(row.findall('a:c', NS))}
sabdab_pdbs = {r['PDB'].replace('pdb_0000', '').lower()
               for r in csv.DictReader(open('databases/sabdab/summary.csv'))}
print(len(andd_pdbs & sabdab_pdbs), '/', len(andd_pdbs))   # -> 7630 / 7782
```

## Reproducing

```bash
python3 databases/src/andd.py                # metadata only (fast)
python3 databases/src/andd.py --structures    # also the 2.2GB structures zip
```
