# sabdab — local antibody-antigen structure and affinity database

Local mirror of two real, known benchmark datasets, fetched by
`databases/src/sabdab.py`. Everything in this folder except this file is gitignored
(large, and 100% reproducible by re-running the script) — see `.gitignore`. The fetch
code itself lives in `databases/src/` (not here) alongside every other source's, so the
whole database can be recreated on another machine from a clone — see `databases/README.md`.

```
databases/
  src/sabdab.py                                    # reproduces everything below (tracked)
  sabdab/
    README.md                                       # this file (tracked)
    summary.csv                                     # bulk structure metadata (gitignored)
    structures/pdb_0000<id>/pdb_0000<id>_sabdab.cif  # one mmCIF per entry (gitignored)
    affinity/antibody_affinity_protein_sabdab.csv     # Kd benchmark (gitignored)
```

## Structures + summary — SAbDab2

Source: SAbDab2 (Oxford Protein Informatics Group's structural antibody database, also
the source of the one hand-curated reference complex in `experiments/reference_targets.py`).
Its own Zenodo mirror (`zenodo.org/records/20083995`) only publishes train/test split
CSVs, not the structures themselves — the actual bulk data comes from SAbDab2's backend
API, found by grepping its JS bundle for `fetch()` call sites:

- `https://sabdab.opig.stats.ox.ac.uk/api/download/all-summary` → `summary.csv`
- `https://sabdab.opig.stats.ox.ac.uk/api/download/all-structures` → one mmCIF per entry

Fetched 2026-08-15. Verified real numbers from that fetch:
- `summary.csv`: **21,914 rows × 45 columns** — one row per antibody chain-pair instance
  (`Hchain`/`Lchain`/`antigen_chain`/`antigen_type`/`antigen_name`/`resolution`/`method`/
  species/expression-system columns, etc.). **19,560 rows have `antigen_chain` set**
  (antigen-bound); the remaining ~2,354 are apo/unbound antibody structures.
- `structures/`: one directory per unique PDB entry,
  `pdb_0000<id>/pdb_0000<id>_sabdab.cif` (SAbDab's own renumbered/standardized mmCIF,
  not the raw RCSB file) — extracted **every** entry, not just antigen-bound ones, so
  apo-vs-holo comparisons are possible later without a second download.
- Source tarball was 2,706,362,644 bytes (2.7GB) compressed.

To find antigen-bound, protein-antigen entries (the subset most relevant to this
project's design work): filter `summary.csv` for `antigen_chain` non-empty and
`antigen_type` containing `PROTEIN` (it's a `|`-delimited multi-value field, e.g.
`PROTEIN|SUGAR` for a glycosylated antigen).

## Affinity — TDC AntibodyAff / Protein_SAbDab

Source: Therapeutics Data Commons' `AntibodyAff` task, `Protein_SAbDab` dataset — 493
antibody-antigen pairs with **experimental Kd** (heavy+light antibody sequence, antigen
sequence, Kd in Molar). This is the standard published benchmark for this task, not
something assembled ad hoc.

Its canonical host is Harvard Dataverse (`doi:10.7910/DVN/21LKWG`, file id `4167357`),
but that endpoint sits behind an **AWS WAF JS challenge that blocks plain HTTP
clients** — verified directly: both a bare `curl` and one with a browser `User-Agent`
got `x-amzn-waf-action: challenge` instead of the file. The `PyTDC` pip package hits this
same endpoint, so it would likely hit the same block (or at best succeed only with a
JS-capable client), on top of pulling in a heavy dependency chain (pandas/scikit-learn/
fuzzywuzzy) just to fetch one CSV. Used a direct, unblocked mirror of the *same* file on
Zenodo instead: `zenodo.org/records/13120765` (Apache 2.0), file
`antibody_affinity_protein_sabdab.csv`.

Fetched 2026-08-15. Verified: 493 rows, columns `Antibody_ID, Antibody, Antigen_ID,
Antigen, Y` (`Antibody` is `"[heavy_seq, light_seq]"` as a stringified Python list;
`Y` is Kd in Molar, e.g. `4e-13` to `2e-4`, per TDC's own task documentation).

**Join key, verified**: `Antibody_ID` (e.g. `1hh6`) is a real PDB id — it joins into
`summary.csv`'s `PDB` column via the `pdb_0000<id>` prefix
(`pdb_0000` + `Antibody_ID` == `PDB`). **491 of the 493 affinity rows join successfully**
against the summary fetched the same day; the other 2 likely reference PDB entries that
have since been obsoleted/superseded in SAbDab2 — not investigated further.

**Scope caveat, worth repeating**: 493 pairs is a small, specific benchmark subset — most
of the ~10,308 unique antigen-bound structures in `structures/` have no published
experimental Kd anywhere. This is the best available *known* benchmark for the entries
that do, not affinity data for the full structural corpus.

## Reproducing

```bash
python3 databases/src/sabdab.py                          # all three, cheap steps first
python3 databases/src/sabdab.py --only summary --only affinity   # skip the slow ~2.7GB step
python3 databases/src/sabdab.py --only structures         # just the slow step (resumable)
```
