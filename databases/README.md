# databases — local antibody design data

Local mirrors of known, citable antibody-antigen databases, assembled to support three
sub-goals of improving antibody design in this project: **generative design** (better
de novo binder generation, currently BoltzGen via `scripts/design_binder.py`),
**structure prediction** (cofolding, currently Boltz-2/OpenFold3 via
`scripts/run_design.py`), and **binding affinity prediction** (not yet built anywhere
in this project — these sources are the starting material for it).

Every source below follows the same pattern: `databases/<name>/README.md` (tracked)
documents exact provenance and real verified numbers; everything else in that folder is
gitignored (large, and 100% reproducible by re-running the fetch code). **All fetch code
lives in one place, `databases/src/`** (tracked in full — `databases/src/<name>.py` per
source, plus shared helpers in `databases/src/_common.py`), specifically so the whole
database can be recreated on a different computer from nothing but a clone of this repo:
see "Reproducing everything" below.

## Index

| Source | Primary subtask fit | Access | Real scale (verified) |
|---|---|---|---|
| [`sabdab/`](sabdab/README.md) | Structure prediction (primary corpus); affinity prediction (small labeled set) | Free, no gate | 11,458 structures; 493 affinity-labeled pairs |
| [`aacdb/`](aacdb/README.md) | Structure prediction (paratope/epitope interface annotation) | Free, no gate | 7,498 complex rows / 3,674 unique PDB (99.4% overlap with `sabdab/`, so annotations only, no structures fetched) |
| [`ab_bind/`](ab_bind/README.md) | Affinity prediction (standard ΔΔG baseline) | Free, no gate | 1,101 mutation rows / 32 complexes |
| [`andd/`](andd/README.md) | Affinity prediction (primary — largest labeled set); generative design (sequence+structure+affinity triples) | Free, CC BY 4.0 | 48,800 sequences; 9,565 affinity values (7,294 real + 2,271 model-predicted, kept tagged); structures default-skipped (98.0% overlap with `sabdab/`, verified) |
| [`abdesign_db/`](abdesign_db/README.md) | Affinity prediction (held-out generalization eval, not training volume) | **CC BY-NC 4.0**, `gdown` | 1,303 mutant rows (658 AbDesign + 377 SKEMPIv2 + 268 AB-Bind, all reprocessed/IMGT-numbered) |
| [`asd/`](asd/README.md) | Generative design / large-scale pretraining (primary); affinity prediction (secondary, heterogeneous quality) | **Non-commercial research use**, `gdown` | ~1.1M sequence+affinity interactions, 9,575 unique antigens; only ~3,969 have structures |

## Why these six, not others

Chosen because each is a real, published/citable benchmark rather than something
assembled ad hoc, and because each source was checked for genuine complementary value
before being added — most importantly, **AACDB and ANDD's own structures turned out to
be ~98-99% redundant with `sabdab/`'s corpus** (verified directly, not assumed), so
those two contribute their annotation/affinity layers, not duplicate structure
downloads. See each source's own README for the exact verification.

## New dependency: `gdown`

`abdesign_db/` and `asd/` are both gated behind Google Drive folder shares with no
direct-URL alternative (unlike Zenodo/GitHub, which are plain HTTP). `gdown` lives in
its own `.venvs/data-fetch/` venv, bootstrapped automatically the first time either
source's fetch script runs (`databases/src/_common.py`'s `ensure_gdown()`) — no separate
manual setup step. Every other source here is stdlib-only, matching the rest of this
project's `scripts/` layer.

## License summary

Four of six sources are free with no usage restriction beyond attribution (`sabdab`,
`aacdb`, `ab_bind`, `andd` — the last is CC BY 4.0, the rest have no stated license
gate at all). **`abdesign_db` is CC BY-NC 4.0 and `asd` is under non-commercial
research-use terms** — both fine for this project's research use, but flag this before
any commercial use of derived models/results.

## Reproducing everything

The single entry point, on any machine (clone the repo, then):

```bash
python3 databases/src/download_all.py
```

Or run a single source directly (also useful for re-running just one after a partial
failure):

```bash
python3 databases/src/sabdab.py
python3 databases/src/aacdb.py
python3 databases/src/ab_bind.py
python3 databases/src/andd.py          # metadata only by default
python3 databases/src/andd.py --structures    # + the 2.2GB structures zip (98% overlaps sabdab/)
python3 databases/src/abdesign_db.py   # CC BY-NC 4.0
python3 databases/src/asd.py           # non-commercial research use
```
