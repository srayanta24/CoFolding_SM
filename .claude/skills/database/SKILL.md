---
name: database
description: Create, recreate, or extend the local antibody-design database at ~/co_folding/databases/ (SAbDab, AACDB, AB-Bind, ANDD, AbDesign DB, ASD — real structures, sequences, and binding-affinity data). Use for "set up the database(s)", "recreate the database on a new machine", "download the antibody databases", "rebuild databases/", or when adding a new source database to this project. Not for running a design or cofolding job — see the "design" skill for that.
---

## What this is

`databases/` holds local mirrors of six real, citable antibody-antigen databases,
assembled to support three sub-goals of improving antibody design in this project:
**generative design**, **structure prediction**, and **binding affinity prediction**.
Full index, license terms, and real verified counts per source: `databases/README.md`.
Per-source detail (exact provenance, join keys, known redundancy with other sources):
`databases/<name>/README.md`.

**All fetch code lives in `databases/src/`** — one `.py` file per source plus shared
helpers in `databases/src/_common.py` — specifically so the whole database can be
recreated on a different machine from nothing but a clone of this repo. Every
per-source data folder (`databases/<name>/`) is gitignored except its `README.md`; only
`databases/src/*.py` and the six `README.md` files are tracked.

## Recreating the database (new machine, or after a clean clone)

```bash
python3 databases/src/download_all.py
```

This runs every source in order, cheap/free ones first, and skips ANDD's 2.2GB
structures zip by default (see "Known redundancy" below for why). To skip a source
entirely (e.g. the two non-commercial-license ones):

```bash
python3 databases/src/download_all.py --skip abdesign_db --skip asd
```

Or run one source directly — also the right move for re-running just one after a
partial failure, rather than re-running everything:

```bash
python3 databases/src/sabdab.py       # structures (~9.5GB extracted) + summary + small affinity set
python3 databases/src/aacdb.py        # annotations only, not structures (see below)
python3 databases/src/ab_bind.py      # small, fast
python3 databases/src/andd.py         # metadata only by default; --structures for the 2.2GB zip
python3 databases/src/abdesign_db.py  # ~3.5GB extracted, CC BY-NC 4.0
python3 databases/src/asd.py          # ~380MB, non-commercial research use
```

Expect the full run to take a while and use real disk (sabdab alone is ~9.5GB
extracted) — run in the background, don't set a short timeout, same convention as
every other long-running job in this project.

## Known gotchas (hit and fixed while building this — don't reintroduce them)

- **`gdown` bootstrap is automatic.** `abdesign_db.py` and `asd.py` are gated behind
  Google Drive folder shares with no direct-URL alternative. The first time either
  runs, `_common.py`'s `ensure_gdown()` creates `.venvs/data-fetch/` and installs
  `gdown` into it — no manual setup step needed. Every other source is stdlib-only.
- **Never use `gdown --folder` on the ASD folder.** Verified it gets stuck in a
  pathological infinite loop on that specific Drive folder (31,483 repeated log lines
  for 42 real files, never downloading a byte across several hours — likely triggered
  by the folder's own "Copy of ASD..." duplicated-name structure on Drive).
  `databases/src/asd.py` instead hardcodes the 20 part files' individual Drive file
  ids and fetches each one with single-file `gdown` (verified reliable, ~19MB/file in
  ~1.5s). If ASD's data ever needs to be re-derived (e.g. the source folder changes),
  get fresh file ids from the Drive folder UI directly rather than trying
  `--folder` again.
- **Archive extraction can silently overwrite this project's own README.md.** Both
  `ab_bind.py` (GitHub repo zip) and `abdesign_db.py` (Drive bundle) contain their
  *own* `README.md` at the path this project also wants to put its own tracked
  provenance doc. Both scripts rename the upstream one to `UPSTREAM_README.md` before
  it can collide — if you add a new source that extracts an archive, check whether it
  contains a top-level `README.md` and apply the same rename, or a `download_all.py`
  run will quietly destroy the tracked doc (this happened once while building this,
  caught by noticing `git status` show an unexpected modification).
- **AACDB and ANDD's own structure files are ~98-99% redundant** with
  `databases/sabdab/structures/` (verified directly, not assumed — see each source's
  README for the exact overlap check, reproducible with stdlib `csv`/`zipfile`/
  `xml.etree`, no `pandas`/`openpyxl` needed). `aacdb.py` fetches only its annotation
  files (paratope/epitope interface residues, corrected metadata), never structures.
  `andd.py` fetches structures only if explicitly asked via `--structures`.
- **`databases/andd/ANDD_v2.xlsx`'s `Predicted_or_Not` column matters.** ~2,271 of its
  affinity rows are model-*predicted* (ANTIPASTI), not experimental — filter on
  `Predicted_or_Not == 'real'` for anything training/evaluating an affinity model
  unless the predicted rows are deliberately wanted too.

## Adding a new source database

Follow the existing six as the template:
1. Verify the source live (real URLs, real file sizes/counts) before writing any code
   — don't trust a paper's abstract or a database homepage's claimed numbers blindly;
   every number in every `databases/<name>/README.md` was checked against an actual
   fetch, and more than one turned out to need correction (a plddt-scale mismatch, a
   miscounted overlap, an affinity column that looked populated but was 80% placeholder
   values — see `databases/andd/README.md` for that last one).
2. Check for redundancy against what's already here (structures against
   `databases/sabdab/summary.csv`'s `PDB` column via the `pdb_0000<id>` join pattern,
   same check done for AACDB/ANDD) before committing to downloading everything a new
   source offers.
3. Add `databases/src/<name>.py` using `databases/src/_common.py`'s `download()` (plain
   HTTP, resumable) or `gdown_folder()`/`ensure_gdown()` (Google-Drive-gated) helpers —
   don't reimplement streaming/resume logic per source.
4. Add `databases/<name>/README.md` documenting exact provenance, real verified counts,
   license/access terms, and join keys to other sources.
5. Wire it into `databases/src/download_all.py`'s `SOURCES` list and fetch sequence.
6. Update the index table in `databases/README.md`.
