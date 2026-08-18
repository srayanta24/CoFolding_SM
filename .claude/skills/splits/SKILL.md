---
name: splits
description: Build or regenerate the leak-free train/dev/test splits at ~/co_folding/databases/splits/, used to evaluate anything trained or fine-tuned as part of the design-pipeline improvement work (epitope prediction, Germinal comparison, BoltzGen fine-tuning). Use for "rebuild the splits", "regenerate the train/test split", "update the splits after re-downloading sabdab", or when a new databases/ source needs to be cross-referenced into the split. Not for running a design or cofolding job — see the "design" skill for that; not for fetching database sources — see the "database" skill for that.
---

## What this is

`databases/splits/` partitions every antigen-bound structure in `databases/sabdab/`
into buckets that are safe to train on vs. safe to evaluate on, so that any
"improvement" measured later (an epitope predictor, a fine-tuned checkpoint, a Germinal
comparison) is real — not an artifact of BoltzGen's base model having already
memorized the test structure. Full methodology and verified counts:
`databases/splits/README.md`. Built by `databases/src/build_splits.py`.

**Unlike every other `databases/<name>/` folder, everything in `databases/splits/` is
tracked in git** (small, decision-bearing files — the split assignment is a specific
analytical decision worth version-controlling, not bulk re-fetchable data). Don't
gitignore anything added here without a specific reason.

## The two-part methodology (don't simplify to just one)

1. **Temporal cutoff**, mirrors BoltzGen's own training filter exactly — verified by
   grep across all three training configs in `src/boltzgen/resources/config/train/`:
   `DateFilter(date="2023-06-01", ref="released")`. Structures released on or before
   that date go in `train_era`; the rest are *candidates* for `test`/`dev`.
2. **MMseqs2 sequence-identity clustering** (antigen sequences, 40% identity) on top of
   the date check. A post-cutoff structure whose antigen sequence clusters with a
   train-era structure's antigen is excluded (`excluded_ambiguous`) — otherwise a later
   crystal form of an already-known antigen would leak into "clean" test data even
   though its own PDB entry postdates the cutoff. **Do not ship a temporal-only split** —
   this second check is why the split is trustworthy, not just convenient.

A post-cutoff structure whose antigen sequence couldn't be extracted is **never**
defaulted into the clean pool (`excluded_no_sequence` instead) — its redundancy against
`train_era` can't be verified, so it isn't assumed safe. This was a real bug caught
during verification (an earlier version silently trusted 291 such structures as clean)
— if you touch `build_split()`, keep this invariant.

## Running

```bash
python3 databases/src/build_splits.py
```

Fully reproducible: hashing is SHA-256-of-PDB-id for the deterministic `dev`/`test`
slice (not random-seeded), so re-running from a clean `databases/splits/` reproduces
identical bucket assignments — verified during development. Takes a few minutes;
MMseqs2 clustering of ~9,200 sequences is the slow part but still fast in absolute
terms (well under a minute) — most of the wall-clock is the local mmCIF parsing pass.

Re-run this whenever `databases/sabdab/` is re-fetched/updated (new structures released
since the last run shift the `train_era`/post-cutoff boundary) or when a new source
needs cross-referencing (see below).

## Gotchas (hit while building this — don't reintroduce them)

- **`antigen_chain` in `databases/sabdab/summary.csv` is `|`-delimited for multiple
  chains** (e.g. `"I|J"`), never `+`-delimited — an early version of the parser assumed
  `+` and silently matched nothing.
- **The `_entity_poly` mmCIF parser must skip past the rest of the marker's own line**
  before reading data rows — `text.find(marker)` lands mid-line (right after the field
  name), so the first "row" naively read is the empty remainder of that header line,
  which looks like a blank-line terminator and stops parsing before it starts. Skip to
  the next `\n` first (see `parse_entity_poly()` in `build_splits.py` for the fix,
  including the comment explaining why).
- **~19% of antigen sequence extractions fail, and that's expected, not a bug**:
  ~1,257 are genuinely non-protein antigens (`antigen_chain == "NA"` — haptens, ions,
  sugars, which have no polymer sequence by definition), ~960 are chain-lettering
  mismatches between `summary.csv` and the specific mmCIF (same class of label-vs-auth
  chain-id gotcha already documented in `databases/sabdab/README.md`). Every failure is
  logged in `splits_summary.json`'s `extraction_failures` list, not silently dropped —
  if this number changes a lot on a re-run, that's worth investigating, but a
  ~20% failure rate on its own is not a regression.
- **MMseqs2 needs `databases/src/_common.py`'s `ensure_mmseqs2()`** (downloads the
  static aarch64 binary into `.venvs/mmseqs2/`, no sudo/apt needed — verified working on
  this machine). Don't add a system-package (`apt install mmseqs2`) dependency instead;
  that was explicitly avoided in favor of the no-sudo static binary.
- **Only `aacdb` and `ab_bind` are cross-referenced today** (`cross_reference_sources()`
  in `build_splits.py`) — `andd` and `abdesign_db` aren't wired in yet. Same pattern to
  extend: look up the source's own PDB-id column (documented per-source in each
  `databases/<name>/README.md`), tally against the bucket assignment, add a row to the
  table in `databases/splits/README.md`.
- **Intermediate MMseqs2 working files (`~/.mmseqs_work/`) are deliberately deleted**
  after each run — only the small `antigen_clusters.tsv` (240KB, human-inspectable,
  supports auditing `excluded_ambiguous` entries) is kept. Don't accidentally commit the
  full working directory if you modify the cleanup step.

## How the buckets get used downstream

Per `IMPROVE_DESIGN.md`: `train_era` for anything we train (epitope predictor now,
fine-tuning later); `dev` for iterating during development; `test` touched only for
final reported comparisons against baseline BoltzGen — don't use `test` for
hyperparameter tuning or repeated checks, that defeats the point of holding it out.
