# asd — Antigen-Specific Antibody Database (ASD)

Source: [naturalantibody.com/asd/](https://www.naturalantibody.com/asd/), Google Drive
folder. **Non-commercial research use terms** (see the site). Companion paper: "ASD:
antigen-specific antibody database" (2026). Aggregates 15 source datasets:
**1,097,946 unique antibody-antigen interactions, 9,575 unique antigens** (per the
paper — this project did not re-derive that count, see "What wasn't verified" below).

## Format: Delta Lake / Parquet, not CSV

Verified 2026-08-15/16: the data is a Delta Lake table — a `_delta_log/` transaction
log (`00000000000000000000.json`) plus **20 `part-*.snappy.parquet` files**, 374.2MB
total (357MB on disk including the json). Querying this needs `pyarrow`, `pandas`, or
`deltalake` — **none of which are installed anywhere in this project** (the `scripts/`
and `databases/src/` layers are deliberately stdlib-only). Not installed here either,
since this script only downloads bytes and doesn't need to read them. Row counts
weren't independently re-verified from the parquet footers (would need a Thrift
compact-protocol parser or one of the above libraries) — the 1,097,946 figure is the
paper's own claim, not something this project confirmed by direct row count.

Per the site, the standardized schema includes: `dataset` (source abbreviation),
`heavy_sequence`, `light_sequence`, `antigen_sequence`, `affinity_type` (e.g. IC50),
`affinity` (numeric), `metadata.target_uniprot`, `metadata.target_pdb`,
`metadata.target_name`, `confidence` (high/medium/very_high), `scfv` (boolean). Only
~3,969 of the aggregate entries have associated structures (per the site) — this is
predominantly a sequence-level resource.

## Real gotcha: `gdown --folder` gets stuck in an infinite loop on this specific folder

**Do not use `gdown --folder <url>` against ASD's Drive folder.** Tried it first (the
same approach that works fine for `abdesign_db`); it got stuck re-listing the same 42
files (20 parquet + 20 `.crc` + the delta json + its `.crc`) over and over —
**31,483 repeated "Processing file" log lines across several hours, zero bytes ever
downloaded** — before being killed by hand. The folder's own name on Drive, "Copy of
ASD: Antigen Specific Antibody Database" (visible in `gdown`'s listing output), suggests
a duplicated folder structure is the likely trigger — `gdown`'s recursive folder walk
doesn't handle it gracefully.

**Fix used**: `databases/src/asd.py` fetches each of the 20 parquet files (+ the delta
log json) individually by Drive file id, via single-file `gdown <id> -O <path>` — no
folder recursion involved. Verified reliable: ~19MB per file in ~1.5s. The file
id → name mapping is hardcoded in the script (captured from the one folder listing that
enumerated far enough before the loop was noticed). If this ever needs regenerating
(e.g. ASD publishes an updated version), get fresh file ids from the Drive folder UI
directly rather than retrying `--folder` mode.

## What wasn't verified

- Exact row/interaction counts (see Format section above — not re-derived from parquet).
- Whether the 1,097,946-interaction and 9,575-antigen figures reflect this exact
  snapshot or an earlier version of the dataset.

## Reproducing

```bash
python3 databases/src/asd.py
```
