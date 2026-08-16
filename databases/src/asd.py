#!/usr/bin/env python3
"""Fetch ASD (Antigen-Specific Antibody Database): ~1.1M antibody-antigen sequence +
affinity interactions aggregated from 15 sources.

**Non-commercial research use terms** — see naturalantibody.com/asd/. Delta Lake /
Parquet format (not CSV): a `_delta_log/` transaction log plus 20 `part-*.snappy.parquet`
files. Reading it later needs pyarrow/pandas/deltalake (none installed in this project
yet — not installed here either, since this script only downloads bytes, it doesn't
query them). Requires gdown (see databases/src/_common.py's ensure_gdown()).

**Does NOT use gdown's `--folder` mode** — verified it gets stuck in a pathological
infinite loop on this specific folder (31,483 repeated "Processing file" log lines for
only 42 real+.crc files, never downloading a single byte across several hours, killed
by hand). The folder's own name on Drive, "Copy of ASD: ...", suggests a duplicated
folder structure is the likely trigger. Fetches each of the 20 known part files
individually by Drive file id instead (single-file gdown, verified reliable: ~19MB in
~1.5s per part) — the file id -> name map below was captured from the one folder
listing that did complete far enough to enumerate them, before the loop was noticed.

Usage:
    python3 databases/src/asd.py
"""

import sys
from pathlib import Path

from _common import DATABASES_DIR, ensure_gdown

ASD_DIR = DATABASES_DIR / "asd"

# id -> filename, captured 2026-08-15 from a (non-terminating) `gdown --folder` listing
# of https://drive.google.com/drive/u/1/folders/1HY2GOVj-HR8t6Jmhe5MIJ6xmBCOxC45q
# (folder path inside: "Copy of ASD.../asd/_delta_log/" for the json, "Copy of ASD.../asd/"
# for the parquet parts).
FILES = {
    "1gyItHBY_jbJ-c_RRrnvDL7lhwkiViqB-": "_delta_log/00000000000000000000.json",
    "1f45RupxXYKiMjdelCPDh_Wx0OwPbnpLj": "part-00000-3a065afd-b2fa-4875-a6e5-911e95e3f86c-c000.snappy.parquet",
    "1MBMIQN2f43FNXqadF2EDBPKN--gRlSHT": "part-00001-74fdee0d-8448-4b0a-921c-f1f0ef356cdf-c000.snappy.parquet",
    "1P2GtlwBxM92HPoKYD8O0_l3KFCKvc4ki": "part-00002-634ededa-5adf-4170-93ba-2dac2bd74705-c000.snappy.parquet",
    "1sFle1IXpeKWFkae2spohCgL6IaQqsfJ0": "part-00003-427bc79e-0e40-4a8c-a2be-d0fbe09f03c0-c000.snappy.parquet",
    "1V6y7XXLfqwIR7fChogQ2gBq3XUUx1P8v": "part-00004-eb3dc336-995c-48bd-840f-49d411a89b8e-c000.snappy.parquet",
    "1859NLPreFjCKqOT6DT8_up5hxqV2XLDi": "part-00005-846a5164-9ca8-4438-af5f-07ad5348f327-c000.snappy.parquet",
    "1rryg2DeHd2QVj7qnBWuGaZLi0O8cEKsL": "part-00006-b818506c-2926-406c-936b-66da5c9acdbc-c000.snappy.parquet",
    "1UoJg_5m4-ipE501P7Mz1wS2fC4ckqYYj": "part-00007-7ab8e466-84f8-43e0-8874-9c1bbf210d4e-c000.snappy.parquet",
    "1OeJVehZ6rmnq-9Rdt2IYup1U1KoD_1Oq": "part-00008-397aa529-5cb9-4e24-8898-c9940200ae64-c000.snappy.parquet",
    "11ZpU6V0Ivi5_gT4rojd638QHfBw279X1": "part-00009-6b319ece-d8eb-4e15-b579-4e98d3a456a1-c000.snappy.parquet",
    "15SqbOfoEO89Nc87wdD2udKdG7-Mz_9Ik": "part-00010-7edd7ab3-f323-4718-a2ac-138fb65b3f42-c000.snappy.parquet",
    "1s985K-Ne2vJHUWDHgRZORQnU18zkmvba": "part-00011-86ada209-259a-4a93-8a61-ee6555e0f25d-c000.snappy.parquet",
    "17zDHS7fseSUE0dA1YvVFybRNdxKr3rLb": "part-00012-fe431735-b7e1-4367-b665-15a59e7bd12c-c000.snappy.parquet",
    "1gYCMqI8K6Tz52C-FJrq0ktzDqhIqPnHA": "part-00013-26624da0-a286-49c6-98c5-c019816424b8-c000.snappy.parquet",
    "1K9eG_Vp_4-KUhHGBfwHlutnT0HZBS04V": "part-00014-0bbd5d34-4a5c-4c7f-9c69-ba69e33861b1-c000.snappy.parquet",
    "1yA7BSOyMebn4goJAL8vlNmINCAwPNHGR": "part-00015-af04209c-672a-4fd7-9cc5-1fdaeaf06aa0-c000.snappy.parquet",
    "1Tq4mp91zKfdY5SNA6rbNu7UT80tGvQG4": "part-00016-883dd12e-3f06-4505-b326-04b6c16a7852-c000.snappy.parquet",
    "1HlWr_VTitVnH52bdiXq3GYayYzdyNKtm": "part-00017-b88913fa-e655-4662-8208-45e9f4d38488-c000.snappy.parquet",
    "1nMCPJToAsatWrShqE7r5sbBqUrBqOXpN": "part-00018-630653e8-04e5-4d69-bd4f-96225f04fb82-c000.snappy.parquet",
    "1YPqg8fEbeO7R3ahvsS_LYe-woMocaMl6": "part-00019-2a17653c-3a60-4f9a-b840-e7b168d3d6f9-c000.snappy.parquet",
}


def fetch(out_dir: Path = ASD_DIR) -> Path:
    import subprocess

    gdown = ensure_gdown()
    for file_id, rel_name in FILES.items():
        out_path = out_dir / rel_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            print(f"[asd] already have {rel_name}, skipping", file=sys.stderr)
            continue
        print(f"[asd] fetching {rel_name} -> {out_path}", file=sys.stderr)
        subprocess.run([gdown, file_id, "-O", str(out_path)], check=True)
    return out_dir


if __name__ == "__main__":
    fetch()
