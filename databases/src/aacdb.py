#!/usr/bin/env python3
"""Fetch AACDB's curated annotation layer (not structures — see README.md).

AACDB (i.uestc.edu.cn/AACDB) is a manually-curated antigen-antibody complex database,
7,498 complex rows / 3,674 unique PDB ids. Verified during planning: 3,651 of those
3,674 (99.4%) already exist in databases/sabdab/structures/, so this script does NOT
fetch AACDB's own structure/fasta zips (that would be ~99% redundant download). What it
adds instead is AACDB's real value: paratope/epitope interface annotations (two
independent methods — SASA-burial and atom-distance) and corrected PDB metadata that
raw SAbDab doesn't carry.

Usage:
    python3 databases/src/aacdb.py
"""

import sys
import zipfile
from pathlib import Path

from _common import DATABASES_DIR, download

AACDB_DIR = DATABASES_DIR / "aacdb"
BASE_URL = "https://i.uestc.edu.cn/AACDB/data_zip"

FILES = {
    "protein_table.txt": f"{BASE_URL}/protein_table.txt",
    "revised_entries.txt": f"{BASE_URL}/The%20detail%20information%20of%20revised%20entries.txt",
    "interacting_res_distance.zip": f"{BASE_URL}/interacting_res_distance.zip",
    "interacting_res_SASA.zip": f"{BASE_URL}/interacting_res_SASA.zip",
}


def fetch_all(out_dir: Path = AACDB_DIR) -> Path:
    for name, url in FILES.items():
        out_path = out_dir / name
        print(f"[aacdb] fetching {name} -> {out_path}", file=sys.stderr)
        download(url, out_path)
        if out_path.suffix == ".zip":
            print(f"[aacdb] extracting {name}", file=sys.stderr)
            with zipfile.ZipFile(out_path) as zf:
                zf.extractall(out_dir / out_path.stem)
    return out_dir


if __name__ == "__main__":
    fetch_all()
