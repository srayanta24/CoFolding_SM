#!/usr/bin/env python3
"""Fetch ANDD (Antibody and Nanobody Design Dataset): sequences, structure metadata,
and binding-affinity measurements unifying 15 source databases.

Metadata (xlsx + data dictionary + QC report) is fetched by default. The structures
zip (ANDD_pdb.zip, 2.2GB) is gated behind --structures (default off) — see README.md
for the measured overlap against databases/sabdab/structures/ before deciding whether
you need it.

Source: Zenodo, DOI 10.5281/zenodo.18151718, v2, CC BY 4.0.

Usage:
    python3 databases/src/andd.py                # metadata only
    python3 databases/src/andd.py --structures    # also fetch+extract the 2.2GB structures zip
"""

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.parse import quote

from _common import DATABASES_DIR, download

ANDD_DIR = DATABASES_DIR / "andd"
BASE_URL = "https://zenodo.org/records/18151718/files"

METADATA_FILES = {
    "ANDD_v2.xlsx": f"{BASE_URL}/{quote('Antibody and Nanobody Design Dataset (ANDD)_v2.xlsx')}?download=1",
    "Data_dictionary.csv": f"{BASE_URL}/Data_dictionary.csv?download=1",
    "Data_quality_control_report.pdf": f"{BASE_URL}/Data_quality_control_report.pdf?download=1",
}
STRUCTURES_URL = f"{BASE_URL}/ANDD_pdb.zip?download=1"


def fetch_metadata(out_dir: Path = ANDD_DIR) -> Path:
    for name, url in METADATA_FILES.items():
        out_path = out_dir / name
        print(f"[andd] fetching {name} -> {out_path}", file=sys.stderr)
        download(url, out_path)
    return out_dir


def fetch_structures(out_dir: Path = ANDD_DIR / "structures") -> Path:
    tmp_zip = ANDD_DIR / ".ANDD_pdb.zip.partial"
    print(f"[andd] fetching structures archive (2.2GB, resumable) -> {tmp_zip}", file=sys.stderr)
    download(STRUCTURES_URL, tmp_zip, resume=True)

    print(f"[andd] extracting -> {out_dir}", file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp_zip) as zf:
        zf.extractall(out_dir)
    tmp_zip.unlink()
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--structures", action="store_true", help="Also fetch the 2.2GB structures zip")
    args = parser.parse_args()

    fetch_metadata()
    if args.structures:
        fetch_structures()


if __name__ == "__main__":
    main()
