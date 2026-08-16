#!/usr/bin/env python3
"""Fetch AB-Bind: 1,101 point-mutation binding-affinity-change (ddG) measurements
across 32 antibody-antigen complexes, with wildtype/homology-model PDB structures.

Standard literature baseline for affinity-change prediction (Sirin et al. 2016).
Freely downloadable, public GitHub repo, no license gate. Fetched as a single repo zip
(https://github.com/sarahsirin/AB-Bind-Database/archive/refs/heads/master.zip) rather
than hitting the GitHub contents API per-file — one small repo, no reason to spend
unauthenticated API rate-limit budget on it.

Usage:
    python3 databases/src/ab_bind.py
"""

import shutil
import sys
import zipfile
from pathlib import Path

from _common import DATABASES_DIR, download

AB_BIND_DIR = DATABASES_DIR / "ab_bind"
ZIP_URL = "https://github.com/sarahsirin/AB-Bind-Database/archive/refs/heads/master.zip"


def fetch(out_dir: Path = AB_BIND_DIR) -> Path:
    tmp_zip = out_dir / ".ab-bind.zip.partial"
    print(f"[ab_bind] fetching repo archive -> {tmp_zip}", file=sys.stderr)
    download(ZIP_URL, tmp_zip)

    print(f"[ab_bind] extracting -> {out_dir}", file=sys.stderr)
    with zipfile.ZipFile(tmp_zip) as zf:
        zf.extractall(out_dir)
    tmp_zip.unlink()

    # zip extracts to a single "AB-Bind-Database-master/" subdirectory; flatten it.
    # The upstream README.md would otherwise collide with (and silently overwrite) this
    # source's own tracked README.md — rename it first, same fix as abdesign_db.py.
    extracted = out_dir / "AB-Bind-Database-master"
    upstream_readme = extracted / "README.md"
    if upstream_readme.exists():
        upstream_readme.rename(extracted / "UPSTREAM_README.md")
    for item in extracted.iterdir():
        shutil.move(str(item), str(out_dir / item.name))
    extracted.rmdir()

    return out_dir


if __name__ == "__main__":
    fetch()
