#!/usr/bin/env python3
"""Fetch AbDesign DB: a bundle of antibody point-mutant data + structures that
actually reprocesses three source datasets (AbDesign itself, AB-Bind, and SKEMPIv2) into
one IMGT-numbered, structurally-consistent format.

**License: CC BY-NC 4.0 (non-commercial use only)** — see the LICENSE.txt this script
fetches. Source: https://naturalantibody.com/ab-design/ (Google Drive folder link on
that page). Requires gdown (see databases/src/_common.py's ensure_gdown()) since Google
Drive folder shares have no direct-URL alternative.

Usage:
    python3 databases/src/abdesign_db.py
"""

import sys
import tarfile
from pathlib import Path

from _common import DATABASES_DIR, gdown_folder

ABDESIGN_DIR = DATABASES_DIR / "abdesign_db"
FOLDER_URL = "https://drive.google.com/drive/folders/1vF568s3Ge-fCQ-8-oAJIEH3BLr3O3k6H"


def fetch(out_dir: Path = ABDESIGN_DIR) -> Path:
    gdown_folder(FOLDER_URL, out_dir)

    # The folder's own README.md would collide with this source's tracked README.md
    # (our provenance doc, not theirs) — keep both, rename theirs.
    upstream_readme = out_dir / "README.md"
    if upstream_readme.exists():
        upstream_readme.rename(out_dir / "UPSTREAM_README.md")

    tarball = out_dir / "abdesign.tar.gz"
    if tarball.exists():
        struct_dir = out_dir / "AbDesign_structures"
        print(f"[abdesign_db] extracting {tarball.name} -> {struct_dir}", file=sys.stderr)
        struct_dir.mkdir(exist_ok=True)
        with tarfile.open(tarball, mode="r:gz") as tar:
            tar.extractall(struct_dir, filter="data")
        tarball.unlink()

    return out_dir


if __name__ == "__main__":
    fetch()
