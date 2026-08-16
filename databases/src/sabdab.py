#!/usr/bin/env python3
"""Fetch the local SAbDab database: bulk antibody-antigen structures, summary
metadata, and a known-benchmark affinity dataset.

All three sources were verified live (not guessed) while building this script:

- Structures + summary come from SAbDab2's own bulk API
  (https://sabdab.opig.stats.ox.ac.uk/api/download/all-structures|all-summary),
  found by grepping its JS bundle for fetch() call sites — the project's Zenodo
  mirror (zenodo.org/records/20083995) only has train/test split CSVs, not the
  structures themselves.
- Affinity data is TDC's AntibodyAff/Protein_SAbDab benchmark (493 antibody-antigen
  pairs with experimental Kd). Its canonical host, Harvard Dataverse, sits behind an
  AWS WAF JS challenge that blocks plain HTTP clients (verified: curl, including with
  a browser User-Agent, gets `x-amzn-waf-action: challenge` instead of the file) — so
  this pulls a direct, unblocked Zenodo mirror of the same file instead of adding a
  PyTDC dependency that would likely hit the same block.

See README.md for provenance detail, real dataset sizes, and the join key between
affinity/*.csv and summary.csv.

Usage:
    python3 databases/src/sabdab.py                          # fetch all three
    python3 databases/src/sabdab.py --only summary --only affinity   # skip the slow 2.7GB step
    python3 databases/src/sabdab.py --only structures         # just the slow step
"""

import argparse
import sys
from pathlib import Path
import tarfile

from _common import DATABASES_DIR, download

SABDAB_DIR = DATABASES_DIR / "sabdab"
SUMMARY_URL = "https://sabdab.opig.stats.ox.ac.uk/api/download/all-summary"
STRUCTURES_URL = "https://sabdab.opig.stats.ox.ac.uk/api/download/all-structures"
AFFINITY_URL = (
    "https://zenodo.org/records/13120765/files/"
    "antibody_affinity_protein_sabdab.csv?download=1"
)


def fetch_summary(out: Path = SABDAB_DIR / "summary.csv") -> Path:
    print(f"[sabdab] fetching summary metadata -> {out}", file=sys.stderr)
    download(SUMMARY_URL, out)
    return out


def fetch_affinity(out: Path = SABDAB_DIR / "affinity" / "antibody_affinity_protein_sabdab.csv") -> Path:
    print(f"[sabdab] fetching affinity benchmark -> {out}", file=sys.stderr)
    download(AFFINITY_URL, out)
    return out


def fetch_structures(out_dir: Path = SABDAB_DIR / "structures") -> Path:
    tmp_tgz = SABDAB_DIR / ".sabdab_all_structures.tgz.partial"
    print(f"[sabdab] fetching structures archive (2.7GB, resumable) -> {tmp_tgz}", file=sys.stderr)
    download(STRUCTURES_URL, tmp_tgz, resume=True)

    print(f"[sabdab] extracting -> {out_dir}", file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tmp_tgz, mode="r:gz") as tar:
        tar.extractall(out_dir, filter="data")

    tmp_tgz.unlink()
    num_entries = sum(1 for p in out_dir.iterdir() if p.is_dir())
    print(f"[sabdab] extracted {num_entries} structure entries", file=sys.stderr)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only", action="append", choices=["summary", "affinity", "structures"],
        help="Fetch only these steps (repeatable). Default: all three, cheap steps first.",
    )
    args = parser.parse_args()
    steps = args.only or ["summary", "affinity", "structures"]

    if "summary" in steps:
        fetch_summary()
    if "affinity" in steps:
        fetch_affinity()
    if "structures" in steps:
        fetch_structures()


if __name__ == "__main__":
    main()
