#!/usr/bin/env python3
"""Fetch sequences for a curated reference antibody-antigen complex (reference_targets.py).

Caches raw FASTA per PDB id under experiments/reference_data/ so repeat runs (e.g. one
per --backend in score_reference.py) don't refetch the same entry.

Usage:
    python3 experiments/fetch_reference.py --name hel
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import fetch_target  # noqa: E402

import reference_targets  # noqa: E402

REFERENCE_DATA_DIR = Path(__file__).resolve().parent / "reference_data"


def _cached_pdb_fasta(pdb_id: str) -> str:
    REFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = REFERENCE_DATA_DIR / f"{pdb_id.upper()}.fasta"
    if cache_path.exists():
        return cache_path.read_text()
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}"
    text = fetch_target.fetch(url)
    cache_path.write_text(text)
    return text


def fetch_reference(name: str, refresh: bool = False) -> dict[str, str]:
    complex_ = reference_targets.REFERENCE_COMPLEXES[name]

    if refresh:
        for pdb_id in {complex_.antigen_pdb, complex_.heavy_pdb, complex_.light_pdb}:
            cache_path = REFERENCE_DATA_DIR / f"{pdb_id.upper()}.fasta"
            cache_path.unlink(missing_ok=True)

    def chain_seq(pdb_id: str, chain: str) -> str:
        entries = fetch_target.parse_fasta(_cached_pdb_fasta(pdb_id))
        _, seq = fetch_target.select_chain(entries, chain)
        return seq

    return {
        "antigen_seq": chain_seq(complex_.antigen_pdb, complex_.antigen_chain),
        "heavy_seq": chain_seq(complex_.heavy_pdb, complex_.heavy_chain),
        "light_seq": chain_seq(complex_.light_pdb, complex_.light_chain),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, choices=list(reference_targets.REFERENCE_COMPLEXES))
    parser.add_argument("--refresh", action="store_true", help="Ignore cache, refetch from RCSB")
    args = parser.parse_args()

    seqs = fetch_reference(args.name, refresh=args.refresh)
    for key, seq in seqs.items():
        print(f"[fetch_reference] {key}: {len(seq)} residues")
        print(seq)


if __name__ == "__main__":
    main()
