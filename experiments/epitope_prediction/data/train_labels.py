#!/usr/bin/env python3
"""AACDB-derived training labels, restricted to databases/splits/train_era.txt.

Positive-unlabeled framing (PLAN.md §3): AACDB's interacting_res_distance/ files only
record *observed* contacts, so there's no direct "confirmed negative" list from AACDB
itself. The full antigen-residue universe (needed to know what's "unlabeled") comes
from the same source data/interface_labels.py already parses — reused directly here
via get_chain_atoms() rather than re-deriving the auth-vs-label chain-id logic a second
time (that logic had one real bug already caught and fixed once, see
interface_labels.py's module docstring).

A PDB entry commonly has multiple AACDB files (multiple antibody copies in the
asymmetric unit) — unioned across all matches, same fix already verified necessary and
applied in interface_labels.py's sanity check.

Usage:
    python3 experiments/epitope_prediction/data/train_labels.py            # build + cache
    python3 experiments/epitope_prediction/data/train_labels.py --no-cache  # force rebuild
"""

import argparse
import json
import sys
from pathlib import Path

from interface_labels import AACDB_DIR, REPO_ROOT, get_chain_atoms

CACHE_PATH = Path(__file__).resolve().parent / ".train_labels_cache.json"


def _aacdb_positive_residues(pdb_id_upper: str) -> set[tuple[str, str]]:
    """{(chain, auth_seq_id)} for every antigen-side residue AACDB records in contact
    with any antibody copy, unioned across all of this PDB's AACDB files."""
    positives = set()
    for match in AACDB_DIR.glob(f"interacting_res_distance/{pdb_id_upper}_*_interacting_residues_distance.txt"):
        with open(match) as f:
            next(f)  # header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                _, antigen = parts[0], parts[1]
                chain, resname_num = antigen.split(":")
                num = "".join(ch for ch in resname_num if ch.isdigit())
                positives.add((chain, num))
    return positives


def build_train_labels() -> dict[str, dict[str, bool]]:
    """Returns {pdb_id: {"auth_asym_id:auth_seq_id:auth_comp_id": is_epitope_residue}}
    (string-joined key, not a tuple — JSON has no tuple keys) for every train_era
    structure with at least one AACDB annotation."""
    with open(REPO_ROOT / "databases" / "splits" / "train_era.txt") as f:
        train_era = sorted(f.read().split())

    result = {}
    n_with_aacdb = 0
    for pdb_id in train_era:
        pdb_upper = pdb_id.replace("pdb_0000", "").upper()
        positives = _aacdb_positive_residues(pdb_upper)
        if not positives:
            continue

        chains = get_chain_atoms(pdb_id)
        if chains is None:
            continue
        _, antigen_atoms = chains
        if not antigen_atoms:
            continue

        labels = {}
        for atom in antigen_atoms:
            key = f"{atom.auth_asym_id}:{atom.auth_seq_id}:{atom.auth_comp_id}"
            is_positive = (atom.auth_asym_id, atom.auth_seq_id) in positives
            labels[key] = labels.get(key, False) or is_positive

        if any(labels.values()):
            result[pdb_id] = labels
            n_with_aacdb += 1

    print(f"[train_labels] {n_with_aacdb} / {len(train_era)} train_era structures have usable AACDB labels", file=sys.stderr)
    return result


def load_train_labels(use_cache: bool = True) -> dict[str, dict[str, bool]]:
    if use_cache and CACHE_PATH.exists():
        print(f"[train_labels] loading cached labels from {CACHE_PATH}", file=sys.stderr)
        return json.loads(CACHE_PATH.read_text())
    labels = build_train_labels()
    CACHE_PATH.write_text(json.dumps(labels))
    print(f"[train_labels] cached to {CACHE_PATH}", file=sys.stderr)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-cache", action="store_true", help="Force rebuild even if a cache file exists")
    args = parser.parse_args()

    labels = load_train_labels(use_cache=not args.no_cache)
    n_residues = sum(len(v) for v in labels.values())
    n_positive = sum(sum(v.values()) for v in labels.values())
    print(f"[train_labels] {len(labels)} structures, {n_residues} residues, {n_positive} positive ({n_positive/n_residues:.1%})")


if __name__ == "__main__":
    main()
