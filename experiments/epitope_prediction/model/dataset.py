#!/usr/bin/env python3
"""Assembles per-structure (features, graph, labels) entries and caches them to disk —
shared by baseline.py (GBT), gnn.py (Model A/B ensembles), and eval/classifier_metrics.py
so features and ESM2 embeddings (the expensive part) are computed once, not once per
model trained.

Usage:
    python3 experiments/epitope_prediction/model/dataset.py build --split train
    python3 experiments/epitope_prediction/model/dataset.py build --split dev
    python3 experiments/epitope_prediction/model/dataset.py build --split test
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esm2_features import compute_esm2_embeddings  # noqa: E402
from features import build_geometric_features, group_residues  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from interface_labels import REPO_ROOT, compute_interface_labels, get_chain_atoms  # noqa: E402
import train_labels as train_labels_mod  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent / ".dataset_cache"
SPLITS_DIR = REPO_ROOT / "databases" / "splits"


def _residue_key_str(key: tuple[str, str, str]) -> str:
    return f"{key[0]}:{key[1]}:{key[2]}"


def build_entry(pdb_id: str, label_source: str, compute_esm2: bool = True) -> dict | None:
    """label_source: "train" (AACDB-derived, via train_labels.py's per-structure dict)
    or "eval" (our own coordinate-based interface_labels.py, computed fresh -- never
    AACDB for eval, per PLAN.md's coverage-gap fix)."""
    chains = get_chain_atoms(pdb_id)
    if chains is None:
        return None
    _, antigen_atoms = chains
    if not antigen_atoms:
        return None
    residues = group_residues(antigen_atoms)
    if len(residues) < 2:
        return None

    geo = build_geometric_features(pdb_id)
    if geo is None:
        return None

    if label_source == "eval":
        raw_labels = compute_interface_labels(pdb_id)
        label_by_key = {k: v for k, v in raw_labels.items()}
    else:
        raise ValueError("label_source='train' handled by caller (needs the shared AACDB cache)")

    labels = torch.tensor([1.0 if label_by_key.get(k, False) else 0.0 for k in geo["residue_keys"]], dtype=torch.float32)

    entry = {
        "pdb_id": pdb_id,
        "residue_keys": [_residue_key_str(k) for k in geo["residue_keys"]],
        "node_features": geo["node_features"],
        "edge_index": geo["edge_index"],
        "labels": labels,
    }
    if compute_esm2:
        emb = compute_esm2_embeddings(pdb_id, residues)
        entry["esm2_embeddings"] = emb if emb is not None else torch.zeros(len(residues), 640)
    return entry


def build_train_entries(compute_esm2: bool = True) -> list[dict]:
    aacdb_labels = train_labels_mod.load_train_labels()
    entries = []
    n_skipped = 0
    for i, (pdb_id, label_by_str_key) in enumerate(aacdb_labels.items()):
        geo = build_geometric_features(pdb_id)
        if geo is None:
            n_skipped += 1
            continue
        chains = get_chain_atoms(pdb_id)
        _, antigen_atoms = chains
        residues = group_residues(antigen_atoms)

        keys_str = [_residue_key_str(k) for k in geo["residue_keys"]]
        labels = torch.tensor([1.0 if label_by_str_key.get(k, False) else 0.0 for k in keys_str], dtype=torch.float32)

        entry = {
            "pdb_id": pdb_id,
            "residue_keys": keys_str,
            "node_features": geo["node_features"],
            "edge_index": geo["edge_index"],
            "labels": labels,
        }
        if compute_esm2:
            emb = compute_esm2_embeddings(pdb_id, residues)
            entry["esm2_embeddings"] = emb if emb is not None else torch.zeros(len(residues), 640)
        entries.append(entry)
        if (i + 1) % 200 == 0:
            print(f"[dataset] {i + 1}/{len(aacdb_labels)} train structures processed ({n_skipped} skipped)", file=sys.stderr)

    print(f"[dataset] built {len(entries)} train entries, {n_skipped} skipped (no usable antigen)", file=sys.stderr)
    return entries


def build_eval_entries(split: str, compute_esm2: bool = True) -> list[dict]:
    with open(SPLITS_DIR / f"{split}.txt") as f:
        pdb_ids = f.read().split()

    entries = []
    n_skipped = 0
    for i, pdb_id in enumerate(pdb_ids):
        entry = build_entry(pdb_id, label_source="eval", compute_esm2=compute_esm2)
        if entry is None:
            n_skipped += 1
            continue
        entries.append(entry)
        if (i + 1) % 200 == 0:
            print(f"[dataset] {i + 1}/{len(pdb_ids)} {split} structures processed ({n_skipped} skipped)", file=sys.stderr)

    print(f"[dataset] built {len(entries)} {split} entries, {n_skipped} skipped (no usable antigen)", file=sys.stderr)
    return entries


def cache_path(split: str) -> Path:
    return CACHE_DIR / f"{split}.pt"


def build_and_cache(split: str) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if split == "train":
        entries = build_train_entries()
    elif split in ("dev", "test"):
        entries = build_eval_entries(split)
    else:
        raise ValueError(f"unknown split {split!r}")
    torch.save(entries, cache_path(split))
    print(f"[dataset] cached {len(entries)} entries to {cache_path(split)}", file=sys.stderr)
    return entries


def load_cached(split: str) -> list[dict]:
    if not cache_path(split).exists():
        return build_and_cache(split)
    return torch.load(cache_path(split), weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["build"])
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"])
    args = parser.parse_args()
    build_and_cache(args.split)


if __name__ == "__main__":
    main()
