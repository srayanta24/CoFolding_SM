#!/usr/bin/env python3
"""Build leak-free train/dev/test splits for anything trained or evaluated as part of
the design-pipeline improvement work (IMPROVE_DESIGN.md).

Two independent leakage checks, combined:

1. **Temporal** (mirrors BoltzGen's own training filter exactly — verified identical
   across every training config in src/boltzgen/: `DateFilter(date="2023-06-01",
   ref="released")`). Structures released on or before that date are presumptively in
   BoltzGen's own training data; after it, presumptively not.
2. **Sequence-identity clustering** (antigen sequences, MMseqs2, default 40% identity).
   Temporal separation alone isn't enough: the same antigen can reappear in a
   later-dated PDB entry (revised crystal form, different complex partner, etc.), which
   would let a near-duplicate leak into a "clean" test set. Any post-cutoff structure
   whose antigen shares a cluster with a pre-cutoff structure is excluded from the test
   set rather than trusted.

Antigen sequences are extracted locally from databases/sabdab/structures/'s own mmCIF
files (a small parser for just the `_entity_poly` loop block — not a full CIF parser,
no gemmi/biotite dependency) rather than re-fetched from RCSB.

Usage:
    python3 databases/src/build_splits.py
"""

import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from _common import DATABASES_DIR, ensure_mmseqs2

SABDAB_DIR = DATABASES_DIR / "sabdab"
SPLITS_DIR = DATABASES_DIR / "splits"
CUTOFF_DATE = "2023-06-01"
MIN_SEQ_ID = 0.4
DEV_FRACTION = 0.15

ENTITY_POLY_ROW_RE = re.compile(
    r"^(\d+)\s+\S+\s+([A-Za-z0-9,]+)\s+(\S+)\s*$"
)


def load_summary() -> list[dict]:
    with open(SABDAB_DIR / "summary.csv", newline="") as f:
        return list(csv.DictReader(f))


def unique_antigen_structures(rows: list[dict]) -> dict[str, dict]:
    """One row per unique PDB id that has an antigen_chain set. If a PDB id appears in
    multiple summary rows (multiple Fab/nanobody instances against the same antigen),
    they agree on date/antigen_chain in practice — take the first."""
    by_pdb = {}
    for r in rows:
        pdb = r["PDB"]
        if not r["antigen_chain"].strip():
            continue
        if pdb not in by_pdb:
            by_pdb[pdb] = r
    return by_pdb


def parse_entity_poly(cif_path: Path) -> list[tuple[str, list[str], str]]:
    """Returns [(entity_id, [chain_ids], sequence), ...] from the _entity_poly loop.
    Small targeted parser, not a general CIF parser: finds the loop_ header block for
    _entity_poly and reads fixed-width whitespace-separated rows until a blank line."""
    text = cif_path.read_text()
    marker = "_entity_poly.pdbx_seq_one_letter_code"
    idx = text.find(marker)
    if idx == -1:
        return []
    # Skip past the rest of the marker's own line (the loop_ header) to the first
    # actual data row — the marker line has nothing useful after the field name itself.
    newline_idx = text.find("\n", idx)
    if newline_idx == -1:
        return []
    lines = text[newline_idx + 1:].splitlines()
    results = []
    for line in lines:
        if not line.strip():
            break
        m = ENTITY_POLY_ROW_RE.match(line.strip())
        if not m:
            continue
        entity_id, chains_str, seq = m.groups()
        results.append((entity_id, chains_str.split(","), seq))
    return results


def extract_antigen_sequences(antigen_structures: dict[str, dict]) -> tuple[dict[str, str], list[str]]:
    sequences = {}
    failures = []
    for pdb, row in antigen_structures.items():
        cif_path = SABDAB_DIR / "structures" / pdb / f"{pdb}_sabdab.cif"
        if not cif_path.exists():
            failures.append(f"{pdb}: structure file missing")
            continue
        antigen_chains = set(row["antigen_chain"].split("|"))
        entities = parse_entity_poly(cif_path)
        if not entities:
            failures.append(f"{pdb}: no _entity_poly block parsed")
            continue
        match = None
        for entity_id, chains, seq in entities:
            if antigen_chains & set(chains):
                match = seq
                break
        if match is None:
            failures.append(f"{pdb}: antigen_chain {row['antigen_chain']!r} not found among parsed entities")
            continue
        sequences[pdb] = match
    return sequences, failures


def temporal_partition(antigen_structures: dict[str, dict]) -> dict[str, str]:
    from datetime import datetime

    cutoff = datetime.strptime(CUTOFF_DATE, "%Y-%m-%d")
    partition = {}
    for pdb, row in antigen_structures.items():
        d = row["date"].strip()
        try:
            dt = datetime.strptime(d, "%Y/%m/%d")
        except ValueError:
            partition[pdb] = "unknown_date"
            continue
        partition[pdb] = "train_era" if dt <= cutoff else "post_cutoff"
    return partition


def cluster_sequences(sequences: dict[str, str]) -> dict[str, str]:
    """Returns pdb_id -> cluster representative pdb_id, via mmseqs easy-cluster."""
    mmseqs = ensure_mmseqs2()
    work_dir = SPLITS_DIR / ".mmseqs_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = work_dir / "antigens.fasta"
    with open(fasta_path, "w") as f:
        for pdb, seq in sequences.items():
            f.write(f">{pdb}\n{seq}\n")

    out_prefix = work_dir / "cluster"
    tmp_dir = work_dir / "tmp"
    print(f"[splits] clustering {len(sequences)} antigen sequences at {MIN_SEQ_ID:.0%} identity", file=sys.stderr)
    subprocess.run(
        [mmseqs, "easy-cluster", str(fasta_path), str(out_prefix), str(tmp_dir),
         "--min-seq-id", str(MIN_SEQ_ID), "-v", "1"],
        check=True,
    )

    cluster_of = {}
    with open(f"{out_prefix}_cluster.tsv") as f:
        for line in f:
            rep, member = line.strip().split("\t")
            cluster_of[member] = rep
    return cluster_of


def build_split(antigen_structures: dict[str, dict], partition: dict[str, str], cluster_of: dict[str, str]) -> dict[str, list[str]]:
    clusters_with_train_era = set()
    for pdb, bucket in partition.items():
        if bucket == "train_era" and pdb in cluster_of:
            clusters_with_train_era.add(cluster_of[pdb])

    buckets = defaultdict(list)
    for pdb, bucket in partition.items():
        if bucket in ("train_era", "unknown_date"):
            buckets[bucket].append(pdb)
            continue
        # post_cutoff: never trust "clean" without a verified cluster assignment — a
        # structure whose antigen sequence we failed to extract was never checked for
        # redundancy against train_era, so it cannot be assumed safe for test/dev.
        rep = cluster_of.get(pdb)
        if rep is None:
            buckets["excluded_no_sequence"].append(pdb)
        elif rep in clusters_with_train_era:
            buckets["excluded_ambiguous"].append(pdb)
        else:
            buckets["post_cutoff_clean"].append(pdb)

    # Deterministic (hash-based, not random-seeded) dev/test split of the clean pool.
    dev, test = [], []
    for pdb in sorted(buckets.pop("post_cutoff_clean", [])):
        h = int(hashlib.sha256(pdb.encode()).hexdigest(), 16)
        (dev if (h % 100) < int(DEV_FRACTION * 100) else test).append(pdb)
    buckets["dev"] = dev
    buckets["test"] = test
    return buckets


def cross_reference_sources(buckets: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    pdb_to_bucket = {}
    for bucket, pdbs in buckets.items():
        for pdb in pdbs:
            pdb_to_bucket[pdb.replace("pdb_0000", "").upper()] = bucket

    def tally(pdb_ids: set[str]) -> dict[str, int]:
        counts = defaultdict(int)
        for pdb in pdb_ids:
            counts[pdb_to_bucket.get(pdb.upper(), "unknown")] += 1
        return dict(counts)

    result = {}

    aacdb_path = DATABASES_DIR / "aacdb" / "protein_table.txt"
    if aacdb_path.exists():
        with open(aacdb_path) as f:
            pdbs = {r["pdb"] for r in csv.DictReader(f, delimiter="\t")}
        result["aacdb"] = tally(pdbs)

    ab_bind_path = DATABASES_DIR / "ab_bind" / "AB-Bind_experimental_data.csv"
    if ab_bind_path.exists():
        with open(ab_bind_path, encoding="latin-1") as f:
            line0 = f.readline()
            reader = csv.DictReader(f, fieldnames=line0.lstrip("#").strip().split(","))
            pdbs = {r["PDB"] for r in reader if not r["PDB"].startswith("HM_")}
        result["ab_bind"] = tally(pdbs)

    return result


def main() -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_summary()
    antigen_structures = unique_antigen_structures(rows)
    print(f"[splits] {len(antigen_structures)} unique antigen-bound structures", file=sys.stderr)

    sequences, failures = extract_antigen_sequences(antigen_structures)
    print(f"[splits] extracted {len(sequences)} antigen sequences, {len(failures)} failures", file=sys.stderr)

    partition = temporal_partition(antigen_structures)
    cluster_of = cluster_sequences(sequences)
    buckets = build_split(antigen_structures, partition, cluster_of)

    for name, pdbs in buckets.items():
        (SPLITS_DIR / f"{name}.txt").write_text("\n".join(sorted(pdbs)) + "\n")

    # Keep only the small, human-inspectable cluster assignment (supports auditing
    # excluded_ambiguous entries, as done during verification); the rest of
    # .mmseqs_work/ (input FASTA, mmseqs DB/tmp files) is regenerable, not tracked.
    import shutil

    work_dir = SPLITS_DIR / ".mmseqs_work"
    cluster_tsv = work_dir / "cluster_cluster.tsv"
    if cluster_tsv.exists():
        shutil.copy(cluster_tsv, SPLITS_DIR / "antigen_clusters.tsv")
    shutil.rmtree(work_dir, ignore_errors=True)

    cross_ref = cross_reference_sources(buckets)

    summary = {
        "cutoff_date": CUTOFF_DATE,
        "min_seq_id": MIN_SEQ_ID,
        "dev_fraction": DEV_FRACTION,
        "total_antigen_structures": len(antigen_structures),
        "sequences_extracted": len(sequences),
        "extraction_failures": failures,
        "bucket_counts": {k: len(v) for k, v in buckets.items()},
        "cross_referenced_sources": cross_ref,
    }
    (SPLITS_DIR / "splits_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
