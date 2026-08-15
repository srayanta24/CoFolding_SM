#!/usr/bin/env python3
"""Predict a single protein chain's 3D structure from sequence alone (no partner).

This is the building block that makes BoltzGen antibody/binder design
(Skill B) usable from a bare sequence: BoltzGen needs a real 3D structure
to condition on, but not every target of interest has a solved one. This
script folds the target by itself (not against any partner) via Boltz-2,
producing a CIF file that stands in for an experimental structure.

IMPORTANT CAVEAT: the output is a PREDICTION, not an experimental
structure. Designing a binder against a low-confidence region (a floppy
loop, an unresolved-in-reality signal peptide/TM segment) will silently
inherit that uncertainty into the design. This script reports the
predicted structure's average pLDDT specifically so that judgment call is
visible, not hidden — always look at it before proceeding to Skill B.

Usage:
    python3 scripts/predict_structure.py --target-id P09758 --name trop2_predicted
    python3 scripts/predict_structure.py --target-seq MKT... --name my_target
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import fetch_target
from _common import REPO_ROOT, WEIGHTS_DIR, produced_structure, venv_bin

TIMEOUT_S = 40 * 60
LOW_CONFIDENCE_PLDDT = 70.0  # below this, flag the structure as unreliable


def resolve_target_sequence(target_id: str | None, target_seq: str | None, target_chain: str | None) -> str:
    if target_seq:
        return target_seq.strip()
    source = fetch_target.identify_source(target_id)
    entries = (
        fetch_target.fetch_uniprot(target_id) if source == "uniprot" else fetch_target.fetch_pdb(target_id)
    )
    header, seq = fetch_target.select_chain(entries, target_chain)
    print(f"[predict_structure] fetched target ({source}): {header} ({len(seq)} residues)", file=sys.stderr)
    return seq


def build_single_chain_config(target_seq: str, out_path: Path) -> Path:
    config = {"version": 1, "sequences": [{"protein": {"id": "A", "sequence": target_seq}}]}
    out_path.write_text(json.dumps(config))
    return out_path


def run_boltz2_single_chain(config_path: Path, out_dir: Path) -> bool:
    cmd = [
        venv_bin("boltz2", "boltz"), "predict", str(config_path),
        "--use_msa_server",
        "--num_workers", "0",
        "--no_kernels",
        "--out_dir", str(out_dir),
        "--cache", str(WEIGHTS_DIR / "boltz2"),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, timeout=TIMEOUT_S)
    return produced_structure(out_dir)


def average_plddt(confidence_json: Path) -> float | None:
    data = json.loads(confidence_json.read_text())
    # Boltz-2 reports complex_plddt for the whole prediction; for a single
    # chain that IS the per-target plddt.
    return data.get("complex_plddt")


def predict(target_seq: str, name: str, out_root: Path) -> tuple[Path, float | None]:
    """Returns (path to predicted CIF, average pLDDT or None)."""
    out_root.mkdir(parents=True, exist_ok=True)
    config_path = build_single_chain_config(target_seq, out_root / f"{name}_fold.yaml")
    out_dir = out_root / "predicted_structure"
    print(f"\n{'=' * 60}\n[predict_structure] folding '{name}' ({len(target_seq)} residues)\n{'=' * 60}")
    ok = run_boltz2_single_chain(config_path, out_dir)
    if not ok:
        sys.exit(f"[predict_structure] FAILED — no structure produced for '{name}'")

    cif_files = sorted(out_dir.rglob("*.cif"))
    conf_files = sorted(out_dir.rglob("confidence_*.json"))
    cif_path = out_root / f"{name}_predicted_target.cif"
    cif_path.write_bytes(cif_files[0].read_bytes())

    plddt = average_plddt(conf_files[0]) if conf_files else None
    if plddt is not None:
        pct = plddt * 100 if plddt <= 1.0 else plddt
        print(f"[predict_structure] predicted structure avg pLDDT: {pct:.1f}")
        if pct < LOW_CONFIDENCE_PLDDT:
            print(
                f"[predict_structure] WARNING: pLDDT below {LOW_CONFIDENCE_PLDDT} — this "
                f"structure has low-confidence regions (common for signal peptides, "
                f"transmembrane segments, disordered loops). If you know the folded "
                f"domain boundaries, re-run with just that subsequence rather than "
                f"designing against the whole thing.",
                file=sys.stderr,
            )
    print(f"[predict_structure] wrote {cif_path}")
    return cif_path, plddt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-id", help="UniProt accession or PDB ID to fetch")
    target.add_argument("--target-seq", help="Raw target protein sequence (skips fetching)")
    parser.add_argument("--target-chain", help="For multi-chain PDB targets, pick one (default: longest)")
    parser.add_argument("--name", default="target", help="Job name, used for output paths")
    parser.add_argument("--out-dir", default=None, help="Output root (default: data/designs/<name>)")
    args = parser.parse_args()

    target_seq = resolve_target_sequence(args.target_id, args.target_seq, args.target_chain)
    out_root = Path(args.out_dir) if args.out_dir else REPO_ROOT / "data" / "designs" / args.name
    predict(target_seq, args.name, out_root)


if __name__ == "__main__":
    main()
