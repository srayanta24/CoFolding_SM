#!/usr/bin/env python3
"""Design a novel antibody/nanobody against a target from SEQUENCE ALONE.

Skill B (BoltzGen) normally needs a real experimental structure for the
target. This script removes that requirement: it first predicts the
target's structure with Boltz-2 (scripts/predict_structure.py), then wires
that prediction into a BoltzGen design spec referencing the bundled Fab or
nanobody scaffolds, validates it, and (only if you pass --launch) runs the
actual design campaign.

Two layers of prediction uncertainty are stacked here — the target fold
itself is predicted, not experimental, and then the binder is designed
against that prediction. The average pLDDT of the predicted structure is
printed specifically so you can judge whether that's trustworthy before
spending GPU time on step 2. If you have a real structure available
(fetch a PDB entry instead), prefer that — this script is for when you
don't.

Usage:
    python3 scripts/design_binder.py --target-id P09758 --name trop2_from_seq
    python3 scripts/design_binder.py --target-seq MKT... --name my_target \\
        --protocol nanobody-anything --num_designs 50 --budget 5 --launch
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _common import REPO_ROOT, WEIGHTS_DIR, venv_bin
from predict_structure import predict, resolve_target_sequence

BOLTZGEN_EXAMPLES = REPO_ROOT / "data" / "boltzgen_examples" / "repo" / "example"

FAB_SCAFFOLDS = [
    "adalimumab.6cr1", "belimumab.5y9k", "dupilumab.6wgb", "golimumab.5yoy",
    "guselkumab.4m6m", "nirsevimab.5udc", "sarilumab.8iow", "secukinumab.6wio",
    "tezepelumab.5j13", "tralokinumab.5l6y", "ustekinumab.3hmw", "mab1.3h42",
    "necitumumab.6b3s", "crenezumab.5vzy",
]
NANOBODY_SCAFFOLDS = [
    "7eow", "7xl0", "8coh", "8z8v", "gontivimab", "isecarosmab", "sonelokimab",
]
PROTOCOL_TO_SCAFFOLDS = {
    "antibody-anything": ("fab_scaffolds", FAB_SCAFFOLDS),
    "nanobody-anything": ("nanobody_scaffolds", NANOBODY_SCAFFOLDS),
}


def write_design_spec(target_cif_name: str, protocol: str, out_path: Path) -> Path:
    scaffold_dir, scaffold_names = PROTOCOL_TO_SCAFFOLDS[protocol]
    scaffold_paths = [
        f"../../boltzgen_examples/repo/example/{scaffold_dir}/{name}.yaml" for name in scaffold_names
    ]
    lines = [
        "entities:",
        "    - file:",
        f"        path: {target_cif_name}",
        "        include:",
        "            - chain:",
        "                id: A",
        "    - file:",
        "        path:",
    ]
    lines += [f"            - {p}" for p in scaffold_paths]
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def run_boltzgen_check(spec_path: Path, out_dir: Path) -> None:
    cmd = [
        venv_bin("boltzgen", "boltzgen"), "check", spec_path.name,
        "--output", str(out_dir), "--cache", str(WEIGHTS_DIR / "boltzgen"),
    ]
    result = subprocess.run(cmd, cwd=spec_path.parent)
    if result.returncode != 0:
        sys.exit("[design_binder] boltzgen check FAILED — fix the design spec before proceeding")


def run_boltzgen_campaign(
    spec_path: Path, protocol: str, num_designs: int, budget: int
) -> None:
    cmd = [
        venv_bin("boltzgen", "boltzgen"), "run", spec_path.name,
        "--output", "run1",
        "--protocol", protocol,
        "--num_designs", str(num_designs),
        "--budget", str(budget),
        "--num_workers", "0",  # avoids the CUDA-fork/DataLoader deadlock (README.md Troubleshooting)
        "--cache", str(WEIGHTS_DIR / "boltzgen"),
    ]
    print(f"\n{'=' * 60}\n[design_binder] launching campaign ({num_designs} designs, budget {budget})\n{'=' * 60}")
    subprocess.run(cmd, cwd=spec_path.parent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-id", help="UniProt accession or PDB ID to fetch")
    target.add_argument("--target-seq", help="Raw target protein sequence (skips fetching)")
    parser.add_argument("--target-chain", help="For multi-chain PDB targets, pick one (default: longest)")
    parser.add_argument("--name", default="binder_design", help="Job name, used for output paths")
    parser.add_argument(
        "--protocol", choices=list(PROTOCOL_TO_SCAFFOLDS), default="antibody-anything",
        help="antibody-anything (Fab, two chains) or nanobody-anything (single domain)",
    )
    parser.add_argument("--num_designs", type=int, default=50, help="BoltzGen's own recommended first-pass scale")
    parser.add_argument("--budget", type=int, default=5, help="Final ranked/diverse set size")
    parser.add_argument(
        "--launch", action="store_true",
        help="Actually run the (potentially hours-long) design campaign. Without this, "
             "the script stops after validating the spec and prints the command to launch it.",
    )
    args = parser.parse_args()

    target_seq = resolve_target_sequence(args.target_id, args.target_seq, args.target_chain)
    out_root = REPO_ROOT / "data" / "designs" / args.name
    cif_path, plddt = predict(target_seq, args.name, out_root)

    spec_path = write_design_spec(cif_path.name, args.protocol, out_root / f"{args.name}.yaml")
    print(f"[design_binder] wrote design spec: {spec_path}")

    check_out = out_root / "check_output"
    run_boltzgen_check(spec_path, check_out)
    print(f"[design_binder] spec validated — visualization at {check_out}")

    launch_cmd = (
        f"cd {out_root} && source {REPO_ROOT}/.venvs/boltzgen/bin/activate && "
        f"boltzgen run {spec_path.name} --output run1 --protocol {args.protocol} "
        f"--num_designs {args.num_designs} --budget {args.budget} --num_workers 0 "
        f"--cache {WEIGHTS_DIR / 'boltzgen'}"
    )

    plddt_pct = plddt * 100 if plddt is not None and plddt <= 1.0 else plddt

    if args.launch:
        run_boltzgen_campaign(spec_path, args.protocol, args.num_designs, args.budget)
    else:
        print(
            f"\n[design_binder] Ready but not launched (pass --launch to run automatically).\n"
            f"Predicted target pLDDT: {plddt_pct:.1f}\n"
            f"To launch manually:\n  {launch_cmd}"
        )


if __name__ == "__main__":
    main()
