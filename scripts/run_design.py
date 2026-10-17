#!/usr/bin/env python3
"""Cofold a target (fetched or supplied) against a partner, via Boltz-2 and/or OpenFold3.

Usage:
    python3 scripts/run_design.py --target-id P69905 \\
        --partner-modality small_molecule --partner-value "CC(=O)Oc1ccccc1C(=O)O" \\
        --name aspirin_test

    python3 scripts/run_design.py --target-seq MKTAYIAK... \\
        --partner-modality protein --partner-value MSEQ... \\
        --backend boltz2 --name my_complex

Modalities antibody/peptide are modeled as plain protein chain(s) — neither
backend has CDR-specific or specialized handling for them yet (DESIGN.md §5).
For 'antibody', --partner-value accepts either a single sequence (one chain)
or "HEAVY_SEQ,LIGHT_SEQ" (two chains) — a real Fab paratope is formed by both
chains together, so a single-chain approximation is only meaningful for a
nanobody/VHH. Passing both chains from a BoltzGen-designed antibody (its
full_sequence_1/full_sequence_2 CSV columns) is the correct way to validate
one of its outputs.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import fetch_target
from _common import REPO_ROOT, WEIGHTS_DIR, produced_structure, venv_bin

TIMEOUT_S = 40 * 60  # weight downloads + MSA server queueing can take a while

MODALITY_TO_OF3 = {
    "small_molecule": "LIGAND",
    "protein": "PROTEIN",
    "antibody": "PROTEIN",
    "peptide": "PROTEIN",
    "rna": "RNA",
    "dna": "DNA",
}
MODALITY_TO_BOLTZ = {
    "small_molecule": "ligand",
    "protein": "protein",
    "antibody": "protein",
    "peptide": "protein",
    "rna": "rna",
    "dna": "dna",
}
NO_SPECIAL_HANDLING = {"antibody", "peptide"}


def resolve_target_sequence(args: argparse.Namespace) -> str:
    if args.target_seq:
        return args.target_seq.strip()
    source = fetch_target.identify_source(args.target_id)
    entries = (
        fetch_target.fetch_uniprot(args.target_id)
        if source == "uniprot"
        else fetch_target.fetch_pdb(args.target_id)
    )
    header, seq = fetch_target.select_chain(entries, args.target_chain)
    print(f"[run_design] fetched target ({source}): {header} ({len(seq)} residues)", file=sys.stderr)
    return seq


def split_partner_chains(modality: str, partner_value: str) -> list[str]:
    """Antibody partner-value may be 'HEAVY,LIGHT' for a real two-chain Fab
    paratope; everything else is a single chain. A literal comma inside a
    SMILES string never happens (comma isn't valid SMILES syntax), so this
    split is unambiguous for small_molecule too."""
    if modality == "antibody" and "," in partner_value:
        return [s.strip() for s in partner_value.split(",")]
    return [partner_value]


def build_boltz_config(target_seq: str, modality: str, partner_value: str, out_path: Path,
                       use_msa: bool = True) -> Path:
    """use_msa=False puts every protein chain in Boltz-2's single-sequence mode
    (`msa: empty`). That is what BoltzGen's own folding stage does -- it refolds
    designs with the target template and *no* MSAs (paper sec 3.5) -- so any
    scoring meant to be comparable with a campaign's own metrics should match it.
    An MSA also makes no sense for a de novo designed binder, which has no
    evolutionary family to align against."""
    partner_key = MODALITY_TO_BOLTZ[modality]
    chains = split_partner_chains(modality, partner_value)
    chain_ids = ["B", "C"][: len(chains)]
    target_entry = {"id": "A", "sequence": target_seq}
    if not use_msa:
        target_entry["msa"] = "empty"
    sequences = [{"protein": target_entry}]
    for cid, seq in zip(chain_ids, chains):
        entry = {"id": cid, ("smiles" if partner_key == "ligand" else "sequence"): seq}
        if not use_msa and partner_key == "protein":
            entry["msa"] = "empty"
        sequences.append({partner_key: entry})
    config = {"version": 1, "sequences": sequences}
    # JSON is valid YAML 1.2, so writing plain JSON avoids needing PyYAML and
    # any string-escaping edge cases (SMILES can contain quotes/backslashes).
    out_path.write_text(json.dumps(config))
    return out_path


def build_of3_config(target_seq: str, modality: str, partner_value: str, name: str, out_path: Path,
                     use_msa: bool = True) -> Path:
    of3_type = MODALITY_TO_OF3[modality]
    chains = split_partner_chains(modality, partner_value)
    chain_ids = ["B", "C"][: len(chains)]
    partner_chains = []
    for cid, seq in zip(chain_ids, chains):
        chain = {"molecule_type": of3_type, "chain_ids": [cid]}
        chain["smiles" if of3_type == "LIGAND" else "sequence"] = seq
        partner_chains.append(chain)
    config = {
        "seeds": [42],
        "queries": {
            name: {
                "chains": [
                    {"molecule_type": "PROTEIN", "chain_ids": ["A"], "sequence": target_seq},
                    *partner_chains,
                ],
                "use_msas": use_msa,
            }
        },
    }
    out_path.write_text(json.dumps(config))
    return out_path


def run_boltz2(config_path: Path, out_dir: Path, use_msa: bool = True) -> bool:
    cmd = [
        venv_bin("boltz2", "boltz"), "predict", str(config_path),
        *(["--use_msa_server"] if use_msa else []),
        "--num_workers", "0",   # avoids a DataLoader/CUDA fork deadlock (DESIGN.md §3)
        "--no_kernels",         # cuequivariance-ops-torch has no aarch64 build (DESIGN.md §3)
        "--out_dir", str(out_dir),
        "--cache", str(WEIGHTS_DIR / "boltz2"),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, timeout=TIMEOUT_S)
    return produced_structure(out_dir)


def run_openfold3(config_path: Path, out_dir: Path, use_msa: bool = True) -> bool:
    ckpt = WEIGHTS_DIR / "openfold3" / "of3-p2-155k.pt"
    cmd = [
        venv_bin("openfold3", "run_openfold"), "predict",
        "--query-json", str(config_path),
        "--use-msa-server", "true" if use_msa else "false",
        "--output-dir", str(out_dir),
    ]
    if ckpt.exists():
        cmd += ["--inference-ckpt-path", str(ckpt)]
    subprocess.run(cmd, cwd=REPO_ROOT, timeout=TIMEOUT_S, input="yes\n" * 10, text=True)
    return produced_structure(out_dir)


def report_confidence(out_dir: Path, backend: str) -> None:
    for cif in sorted(out_dir.rglob("*.cif")):
        print(f"[{backend}] structure: {cif}")
    conf_files = sorted(out_dir.rglob("confidence_*.json")) + sorted(
        out_dir.rglob("*_confidences_aggregated.json")
    )
    keys = ("confidence_score", "ptm", "iptm", "avg_plddt", "sample_ranking_score")
    for cf in conf_files:
        data = json.loads(cf.read_text())
        summary = {k: data[k] for k in keys if k in data}
        print(f"[{backend}] {cf.name}: {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-id", help="UniProt accession or PDB ID to fetch")
    target.add_argument("--target-seq", help="Raw target protein sequence (skips fetching)")
    parser.add_argument("--target-chain", help="For multi-chain PDB targets, pick one (default: longest)")
    parser.add_argument(
        "--partner-modality", required=True, choices=list(MODALITY_TO_OF3),
        help="What to cofold the target against",
    )
    parser.add_argument(
        "--partner-value", required=True,
        help="SMILES (small_molecule) or sequence (protein/antibody/peptide/rna/dna)",
    )
    parser.add_argument("--name", default="design", help="Job name, used for output paths")
    parser.add_argument("--backend", choices=["boltz2", "openfold3", "both"], default="both")
    parser.add_argument("--out-dir", default=None, help="Output root (default: data/designs/<name>)")
    parser.add_argument(
        "--no-msa", action="store_true",
        help="Single-sequence mode: no MSA search. Matches BoltzGen's own refold "
             "stage (paper sec 3.5) and is much faster, at some accuracy cost on "
             "natural sequences. Use when scores must be comparable to a campaign's "
             "own metrics, or when the partner is a de novo design with no family.",
    )
    args = parser.parse_args()

    if args.partner_modality in NO_SPECIAL_HANDLING:
        print(
            f"[run_design] NOTE: '{args.partner_modality}' has no CDR-specific or "
            f"other specialized handling in either backend yet — modeled as a "
            f"plain protein chain (see DESIGN.md §5).",
            file=sys.stderr,
        )

    target_seq = resolve_target_sequence(args)
    out_root = Path(args.out_dir) if args.out_dir else REPO_ROOT / "data" / "designs" / args.name
    out_root.mkdir(parents=True, exist_ok=True)

    backends = ["boltz2", "openfold3"] if args.backend == "both" else [args.backend]
    any_pass = False
    for backend in backends:
        print(f"\n{'=' * 60}\n[{backend}] running design '{args.name}'\n{'=' * 60}")
        if backend == "boltz2":
            config_path = build_boltz_config(
                target_seq, args.partner_modality, args.partner_value,
                out_root / f"{args.name}_boltz2.yaml", use_msa=not args.no_msa,
            )
            out_dir = out_root / "boltz2"
            passed = run_boltz2(config_path, out_dir, use_msa=not args.no_msa)
        else:
            config_path = build_of3_config(
                target_seq, args.partner_modality, args.partner_value, args.name,
                out_root / f"{args.name}_of3.json", use_msa=not args.no_msa,
            )
            out_dir = out_root / "openfold3"
            passed = run_openfold3(config_path, out_dir, use_msa=not args.no_msa)

        print(f"[{backend}] {'PASS' if passed else 'FAIL'}")
        if passed:
            report_confidence(out_dir, backend)
            any_pass = True

    sys.exit(0 if any_pass else 1)


if __name__ == "__main__":
    main()
