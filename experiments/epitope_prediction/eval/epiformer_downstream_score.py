#!/usr/bin/env python3
"""Phase 3 (see PLAN.md's EpiFormer write-up): score BoltzGen's already-completed
downstream campaigns (eval/.downstream_runs/<pdb_id>/<variant>/run1/) with EpiFormer's
own pretrained model as a second, independently-trained opinion -- unlike our own
purely-geometric 5A-contact recomputation (downstream_eval.py's
design_contacts_by_label_seq), EpiFormer sees the real antigen+designed-antibody pair
and makes its own learned epitope call.

Sanity-checked first on 5 known real antibody-antigen pairs (not generated designs):
2/5 strong hits (recall 0.79, 1.00), 3/5 misses (recall 0.00) -- consistent with
EpiFormer's own reported F1=0.305 on its harder benchmark split, not a broken pipeline.
Treat its verdict here as one more signal alongside the existing 5A-contact metrics, not
a tie-breaker on its own -- and note the real caveat: EpiFormer was trained on natural
antibody-antigen complexes (AsEP), not BoltzGen-generated ones, so transfer here is
unverified beyond this scoring exercise itself.

Retroactive scoring only -- reads already-completed campaign outputs, launches no new
BoltzGen campaigns.

Usage:
    python3 experiments/epitope_prediction/eval/epiformer_downstream_score.py <pdb_id> [--variant conditioned|baseline|conditioned_D]
    python3 experiments/epitope_prediction/eval/epiformer_downstream_score.py --all
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from interface_labels import Atom, compute_interface_labels_by_label_seq, parse_atom_site_generic  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EPIFORMER_DIR = REPO_ROOT / "src" / "epiformer"
EPIFORMER_PYTHON = REPO_ROOT / ".venvs" / "epiformer" / "bin" / "python"
HMMER_BIN_DIR = Path.home() / ".local" / "hmmer-3.4" / "bin"  # ANARCI's hmmscan dependency, built from source -- see README.md sec 2.7
CHECKPOINT = EPIFORMER_DIR / "checkpoints" / "epiformer-epitope-group" / "epiformer_best.pt"  # harder split, per the paper's own framing
DOWNSTREAM_RUNS_DIR = Path(__file__).resolve().parent / ".downstream_runs"

ALL_TARGETS = [
    "pdb_000010gh", "pdb_00008pmy", "pdb_00008tzu", "pdb_00009cb5",
    "pdb_00009cct", "pdb_00009me5", "pdb_00009me7", "pdb_00009uvi",
]


def write_pdb(atoms: list[Atom], path: Path, use_label_seq: bool) -> None:
    """Minimal fixed-column PDB ATOM writer -- resnum column uses label_seq_id for the
    antigen (to match compute_interface_labels_by_label_seq()'s keying of the ground
    truth) or auth_seq_id for the antibody (numbering there is arbitrary; ANARCI's CDR
    detection works off sequence via its own HMM alignment, not input numbering)."""
    with open(path, "w") as f:
        i = 0
        for a in atoms:
            resnum = a.label_seq_id if use_label_seq else a.auth_seq_id
            if not resnum.isdigit():
                continue
            i += 1
            f.write(
                f"ATOM  {i:5d} {a.atom_name:<4s} {a.auth_comp_id:<3s} {a.auth_asym_id:1s}{resnum:>4s}    "
                f"{a.xyz[0]:8.3f}{a.xyz[1]:8.3f}{a.xyz[2]:8.3f}{1.0:6.2f}{0.0:6.2f}          {a.type_symbol:>2s}\n"
            )
        f.write("END\n")


def split_design_cif(cif_path: Path) -> tuple[list[Atom], list[Atom]]:
    """Antigen (largest chain by resolved residue count) vs antibody (everything else)
    -- same chain-identification logic as downstream_eval.py's
    design_contacts_by_label_seq, reused rather than re-derived (chain letters get
    renamed on conflict by BoltzGen itself, so a fixed-letter assumption isn't safe)."""
    atoms = [a for a in parse_atom_site_generic(cif_path) if a.type_symbol != "H"]
    by_chain: dict[str, list[Atom]] = {}
    for a in atoms:
        by_chain.setdefault(a.label_asym_id, []).append(a)
    antigen_chain = max(by_chain, key=lambda c: len({a.label_seq_id for a in by_chain[c]}))
    antigen_atoms = by_chain[antigen_chain]
    antibody_atoms = [a for chain_id, chain_atoms in by_chain.items() if chain_id != antigen_chain
                       for a in chain_atoms]
    return antibody_atoms, antigen_atoms


def run_epiformer(antigen_pdb: Path, antibody_pdb: Path, out_json: Path) -> dict | None:
    env = dict(os.environ)
    env["PATH"] = f"{HMMER_BIN_DIR}:{env.get('PATH', '')}"
    cmd = [str(EPIFORMER_PYTHON), "inference.py",
           "--antigen_pdb", str(antigen_pdb), "--antibody_pdb", str(antibody_pdb),
           "--checkpoint", str(CHECKPOINT), "--output", str(out_json)]
    result = subprocess.run(cmd, cwd=EPIFORMER_DIR, env=env, capture_output=True, text=True)
    if result.returncode != 0 or not out_json.exists():
        print(f"[epiformer_downstream_score] inference failed:\n{result.stderr[-2000:]}", file=sys.stderr)
        return None
    return json.loads(out_json.read_text())


def score_design(cif_path: Path, true_positive: set[int]) -> dict | None:
    antibody_atoms, antigen_atoms = split_design_cif(cif_path)
    if not antibody_atoms or not antigen_atoms:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        antigen_pdb, antibody_pdb, out_json = tmp_dir / "antigen.pdb", tmp_dir / "antibody.pdb", tmp_dir / "predictions.json"
        write_pdb(antigen_atoms, antigen_pdb, use_label_seq=True)
        write_pdb(antibody_atoms, antibody_pdb, use_label_seq=False)
        result = run_epiformer(antigen_pdb, antibody_pdb, out_json)
    if result is None:
        return None

    pred = {r["residue_id"] for r in result["residue_details"] if r.get("is_epitope")}
    inter = pred & true_positive
    return {
        "n_predicted": len(pred),
        "recall": len(inter) / len(true_positive) if true_positive else float("nan"),
        "precision": len(inter) / len(pred) if pred else float("nan"),
    }


def _mean(values: list[float]) -> float:
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else float("nan")


def score_target(pdb_id: str, variant: str) -> None:
    true_positive = {k for k, v in compute_interface_labels_by_label_seq(pdb_id).items() if v}
    run_dir = DOWNSTREAM_RUNS_DIR / pdb_id / variant / "run1"
    design_dir = run_dir / "final_ranked_designs" / "final_5_designs"
    cifs = sorted(design_dir.glob("rank*.cif")) if design_dir.exists() else []
    if not cifs:
        print(f"[epiformer_downstream_score] {pdb_id}/{variant}: no ranked designs found at {design_dir}")
        return

    print(f"[epiformer_downstream_score] {pdb_id}/{variant}: {len(true_positive)} true epitope residues, "
          f"{len(cifs)} ranked designs")
    recalls, precisions = [], []
    for cif_path in cifs:
        result = score_design(cif_path, true_positive)
        if result is None:
            print(f"    {cif_path.name}: EpiFormer scoring failed, skipped")
            continue
        recalls.append(result["recall"])
        precisions.append(result["precision"])
        print(f"    {cif_path.name}: {result['n_predicted']} EpiFormer-predicted epitope residues, "
              f"recall={result['recall']:.3f} precision={result['precision']:.3f}")

    print(f"[epiformer_downstream_score] {pdb_id}/{variant} summary: "
          f"mean recall={_mean(recalls):.3f}, mean precision={_mean(precisions):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdb_id", nargs="?")
    parser.add_argument("--variant", default="conditioned", help="conditioned | baseline | conditioned_D")
    parser.add_argument("--all", action="store_true", help="Score all 8 targets' baseline + conditioned + conditioned_D (whichever exist)")
    args = parser.parse_args()

    if args.all:
        for pdb_id in ALL_TARGETS:
            for variant in ("baseline", "conditioned", "conditioned_D"):
                if (DOWNSTREAM_RUNS_DIR / pdb_id / variant / "run1").exists():
                    score_target(pdb_id, variant)
        return

    if not args.pdb_id:
        sys.exit("pdb_id required unless --all is passed")
    score_target(args.pdb_id, args.variant)


if __name__ == "__main__":
    main()
