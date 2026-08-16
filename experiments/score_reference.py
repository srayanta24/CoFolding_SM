#!/usr/bin/env python3
"""Score a curated reference antibody-antigen complex through the same cofolding
pipeline (scripts/run_design.py) used to externally validate generated designs.

This is a calibration baseline, not a structural/epitope check: it tells you what a
genuine true positive scores on this pipeline, so generated-design metrics can be read
against a real number instead of only generic literature thresholds. It cannot detect
whether a generated design binds the same surface as the reference complex — see
experiments/README.md.

scripts/run_design.py's main() calls sys.exit(), so it's invoked as a subprocess here,
not imported and called directly.

Usage:
    python3 experiments/score_reference.py --name hel
    python3 experiments/score_reference.py --name hel --backend boltz2
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _common import REPO_ROOT  # noqa: E402

import fetch_reference  # noqa: E402

TIMEOUT_S = 40 * 60


def score_reference(name: str, backend: str = "both") -> Path:
    seqs = fetch_reference.fetch_reference(name)
    out_dir = REPO_ROOT / "data" / "designs" / f"{name}_reference"

    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_design.py"),
        "--target-seq", seqs["antigen_seq"],
        "--partner-modality", "antibody",
        "--partner-value", f"{seqs['heavy_seq']},{seqs['light_seq']}",
        "--name", f"{name}_reference",
        "--backend", backend,
    ]
    print(f"[score_reference] scoring '{name}' via: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=REPO_ROOT, timeout=TIMEOUT_S)
    return out_dir


def load_reference_confidence(out_dir: Path) -> dict:
    """Mirrors run_design.report_confidence()'s file-discovery logic and key set, so
    the baseline numbers in a report are directly comparable to what that script
    prints for any other cofolding run."""
    keys = ("confidence_score", "ptm", "iptm", "avg_plddt", "sample_ranking_score")
    result = {}
    for backend_dir in ("boltz2", "openfold3"):
        backend_out = out_dir / backend_dir
        if not backend_out.exists():
            continue
        conf_files = sorted(backend_out.rglob("confidence_*.json")) + sorted(
            backend_out.rglob("*_confidences_aggregated.json")
        )
        summaries = []
        for cf in conf_files:
            data = json.loads(cf.read_text())
            summaries.append({k: data[k] for k in keys if k in data})
        if summaries:
            result[backend_dir] = summaries
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Reference complex name from reference_targets.py")
    parser.add_argument("--backend", choices=["boltz2", "openfold3", "both"], default="both")
    args = parser.parse_args()

    out_dir = score_reference(args.name, backend=args.backend)
    confidence = load_reference_confidence(out_dir)
    print(json.dumps(confidence, indent=2))


if __name__ == "__main__":
    main()
