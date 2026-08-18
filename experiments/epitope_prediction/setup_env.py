#!/usr/bin/env python3
"""Set up the epitope-prediction training venv (.venvs/epitope-prediction/).

Verified working on this GB10 (aarch64) machine during planning: torch (cu130,
matching scripts/setup_env.py's index), torch-geometric, fair-esm, and freesasa all
install and import cleanly here — no repeat of the cuequivariance/DeepSpeed/Protenix
aarch64 gaps documented in DESIGN.md §3 for the cofolding backends.

This venv is real training infrastructure (not a one-shot CLI tool like gdown/mmseqs2),
so it gets its own setup script matching scripts/setup_env.py's pattern, rather than
databases/src/_common.py's lazy per-call bootstrap.

Usage:
    python3 experiments/epitope_prediction/setup_env.py
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENV_DIR = REPO_ROOT / ".venvs" / "epitope-prediction"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"


def venv_python() -> Path:
    return VENV_DIR / "bin" / "python3"


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def uv_pip(*args: str) -> None:
    run([str(venv_python()), "-m", "uv", "pip", "install", "-q", *args])


def main() -> None:
    if venv_python().exists():
        print(f"venv already exists at {VENV_DIR}, reusing")
    else:
        print(f"creating venv at {VENV_DIR}")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
        run([str(venv_python()), "-m", "pip", "install", "-q", "--upgrade", "pip", "uv"])

    print("installing CUDA torch (cu130, aarch64)")
    uv_pip("torch", "--index-url", TORCH_INDEX)

    print("installing torch-geometric, fair-esm, freesasa, scikit-learn")
    uv_pip("torch-geometric", "fair-esm", "freesasa", "scikit-learn")

    print("Environment setup complete.")


if __name__ == "__main__":
    main()
