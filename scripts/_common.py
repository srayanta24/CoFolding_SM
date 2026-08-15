"""Shared helpers for co_folding scripts."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENVS_DIR = REPO_ROOT / ".venvs"
SRC_DIR = REPO_ROOT / "src"
WEIGHTS_DIR = REPO_ROOT / "weights"

# Each backend is installed editable from src/<name> (not PyPI) so the
# actual running code is inspectable/modifiable on disk. Verified
# 2026-07-18: a stray non-editable install directory left in a venv's
# site-packages will silently shadow the editable one as a namespace
# package — if a backend's import suddenly resolves to a None __file__ or
# an unexpected ModuleNotFoundError after switching to editable, check for
# and remove a leftover `site-packages/<pkg>/` directory first.
PROTENIX_ROOT_DIR = WEIGHTS_DIR / "protenix_root"  # $PROTENIX_ROOT_DIR/checkpoint -> weights/protenix, see setup_env.py


def venv_bin(name: str, tool: str) -> str:
    return str(VENVS_DIR / name / "bin" / tool)


def produced_structure(out_dir: Path) -> bool:
    """Ground truth for "did this actually work": at least one structure
    file on disk. Exit codes alone are NOT reliable — Protenix's CLI
    catches per-target failures internally and exits 0 even when the
    prediction never completed (verified 2026-07-12: a run that hit "CUDA
    error: no kernel image is available" still exited 0)."""
    return out_dir.exists() and any(out_dir.rglob("*.cif"))
