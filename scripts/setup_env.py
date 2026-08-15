#!/usr/bin/env python3
"""Set up one isolated venv per cofolding backend on this GB10 (aarch64) machine.

Verified 2026-07-12: a shared venv is not viable — Protenix pins
torch==2.3.1 exactly, which has no aarch64 CUDA wheel at all (its nvidia-*
deps are gated to platform_machine == "x86_64" in the wheel metadata), so
installing all three backends together silently downgrades torch and kills
CUDA for everyone in that env. Each backend gets its own venv instead.

Each backend's actual source lives in src/<name> (a shallow git clone) and
is installed editable (`pip install -e`), not pulled opaquely from PyPI —
so the running code is on disk and inspectable/modifiable. Verified
2026-07-18: switching an already-pip-installed backend to editable can
leave a stale non-editable package directory in site-packages that silently
shadows the editable one as a namespace package (import succeeds but
resolves to the old location, or submodules 404). If that happens, remove
`.venvs/<name>/lib/python3.*/site-packages/<pkg>/` and reinstall editable.

Model weights live permanently under weights/<name>/ rather than each
tool's own default cache location (~/.cache, ~/.openfold3, ~/checkpoint,
etc.) — see WEIGHTS_DIR usage below and in smoke_test.py/run_design.py.

Usage:
    python3 scripts/setup_env.py                 # set up all backends
    python3 scripts/setup_env.py --only boltz2 openfold3
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENVS_DIR = REPO_ROOT / ".venvs"
SRC_DIR = REPO_ROOT / "src"
WEIGHTS_DIR = REPO_ROOT / "weights"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"

REPOS = {
    "boltz2": "https://github.com/jwohlwend/boltz",
    "openfold3": "https://github.com/aqlaboratory/openfold-3",
    "protenix": "https://github.com/bytedance/Protenix",
    "boltzgen": "https://github.com/HannesStark/boltzgen",
}

# Protenix's non-torch runtime deps, installed explicitly with --no-deps so the
# resolver never sees (and can't act on) Protenix's exact torch==2.3.1 pin.
PROTENIX_DEPS = [
    "scipy>=1.9.0", "ml_collections==1.1.0", "tqdm==4.67.1", "pandas==2.3.1",
    "PyYAML==6.0.2", "matplotlib==3.10.5", "ipywidgets==8.1.7", "py3Dmol==2.5.2",
    "rdkit==2025.9.3", "biopython==1.85", "biotite==1.4.0", "modelcif==1.4",
    "gemmi==0.6.7", "pdbeccdutils==1.0.0", "fair-esm==2.0.0",
    "scikit-learn==1.7.1", "scikit-learn-extra==0.3.0", "deepspeed==0.17.5",
    "pydantic>=2.0.0", "optree==0.17.0", "protobuf==6.31.1", "icecream==2.1.7",
    "ipdb==0.13.13", "wandb==0.21.1", "numpy==2.4.1", "networkx>=3.4.2",
]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def venv_python(name: str) -> Path:
    return VENVS_DIR / name / "bin" / "python3"


def ensure_venv(name: str) -> None:
    venv_dir = VENVS_DIR / name
    if venv_python(name).exists():
        print(f"[{name}] venv already exists, reusing")
        return
    print(f"[{name}] creating venv")
    run([sys.executable, "-m", "venv", str(venv_dir)])
    run([str(venv_python(name)), "-m", "pip", "install", "-q", "--upgrade", "pip", "uv"])


def uv_pip(name: str, *args: str) -> None:
    run([str(venv_python(name)), "-m", "uv", "pip", "install", "-q", *args])


def install_torch(name: str) -> None:
    print(f"[{name}] installing CUDA torch (cu130, aarch64)")
    uv_pip(name, "torch", "--index-url", TORCH_INDEX)


def ensure_source(name: str) -> Path:
    src_path = SRC_DIR / name
    if src_path.exists():
        print(f"[{name}] source already cloned at {src_path}, reusing")
        return src_path
    print(f"[{name}] cloning source from {REPOS[name]}")
    run(["git", "clone", "--depth", "1", REPOS[name], str(src_path)])
    return src_path


def clear_stale_site_packages(name: str, pkg: str) -> None:
    """See module docstring — a leftover non-editable install directory
    will shadow a fresh editable one as a namespace package."""
    site_pkgs = next((VENVS_DIR / name / "lib").glob("python3.*")) / "site-packages"
    stale = site_pkgs / pkg
    if stale.exists() and not stale.is_symlink():
        print(f"[{name}] removing stale {stale} before editable install")
        run(["rm", "-rf", str(stale)])


def setup_boltz2() -> None:
    ensure_venv("boltz2")
    install_torch("boltz2")
    src_path = ensure_source("boltz2")
    clear_stale_site_packages("boltz2", "boltz")
    print("[boltz2] installing boltz (editable from src/) + cuequivariance-torch")
    uv_pip("boltz2", "-e", str(src_path))
    # cuequivariance-torch is pure-python (py3-none-any) and installs fine on
    # aarch64, but its compiled backend cuequivariance-ops-torch has NO
    # aarch64 distribution — the fused triangle-mult kernel is unusable here.
    # We still install cuequivariance-torch since import succeeds; the
    # unusable kernel path is avoided at runtime via --no_kernels (see
    # scripts/smoke_test.py).
    uv_pip("boltz2", "cuequivariance-torch")
    (WEIGHTS_DIR / "boltz2").mkdir(parents=True, exist_ok=True)


def setup_openfold3() -> None:
    ensure_venv("openfold3")
    install_torch("openfold3")
    src_path = ensure_source("openfold3")
    clear_stale_site_packages("openfold3", "openfold3")
    print("[openfold3] installing openfold3 (editable from src/)")
    uv_pip("openfold3", "-e", str(src_path))
    (WEIGHTS_DIR / "openfold3").mkdir(parents=True, exist_ok=True)


def setup_protenix() -> None:
    ensure_venv("protenix")
    install_torch("protenix")
    src_path = ensure_source("protenix")
    clear_stale_site_packages("protenix", "protenix")
    print("[protenix] installing protenix --no-deps, editable from src/ (dodges its exact torch==2.3.1 pin)")
    uv_pip("protenix", "-e", str(src_path), "--no-deps")
    print("[protenix] installing remaining non-torch dependencies")
    uv_pip("protenix", *PROTENIX_DEPS)
    # Protenix has no --checkpoint-dir flag, only $PROTENIX_ROOT_DIR (which
    # also affects other data dirs, e.g. mmcif) — resolved as
    # $PROTENIX_ROOT_DIR/checkpoint. Point just the checkpoint subpath at our
    # permanent weights/ folder via a symlink, rather than repointing
    # PROTENIX_ROOT_DIR broadly and scattering unrelated data dirs there too.
    weights_dir = WEIGHTS_DIR / "protenix"
    weights_dir.mkdir(parents=True, exist_ok=True)
    root_dir = WEIGHTS_DIR / "protenix_root"
    root_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_link = root_dir / "checkpoint"
    if not checkpoint_link.exists():
        checkpoint_link.symlink_to(Path("..") / "protenix")
    print(
        "[protenix] NOTE: inference is currently BLOCKED on this machine — "
        "GB10 reports CUDA compute capability sm_121, but this torch build's "
        "get_arch_list() tops out at sm_120. At least one op in Protenix's "
        "forward pass has no compatible kernel. Install succeeds; "
        "smoke_test.py will report this as a known failure. See DESIGN.md §3."
    )


def setup_boltzgen() -> None:
    ensure_venv("boltzgen")
    install_torch("boltzgen")
    src_path = ensure_source("boltzgen")
    clear_stale_site_packages("boltzgen", "boltzgen")
    print("[boltzgen] installing boltzgen (editable from src/)")
    uv_pip("boltzgen", "-e", str(src_path))
    (WEIGHTS_DIR / "boltzgen").mkdir(parents=True, exist_ok=True)
    print(
        "[boltzgen] weights auto-download (~6GB) to weights/boltzgen on "
        "first `boltzgen run ... --cache weights/boltzgen` — pass that flag "
        "explicitly (see scripts/run_binder_design.py / SKILL.md §B), it's "
        "not picked up automatically like the other backends' --cache flags."
    )


BACKENDS = {
    "boltz2": setup_boltz2,
    "openfold3": setup_openfold3,
    "protenix": setup_protenix,
    "boltzgen": setup_boltzgen,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only", nargs="+", choices=list(BACKENDS), default=list(BACKENDS),
        help="Only set up these backends (default: all)",
    )
    args = parser.parse_args()

    VENVS_DIR.mkdir(exist_ok=True)
    SRC_DIR.mkdir(exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    for name in args.only:
        BACKENDS[name]()
        print(f"[{name}] done\n")

    print("Environment setup complete. Run scripts/smoke_test.py next.")


if __name__ == "__main__":
    main()
