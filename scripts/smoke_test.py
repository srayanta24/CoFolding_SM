#!/usr/bin/env python3
"""Run a toy protein-ligand prediction through each cofolding backend.

Verified working commands as of 2026-07-12 (see DESIGN.md §3 for full
findings). Requires scripts/setup_env.py to have been run first.

Usage:
    python3 scripts/smoke_test.py                 # run all three backends
    python3 scripts/smoke_test.py --only boltz2 openfold3
"""

import argparse
import subprocess
import sys

import os

from _common import PROTENIX_ROOT_DIR, REPO_ROOT, WEIGHTS_DIR, produced_structure, venv_bin

CONFIGS_DIR = REPO_ROOT / "configs" / "examples"
OUT_DIR = REPO_ROOT / "data" / "designs" / "smoke_test"

TIMEOUT_S = 40 * 60  # weight downloads + MSA server queueing can take a while


def run_boltz2() -> bool:
    """Known-working flags: --num_workers 0 (avoids a DataLoader/CUDA fork
    deadlock) and --no_kernels (cuequivariance-ops-torch has no aarch64
    build, so the fused triangle-mult kernel is unusable — falls back to
    pure PyTorch)."""
    out_dir = OUT_DIR / "boltz2"
    cmd = [
        venv_bin("boltz2", "boltz"), "predict",
        str(CONFIGS_DIR / "toy_protein_ligand.yaml"),
        "--use_msa_server",
        "--num_workers", "0",
        "--no_kernels",
        "--out_dir", str(out_dir),
        "--cache", str(WEIGHTS_DIR / "boltz2"),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, timeout=TIMEOUT_S)
    return produced_structure(out_dir)


def run_openfold3() -> bool:
    """Ran clean with no workarounds needed. The only wrinkle is a one-time
    interactive y/n prompt on first weight download, which we auto-confirm."""
    out_dir = OUT_DIR / "openfold3"
    ckpt = WEIGHTS_DIR / "openfold3" / "of3-p2-155k.pt"
    cmd = [
        venv_bin("openfold3", "run_openfold"), "predict",
        "--query-json", str(CONFIGS_DIR / "toy_protein_ligand_of3.json"),
        "--use-msa-server", "true",
        "--output-dir", str(out_dir),
    ]
    if ckpt.exists():
        cmd += ["--inference-ckpt-path", str(ckpt)]
    subprocess.run(
        cmd, cwd=REPO_ROOT, timeout=TIMEOUT_S,
        input="yes\n" * 10,  # confirms any weight-download prompts, if ckpt is missing
        text=True,
    )
    return produced_structure(out_dir)


def run_protenix() -> bool:
    """KNOWN FAILING as of 2026-07-12: GB10 reports CUDA compute capability
    sm_121, but this torch build's get_arch_list() tops out at sm_120 — at
    least one op in Protenix's forward pass has no compatible kernel
    ("CUDA error: no kernel image is available for execution on the
    device"). trimul/triatt kernels are forced to 'torch' to dodge the same
    missing cuequivariance-ops-torch gap Boltz-2 has, and the default
    (protenix) MSA server mode is used since --msa_server_mode colabfold has
    an unrelated file-naming bug in its result parser. Kept in the smoke
    test so this is automatically re-checked if a future torch/Protenix
    release fixes the underlying gap.

    IMPORTANT: Protenix's CLI exits 0 even when the prediction fails
    internally (verified — it catches and logs per-target errors rather
    than propagating them), so success here is judged purely by whether a
    structure file was actually written, not by the exit code."""
    out_dir = OUT_DIR / "protenix"
    cmd = [
        venv_bin("protenix", "protenix"), "pred",
        "-i", str(CONFIGS_DIR / "toy_protein_ligand_protenix.json"),
        "-o", str(out_dir),
        "--sample", "1",
        "--step", "20",
        "--trimul_kernel", "torch",
        "--triatt_kernel", "torch",
    ]
    # $PROTENIX_ROOT_DIR/checkpoint resolves to weights/protenix via a
    # symlink (see setup_env.py) — Protenix has no direct --checkpoint-dir
    # flag, only this env var (configs/configs_inference.py).
    env = {**os.environ, "PROTENIX_ROOT_DIR": str(PROTENIX_ROOT_DIR)}
    subprocess.run(cmd, cwd=REPO_ROOT, timeout=TIMEOUT_S, env=env)
    return produced_structure(out_dir)


BACKENDS = {
    "boltz2": run_boltz2,
    "openfold3": run_openfold3,
    "protenix": run_protenix,
}
KNOWN_FAILING = {"protenix"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="+", choices=list(BACKENDS), default=list(BACKENDS),
        help="Only test these backends (default: all)",
    )
    args = parser.parse_args()

    results: dict[str, bool] = {}
    for name in args.only:
        print(f"\n{'=' * 60}\n[{name}] running toy protein-ligand prediction\n{'=' * 60}")
        try:
            results[name] = BACKENDS[name]()
        except subprocess.TimeoutExpired:
            print(f"[{name}] TIMED OUT after {TIMEOUT_S}s")
            results[name] = False
        except FileNotFoundError as e:
            print(f"[{name}] venv not found — run scripts/setup_env.py first ({e})")
            results[name] = False

    print(f"\n{'=' * 60}\nSummary\n{'=' * 60}")
    all_expected = True
    for name, passed in results.items():
        expected_fail = name in KNOWN_FAILING
        if passed:
            status = "PASS" + (" (unexpected — a known issue may be fixed now!)" if expected_fail else "")
        else:
            status = "FAIL (known issue, see DESIGN.md §3)" if expected_fail else "FAIL (unexpected)"
            if not expected_fail:
                all_expected = False
        print(f"  {name:12s} {status}")

    sys.exit(0 if all_expected else 1)


if __name__ == "__main__":
    main()
