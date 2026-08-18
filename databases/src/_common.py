"""Shared helpers for databases/src/<source>.py scripts — the code that (re)creates
databases/<source>/ data on any machine. All the actual fetch code lives here in
databases/src/ (not scattered inside each source's own data folder) specifically so the
whole database can be reproduced elsewhere: clone the repo, run these scripts, done —
nothing to hunt for across gitignored data directories.

Mirrors scripts/_common.py's conventions (REPO_ROOT, venv_bin) but for the data-fetch
layer: a resumable streaming HTTP downloader, and a bootstrap for the one new
dependency this layer needs (gdown, for the two Google-Drive-gated sources).
"""

import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

# Deliberately not importing scripts/_common.py here: both files are named _common.py
# and inserting scripts/ onto sys.path while this module is itself mid-import under the
# same name causes Python to reuse this (partially-initialized) module instead of
# loading the other one — a real circular-import bug hit while building this. REPO_ROOT
# and venv_bin() are trivial enough to duplicate rather than fight that.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATABASES_DIR = REPO_ROOT / "databases"
VENVS_DIR = REPO_ROOT / ".venvs"
CHUNK = 1024 * 1024  # 1MB


def venv_bin(name: str, tool: str) -> str:
    return str(VENVS_DIR / name / "bin" / tool)


def download(url: str, out_path: Path, resume: bool = False) -> None:
    """Stream url to out_path. If resume and a partial file exists, continues it via
    a Range request (only meaningful for servers that advertise Accept-Ranges)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    mode = "wb"
    existing = 0
    if resume and out_path.exists():
        existing = out_path.stat().st_size
        if existing:
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = resp.length
        if total is not None and mode == "ab":
            total += existing
        written = existing
        with open(out_path, mode) as f:
            while chunk := resp.read(CHUNK):
                f.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r[download] {out_path.name}: {written / 1e6:.1f} / {total / 1e6:.1f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)


def ensure_gdown() -> str:
    """Creates .venvs/data-fetch/ and installs gdown into it if not already present.
    gdown is the only practical way to bulk-fetch a public Google Drive folder
    (verified during planning: no direct-URL alternative exists for Drive folder
    shares, unlike Zenodo/Dataverse-style single-file hosts)."""
    gdown = venv_bin("data-fetch", "gdown")
    if not Path(gdown).exists():
        venv_dir = VENVS_DIR / "data-fetch"
        print(f"[databases] setting up {venv_dir} (one-time)", file=sys.stderr)
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        subprocess.run([venv_bin("data-fetch", "pip"), "install", "-q", "gdown"], check=True)
    return gdown


def gdown_folder(url: str, out_dir: Path) -> Path:
    gdown = ensure_gdown()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[databases] fetching Google Drive folder -> {out_dir}", file=sys.stderr)
    subprocess.run([gdown, "--folder", url, "-O", str(out_dir)], check=True)
    return out_dir


MMSEQS2_URL = "https://mmseqs.com/latest/mmseqs-linux-arm64.tar.gz"


def ensure_mmseqs2() -> str:
    """Downloads the static aarch64 MMseqs2 binary into .venvs/mmseqs2/ if not already
    present (verified working on this machine during planning — no sudo/apt needed,
    unlike the system package). Used for sequence-identity clustering when building
    train/test splits (databases/src/build_splits.py)."""
    mmseqs = venv_bin("mmseqs2", "mmseqs")
    if not Path(mmseqs).exists():
        venv_dir = VENVS_DIR / "mmseqs2"
        print(f"[databases] setting up {venv_dir} (one-time)", file=sys.stderr)
        tmp_tgz = venv_dir / ".mmseqs.tar.gz"
        download(MMSEQS2_URL, tmp_tgz)
        with tarfile.open(tmp_tgz, mode="r:gz") as tar:
            tar.extractall(venv_dir, filter="data")
        tmp_tgz.unlink()
        # archive extracts to <venv_dir>/mmseqs/bin/mmseqs; venv_bin() expects
        # <venv_dir>/bin/<tool>, so flatten it to match every other tool's layout.
        extracted_bin = venv_dir / "mmseqs" / "bin"
        if extracted_bin.exists() and not (venv_dir / "bin").exists():
            extracted_bin.rename(venv_dir / "bin")
    return mmseqs
