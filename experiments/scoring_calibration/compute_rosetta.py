#!/usr/bin/env python3
"""Rosetta interface energetics over the calibration set (EXPERIMENT_02, tier 1).

The first metric in this experiment that is **not** a function of the PAE matrix.
Tier 0 (ipSAE) failed precisely because it reweights the same matrix ipTM comes from,
and so inherits its blindness to affinity; Rosetta scores physics on the coordinates
instead, which is an independent source of information.

The target metric is **dG_separated / dSASA** — interface energy normalised by buried
surface area — which is the best *physics* feature in the 3,766-binder meta-analysis
(Overath et al. 2025), where `ipSAE_min x interface dG/dSASA` is the best interaction
feature overall. Shape complementarity is also collected, since the same paper's
recommended pre-filter uses `sc > 0.62`.

FastRelax is mandatory, not optional
-------------------------------------
Measured directly on `clean__pdb_000013bf` before writing this:

    raw Boltz structure   dG_separated = +616.57   hbonds = 0    score =   -93.2
    after FastRelax       dG_separated =  -30.47   hbonds = 12   score = -1464.6

A positive 600-unit "interface energy" is not a weak interface, it is a meaningless
number: the Rosetta scorefunction is steeply repulsive and sensitive to sub-angstrom
atom placement, so a structure produced by anything other than Rosetta sits on a
clash-dominated part of the energy landscape. Scoring unrelaxed would have produced a
full column of plausible-looking garbage. `constrain_relax_to_start_coords(True)`
keeps the relaxed structure near the prediction rather than letting it drift into a
different pose.

Cost
----
~9.5 min per 487-residue complex single-threaded, so ~60+ CPU-hours for all 332.
Parallelised across worker processes (20 cores on this box) that is ~4h wall-clock.
Each worker initialises PyRosetta once; the parent does all the writing, so there is
no lock contention and no partially-written row.

Runs under `.venvs/germinal` (Python 3.10) — the only venv with PyRosetta, installed
during the paused Germinal work. That work's one completed milestone turns out to be
load-bearing here rather than there.

Usage:
    .venvs/germinal/bin/python3 experiments/scoring_calibration/compute_rosetta.py --limit 2
    .venvs/germinal/bin/python3 experiments/scoring_calibration/compute_rosetta.py --workers 16
"""

import argparse
import csv
import glob
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCORES_DIR = HERE / ".scores"
RESULTS = HERE / "results" / "scores.tsv"
OUT = HERE / "results" / "rosetta.tsv"

CARRY = ["entry_id", "set", "label", "kd_molar", "ddg_kcal", "elisa_ratio", "iptm",
         "target_pdb", "neg_log_kd"]
METRICS = ["dG_separated", "dSASA_int", "dG_dSASA_x100", "sc_value",
           "interface_hbonds", "delta_unsat_hbonds", "packstat",
           "total_score_relaxed", "total_score_raw", "nres"]
FIELDS = CARRY + METRICS + ["status", "seconds"]

# Interface definition: antigen (A) against the antibody (B heavy + C light).
# build_boltz_config always emits the target as chain A, then partner chains B and C;
# verified against the output CIF (A=51, B=221, C=215 for pdb_000013bf, matching the
# submitted config exactly). The paratope spans both antibody chains, so grouping them
# is what makes this comparable with design_to_target_iptm.
INTERFACE = "A_BC"


def _init_worker() -> None:
    import pyrosetta
    pyrosetta.init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false",
                   silent=True)


def _score_one(args: tuple) -> dict:
    entry_id, carry = args
    out = dict(carry)
    out.update({k: "" for k in METRICS})
    t0 = time.time()
    try:
        import pyrosetta
        from pyrosetta import pose_from_file, get_fa_scorefxn
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
        from pyrosetta.rosetta.protocols.relax import FastRelax

        hits = glob.glob(str(SCORES_DIR / entry_id / "**" / "*_model_0.cif"),
                         recursive=True)
        if not hits:
            out["status"] = "no_structure"
            out["seconds"] = f"{time.time()-t0:.1f}"
            return out

        pose = pose_from_file(hits[0])
        if pose.num_chains() != 3:
            out["status"] = f"unexpected_chains_{pose.num_chains()}"
            out["seconds"] = f"{time.time()-t0:.1f}"
            return out

        sf = get_fa_scorefxn()
        out["nres"] = pose.total_residue()
        out["total_score_raw"] = f"{sf(pose):.3f}"

        fr = FastRelax()
        fr.set_scorefxn(sf)
        fr.constrain_relax_to_start_coords(True)
        fr.apply(pose)
        out["total_score_relaxed"] = f"{sf(pose):.3f}"

        ia = InterfaceAnalyzerMover(INTERFACE)
        ia.set_pack_separated(True)
        ia.set_compute_interface_sc(True)
        ia.set_compute_packstat(True)
        ia.apply(pose)
        data = ia.get_all_data()

        dg = ia.get_separated_interface_energy()
        dsasa = ia.get_interface_delta_sasa()
        out["dG_separated"] = f"{dg:.4f}"
        out["dSASA_int"] = f"{dsasa:.2f}"
        # x100 matches Rosetta's own dG_separated/dSASAx100 convention, so the numbers
        # are directly comparable to values quoted in the literature.
        out["dG_dSASA_x100"] = f"{dg / dsasa * 100:.5f}" if dsasa > 1e-6 else ""
        out["sc_value"] = f"{data.sc_value:.4f}"
        out["interface_hbonds"] = data.interface_hbonds
        out["delta_unsat_hbonds"] = data.delta_unsat_hbonds
        out["packstat"] = f"{data.packstat:.4f}"
        out["status"] = "ok"
    except Exception:
        out["status"] = "error"
        (SCORES_DIR / entry_id).mkdir(parents=True, exist_ok=True)
        (SCORES_DIR / entry_id / "rosetta_error.txt").write_text(traceback.format_exc())
    out["seconds"] = f"{time.time()-t0:.1f}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4),
                    help="worker processes (default: cores - 4, leaving headroom)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sets", default=None, help="comma-separated set names")
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    results_path, out_path = Path(args.results), Path(args.out)
    scored = list(csv.DictReader(results_path.open(encoding="utf-8"), delimiter="\t"))
    if args.sets:
        want = {s.strip() for s in args.sets.split(",")}
        scored = [r for r in scored if r["set"] in want]

    done = set()
    if out_path.exists():
        done = {r["entry_id"] for r in
                csv.DictReader(out_path.open(encoding="utf-8"), delimiter="\t")
                if r.get("entry_id")}
    todo = [r for r in scored if r["entry_id"] not in done and r["status"] == "ok"]
    if args.limit:
        todo = todo[: args.limit]

    print(f"[rosetta] {len(scored)} entries, {len(done)} done, {len(todo)} to score, "
          f"{args.workers} workers", flush=True)
    if not todo:
        return

    jobs = [(r["entry_id"], {k: r.get(k, "") for k in CARRY}) for r in todo]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()
    # Same schema guard as score_sets.py. Extending the carried label columns while
    # an output file already existed silently shifted every later column; the rows
    # stayed parseable as text but `status` read blank, so downstream steps reported
    # "nothing to do" and exited successfully. Fail loudly instead.
    if not new_file:
        with out_path.open(encoding="utf-8") as _fh:
            _existing = _fh.readline().rstrip("\n").split("\t")
        if _existing != FIELDS:
            raise SystemExit(
                f"[{__file__}] REFUSING TO APPEND: {out_path} has "
                f"{len(_existing)} columns, this build writes {len(FIELDS)}. "
                "Rewrite the file with the current schema or use a new --out.")
    t_start = time.time()

    with out_path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t",
                           extrasaction="ignore", lineterminator="\n")
        if new_file:
            w.writeheader()
            fh.flush()
        with Pool(args.workers, initializer=_init_worker) as pool:
            for i, res in enumerate(pool.imap_unordered(_score_one, jobs), 1):
                w.writerow(res)
                fh.flush()
                el = time.time() - t_start
                print(f"[{i}/{len(jobs)}] {res['entry_id']:<42} {res['status']:<10} "
                      f"dG/dSASA={res.get('dG_dSASA_x100') or '-':>10} "
                      f"sc={res.get('sc_value') or '-':>7} {res['seconds']:>7}s "
                      f"| ETA {(len(jobs)-i)*el/i/3600:.1f}h", flush=True)

    print(f"[rosetta] done in {(time.time()-t_start)/3600:.2f}h -> {out_path}")


if __name__ == "__main__":
    main()
