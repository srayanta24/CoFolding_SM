#!/usr/bin/env python3
"""Does packstat actually need FastRelax? (milestone 2 feasibility test)

packstat is the only metric in this project that has beaten ipTM on a valid test
(23/28, p=0.0005, clears its null -- results/paired_comparison.md). It is also
unusable in a real funnel: Rosetta scoring costs 22 min/complex on the paired set and
~90 min on the larger ASD ones, against BoltzGen's protocol of 60,000 designs.

But the cost is not packstat. Measured directly on one complex with an idle box:

    FastRelax            571 s
    InterfaceAnalyzer      2.4 s   <- computes packstat, sc, dG, dSASA

**FastRelax is 99.6% of the cost.** So the question is whether packstat needs the
relaxed structure at all. There is already a hint it might not: on pdb_000013bf the
*energy* terms were destroyed without relax (dG_separated +616.57 vs -30.47, H-bonds 0
vs 12) but shape complementarity barely moved (0.513 vs 0.530). `sc` and `packstat` are
**geometric** measures of surface fit, not energies, so they should be far less
sensitive to the sub-angstrom placement that wrecks the Rosetta score function.

If packstat is robust, we get our one working metric at ~2.4 s instead of 22-90 min --
a 500-2000x speedup with no model, no training and no approximation error.

Method, and why it is cheap
---------------------------
The relaxed packstat for 379 paired complexes is **already in results/rosetta.tsv**, a
by-product of work already done. This script computes only the *unrelaxed* value and
compares. That is ~2.4 s per complex rather than re-running FastRelax, which an earlier
attempt did pointlessly and which was then starved by the 16 ASD Rosetta workers
already saturating the box.

For the same reason this defaults to 2 workers: the ASD Rosetta run owns most cores and
must not be starved in turn.

What counts as a pass
---------------------
Not "the absolute values match" -- an offset is harmless, since every use of packstat in
this project is a *ranking* (the paired sign test compares two mutants of one parent).
What matters is:

  1. high rank correlation between raw and relaxed packstat, and
  2. the paired sign test reproducing on raw values (relaxed gives 23/28).

(2) is the real criterion; (1) is diagnostic.

Usage:
    .venvs/germinal/bin/python3 experiments/scoring_calibration/packstat_relax_test.py
"""

import argparse
import csv
import glob
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCORES_DIR = HERE / ".scores"
RES = HERE / "results"
OUT = RES / "packstat_raw.tsv"
INTERFACE = "A_BC"
FIELDS = ["entry_id", "packstat_raw", "sc_raw", "dG_raw", "dSASA_raw", "seconds", "status"]


def _init():
    import pyrosetta
    pyrosetta.init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false",
                   silent=True)


def _one(entry_id: str) -> dict:
    out = {"entry_id": entry_id, "packstat_raw": "", "sc_raw": "", "dG_raw": "",
           "dSASA_raw": "", "seconds": "", "status": "error"}
    t = time.time()
    try:
        from pyrosetta import pose_from_file
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
        hits = glob.glob(str(SCORES_DIR / entry_id / "**" / "*_model_0.cif"),
                         recursive=True)
        if not hits:
            out["status"] = "no_structure"
            return out
        pose = pose_from_file(hits[0])
        if pose.num_chains() != 3:
            out["status"] = f"chains_{pose.num_chains()}"
            return out
        ia = InterfaceAnalyzerMover(INTERFACE)
        ia.set_pack_separated(True)
        ia.set_compute_interface_sc(True)
        ia.set_compute_packstat(True)
        ia.apply(pose)                      # NO FastRelax -- that is the whole point
        d = ia.get_all_data()
        out["packstat_raw"] = f"{d.packstat:.5f}"
        out["sc_raw"] = f"{d.sc_value:.5f}"
        out["dG_raw"] = f"{ia.get_separated_interface_energy():.4f}"
        out["dSASA_raw"] = f"{ia.get_interface_delta_sasa():.2f}"
        out["status"] = "ok"
    except Exception as exc:
        out["status"] = f"error:{type(exc).__name__}"
    out["seconds"] = f"{time.time()-t:.1f}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=2,
                    help="keep low: the ASD Rosetta run owns most cores (default 2)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with (RES / "rosetta.tsv").open(encoding="utf-8") as fh:
        relaxed = {r["entry_id"]: r for r in csv.DictReader(fh, delimiter="\t")
                   if r.get("status") == "ok" and r.get("packstat")}
    done = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as fh:
            done = {r["entry_id"] for r in csv.DictReader(fh, delimiter="\t")}
    todo = [e for e in relaxed if e not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[packstat-raw] {len(relaxed)} with relaxed values, {len(done)} done, "
          f"{len(todo)} to compute, {args.workers} workers", flush=True)
    if not todo:
        return

    new_file = not OUT.exists()
    t0 = time.time()
    with OUT.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t",
                           extrasaction="ignore", lineterminator="\n")
        if new_file:
            w.writeheader()
            fh.flush()
        with Pool(args.workers, initializer=_init) as pool:
            for i, res in enumerate(pool.imap_unordered(_one, todo), 1):
                w.writerow(res)
                fh.flush()
                if i % 25 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  [{i}/{len(todo)}] {el/i:.1f}s/complex, "
                          f"ETA {(len(todo)-i)*el/i/60:.0f}m", flush=True)
    print(f"[packstat-raw] done in {(time.time()-t0)/60:.1f}m -> {OUT}")


if __name__ == "__main__":
    main()
