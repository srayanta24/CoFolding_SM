#!/usr/bin/env python3
"""Milestone 7 manifest: packstat on REAL DESIGNS, not point mutants.

PRE-REGISTERED. Everything below the "Criteria" heading was written before any Rosetta
job was launched, because this benchmark has twice taught that a criterion chosen after
seeing the table is not a criterion (shape complementarity's 13/17 -> 16/28, and
EXPERIMENT_04's extremes rescue).

The gap this closes
-------------------
Every result in the scoring-calibration arc rests on either point mutants of real
complexes (the 28-parent paired benchmark) or real antibodies against real antigens
(the ASD benchmark). **Neither is a design.** PLAN.md sec 9 names this as the arc's
central unproven assumption: that a metric separating real binders from real
non-binders also separates good designs from bad ones. Point mutants are a far subtler
perturbation than independently generated designs, so transfer does not follow.

The hypothesis, which EXPERIMENT_04 made falsifiable
----------------------------------------------------
EXPERIMENT_04 concluded packstat is an **interface-integrity detector** -- it tells a
binding-retained point mutant from a binding-destroyed one (23/28, p=0.0005) but does
not rank graded affinity (rho=+0.058, below its null). It explicitly could NOT establish
that interpretation: the extremes test that would have shown it had no power at 6
targets.

This tests it where it should be easiest to see. Design pools are mostly structural
failures -- median filter_rmsd is 8.03 on one target and 17.29 on the other, against a
<=2.5 bar -- so if packstat detects structural integrity, this is the regime where it
should separate cleanly.

The label, and why it is not a binding label
--------------------------------------------
`filter_rmsd`: does BoltzGen's own refold of the design match the backbone that was
designed? It is a **structural self-consistency** measure, computed by the pipeline
independently of anything Rosetta sees. It is NOT evidence that a design binds -- there
are no binding labels for designs anywhere in this project, and sec 9's risk is
untouched by this experiment.

It carries real independent information: it correlates with ipTM at only rho=+0.340
(target 1) and +0.348 (target 2), two unrelated antigens agreeing to within 0.008.

Two targets, because one proves nothing
---------------------------------------
    target 1  1,000 designs, 165 pass (16%), median filter_rmsd  8.03, 419 res
    target 2  3,000 designs, 111 pass  (4%), median filter_rmsd 17.29, 439 res

Each is analysed WITHIN antigen, so antigen identity is controlled by construction (the
confound that invalidated EXPERIMENT_02's cross-complex column). The requirement is the
same direction in BOTH -- the structure of the paired benchmark's independent "new 11"
column, where a real effect should agree in direction even if it misses significance.

Target 2's pool is gitignored along with its campaign, at the user's instruction. Its
derived results stay out of the repo; the write-up states the direction it showed
without its numbers, and says so.

Sampling
--------
All passers, plus `NEG_RATIO` x as many non-passers stratified evenly across deciles of
filter_rmsd above the bar. Stratification is for rank coverage, and it has a cost worth
stating: it distorts the marginal distribution of filter_rmsd, so a naive rank
correlation over the sample is biased. Hence the primary test is the two-class
separation (unbiased given known strata) and the correlation is secondary and flagged.

Criteria (pre-registered)
-------------------------
PRIMARY -- two-class separation, passers vs non-passers, within each target:
  * PASS      AUC >= 0.65 on BOTH targets, and above the null control on both
  * WEAK      AUC >= 0.60 on both and above the null on both
  * FAIL      anything else, including "works on one target only"

The null control is `ll_antigen` (ESM-IF1 on chain A). Chain A's SEQUENCE is invariant
across designs, but its refolded STRUCTURE is not, so ll_antigen measures *global refold
quality*. It is expected to fire here, and that is the point: it is the quantity that
separates "the whole structure is bad" from "the interface specifically is bad". A
packstat that does not beat it has shown nothing interface-specific.

SECONDARY (reported, not decisive): Spearman of each metric against -filter_rmsd over
the stratified sample, and whether packstat adds anything over ipTM, which is already
free and already in the pool.

Usage:
    python3 experiments/scoring_calibration/build_design_sets.py
"""

import argparse
import csv
import glob
import json
import os
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
SETS = HERE / "sets"
ROOT = HERE.parent.parent

RMSD_BAR = 2.5
NEG_RATIO = 3
N_STRATA = 10

# (key, metrics csv, refold cif glob, tracked-in-git?)
CAMPAIGNS = [
    ("ctrop2",
     "data/designs/ctrop2_antibody/run1/final_ranked_designs/all_designs_metrics.csv",
     "data/designs/ctrop2_antibody/run1/intermediate_designs_inverse_folded/refold_cif/*.cif",
     True),
    # Gitignored campaign; its manifest and results are gitignored too (.gitignore).
    ("chrna2",
     "experiments/actual_campaigns/chrna2/campaign/run1/final_ranked_designs/all_designs_metrics.csv",
     "experiments/actual_campaigns/chrna2/campaign/run1/intermediate_designs_inverse_folded/refold_cif/*.cif",
     False),
]

CARRY = ["id", "final_rank", "design_to_target_iptm", "design_ptm",
         "min_design_to_target_pae", "filter_rmsd", "plip_hbonds_refolded",
         "delta_sasa_refolded", "num_prot_tokens"]
FIELDS = ["entry_id", "target", "cif_path", "stratum", "label"] + CARRY[1:]


def build(key: str, metrics_csv: str, cif_glob: str, seed_note: str) -> list[dict]:
    rows = list(csv.DictReader(open(ROOT / metrics_csv)))
    cifs = {os.path.basename(p)[:-4]: p for p in glob.glob(str(ROOT / cif_glob))}
    usable = [r for r in rows if r["id"] in cifs and r.get("filter_rmsd")]
    if len(usable) != len(rows):
        print(f"  [{key}] WARNING {len(rows)-len(usable)} rows dropped "
              f"(no cif or no filter_rmsd)")

    passers = [r for r in usable if float(r["filter_rmsd"]) <= RMSD_BAR]
    fails = sorted((r for r in usable if float(r["filter_rmsd"]) > RMSD_BAR),
                   key=lambda r: float(r["filter_rmsd"]))

    # Even strata across the failing range, then a deterministic even sweep within each
    # stratum -- no RNG, so the manifest is reproducible without a seed.
    want = min(len(fails), NEG_RATIO * len(passers))
    per = max(1, want // N_STRATA)
    chunk = max(1, len(fails) // N_STRATA)
    picked, strata = [], {}
    for s in range(N_STRATA):
        block = fails[s * chunk: (s + 1) * chunk if s < N_STRATA - 1 else len(fails)]
        if not block:
            continue
        step = max(1, len(block) // per)
        take = block[::step][:per]
        for r in take:
            strata[r["id"]] = s
        picked += take

    out = []
    for r in passers + picked:
        out.append({
            "entry_id": r["id"], "target": key, "cif_path": cifs[r["id"]],
            "stratum": strata.get(r["id"], -1),       # -1 = passer
            "label": 1 if float(r["filter_rmsd"]) <= RMSD_BAR else 0,
            **{k: r.get(k, "") for k in CARRY[1:]},
        })
    rm = [float(r["filter_rmsd"]) for r in out]
    print(f"  [{key}] {len(out)} selected: {len(passers)} pass / {len(picked)} fail "
          f"| filter_rmsd {min(rm):.2f}-{max(rm):.2f}, median {st.median(rm):.2f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--neg-ratio", type=int, default=NEG_RATIO)
    args = ap.parse_args()
    globals()["NEG_RATIO"] = args.neg_ratio

    SETS.mkdir(parents=True, exist_ok=True)
    summary = {"rmsd_bar": RMSD_BAR, "neg_ratio": args.neg_ratio,
               "n_strata": N_STRATA, "targets": {}}
    for key, mcsv, cglob, tracked in CAMPAIGNS:
        rows = build(key, mcsv, cglob, key)
        path = SETS / f"designs_{key}.tsv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t",
                               extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        summary["targets"][key] = {
            "n": len(rows),
            "n_pass": sum(r["label"] for r in rows),
            "n_fail": sum(1 - r["label"] for r in rows),
            "tracked_in_git": tracked,
            "manifest": str(path.relative_to(ROOT)),
        }
        print(f"  [{key}] -> {path.name}"
              f"{'' if tracked else '   (gitignored)'}")

    # Only the tracked targets' counts go in the committed summary.
    pub = {**summary, "targets": {k: v for k, v in summary["targets"].items()
                                  if v["tracked_in_git"]}}
    pub["withheld_targets"] = sorted(k for k, v in summary["targets"].items()
                                     if not v["tracked_in_git"])
    (SETS / "designs_summary.json").write_text(json.dumps(pub, indent=2) + "\n")
    print(f"\nwrote {SETS/'designs_summary.json'} "
          f"(withholding: {', '.join(pub['withheld_targets']) or 'none'})")
    print(f"total to score: {sum(v['n'] for v in summary['targets'].values())}")


if __name__ == "__main__":
    main()
