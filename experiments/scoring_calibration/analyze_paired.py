#!/usr/bin/env python3
"""Regenerate the paired within-parent test (EXPERIMENT_02's only valid column).

Fourth analysis script here, after `analyze.py` (EXPERIMENT_01), `analyze_metrics.py`
(EXPERIMENT_02) and `analyze_asd.py` (EXPERIMENT_03). Writes
`results/paired_comparison.{md,json}`.

It exists because the 28-parent result was first computed with a throwaway shell
script and survived only as hand-transcribed prose in a commit message — the exact
gap that `analyze_metrics.py` was written to close for EXPERIMENT_02, repeated.

The test
--------
For each parent complex holding BOTH a binding-retained and a binding-destroyed point
mutant, does the metric score the retained one higher? A sign test over parents, with
an exact one-sided binomial p. This controls for complex identity completely, which is
why EXPERIMENT_02 established it as the only trustworthy column: the unpaired AUC is
confounded by which parents land in which arm, and the cross-complex correlation by
antigen identity.

Three columns are reported, and the third is the one that matters
-----------------------------------------------------------------
    all          every parent with both classes (currently 28)
    original     the 17 parents that were in the benchmark before the top-up
    new only     the 11 parents added by the top-up

`all` is NOT an independent replication of `original` — the 17 are inside the 28. The
`new only` column is the independent test, and it is small, so a metric that is real
should show the same direction there even if it misses significance.

Null control
------------
`ll_antigen` is ESM-IF1's score for the antigen chain. Within a parent the antigen is
identical between the retained and destroyed mutant (the mutation is in H or L), so it
cannot discriminate and must land at chance. It is included for the same reason it was
in EXPERIMENT_02 and EXPERIMENT_03, where it changed the conclusion both times.

Multiple comparisons
--------------------
A Bonferroni threshold over the metrics tested is reported alongside the raw p, because
an uncorrected p=0.025 across seven metrics is not a finding — a lesson this benchmark
taught directly when shape complementarity and buried SASA, both p=0.025 at 17 parents,
collapsed to chance at 28.

Stdlib only.

Usage:
    python3 experiments/scoring_calibration/analyze_paired.py
"""

import csv
import json
import statistics as st
from collections import defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SETS = HERE / "sets"
OUT_MD = RES / "paired_comparison.md"
OUT_JSON = RES / "paired_comparison.json"

# (label, source table, accessor, is_null_control). Oriented so higher = better binder,
# so Rosetta's energy term is negated.
METRICS = [
    ("packstat", lambda r: float(r["r"]["packstat"]), False),
    ("-dG/dSASA", lambda r: -float(r["r"]["dG_dSASA_x100"]), False),
    ("-dG_separated", lambda r: -float(r["r"]["dG_separated"]), False),
    ("shape complementarity", lambda r: float(r["r"]["sc_value"]), False),
    ("buried SASA", lambda r: float(r["r"]["dSASA_int"]), False),
    ("interface H-bonds", lambda r: float(r["r"]["interface_hbonds"]), False),
    ("ipTM", lambda r: float(r["b"]["iptm"]), False),
    ("ipSAE_min", lambda r: float(r["i"]["ipsae_min"]), False),
    ("ESM-IF1 ll_antibody", lambda r: float(r["e"]["ll_antibody_mean"]), False),
    ("ESM-IF1 ll_antigen", lambda r: float(r["e"]["ll_antigen"]), True),
]


def binom_p(k: int, n: int) -> float:
    """Exact one-sided binomial tail, P(X >= k | p = 0.5)."""
    if n == 0:
        return float("nan")
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def load_ok(name: str) -> dict:
    with (RES / name).open(encoding="utf-8") as fh:
        return {r["entry_id"]: r for r in csv.DictReader(fh, delimiter="\t")
                if r.get("status") == "ok"}


def sign_test(rows: list[dict], fn) -> tuple[int, int, list[str]]:
    pos, neg = defaultdict(list), defaultdict(list)
    for r in rows:
        if r["set"] == "matched_positive":
            pos[r["pdb"]].append(fn(r))
        elif r["set"] == "hard_negative":
            neg[r["pdb"]].append(fn(r))
    shared = sorted(set(pos) & set(neg))
    wins = sum(1 for p in shared if st.median(pos[p]) > st.median(neg[p]))
    return wins, len(shared), shared


def main() -> None:
    ros, ips, esm = load_ok("rosetta.tsv"), load_ok("ipsae.tsv"), load_ok("esmif.tsv")
    with (RES / "scores.tsv").open(encoding="utf-8") as fh:
        base = [r for r in csv.DictReader(fh, delimiter="\t") if r["status"] == "ok"]
    with (SETS / "paired_topup.tsv").open(encoding="utf-8") as fh:
        topup = {r["entry_id"] for r in csv.DictReader(fh, delimiter="\t")}

    rows = [{"b": b, "r": ros[b["entry_id"]], "i": ips[b["entry_id"]],
             "e": esm[b["entry_id"]], "set": b["set"], "pdb": b["pdb"],
             "eid": b["entry_id"]}
            for b in base
            if b["entry_id"] in ros and b["entry_id"] in ips and b["entry_id"] in esm
            and b["set"] in ("matched_positive", "hard_negative")]

    pre = [r for r in rows if r["eid"] not in topup]
    _, _, all_parents = sign_test(rows, METRICS[0][1])
    _, _, pre_parents = sign_test(pre, METRICS[0][1])
    new_parents = [p for p in all_parents if p not in pre_parents]
    new_rows = [r for r in rows if r["pdb"] in new_parents]

    n_tested = sum(1 for _, _, c in METRICS if not c)
    bonf = 0.05 / n_tested

    rep = {"n_entries": len(rows),
           "parents": {"all": len(all_parents), "pre_topup": len(pre_parents),
                       "added_by_topup": len(new_parents),
                       "new_parent_ids": new_parents},
           "n_metrics_tested": n_tested,
           "bonferroni_alpha": round(bonf, 5),
           "metrics": []}

    for lab, fn, ctl in METRICS:
        wa, na, _ = sign_test(rows, fn)
        wp, np_, _ = sign_test(pre, fn)
        wn, nn, _ = sign_test(new_rows, fn)
        pa = binom_p(wa, na)
        rep["metrics"].append({
            "metric": lab, "null_control": ctl,
            "all": {"wins": wa, "n": na, "p": round(pa, 5)},
            "pre_topup": {"wins": wp, "n": np_, "p": round(binom_p(wp, np_), 5)},
            "new_only": {"wins": wn, "n": nn, "p": round(binom_p(wn, nn), 5)},
            "survives_bonferroni": bool(pa <= bonf) and not ctl,
            "same_direction_in_new": bool(nn and (wn / nn > 0.5) and (wa / na > 0.5)),
        })

    OUT_JSON.write_text(json.dumps(rep, indent=2) + "\n")

    survivors = [m["metric"] for m in rep["metrics"] if m["survives_bonferroni"]]
    L = ["# Paired within-parent test (generated)", "",
         f"`analyze_paired.py` over {rep['n_entries']} mutant complexes on "
         f"**{rep['parents']['all']} parent complexes**. Narrative: "
         f"[EXPERIMENT_02](../EXPERIMENT_02_alternative_metrics.md).", "",
         "For each parent holding both a binding-retained and a binding-destroyed point "
         "mutant, does the metric score the retained one higher? Sign test over parents, "
         "exact one-sided binomial p. This controls for complex identity completely and "
         "is the only column EXPERIMENT_02 found trustworthy.", "",
         f"Bonferroni α over {n_tested} metrics = **{bonf:.4f}**.", "",
         "| metric | all | p | original 17 | **new 11 only** | p |",
         "|---|---|---|---|---|---|"]
    for m in rep["metrics"]:
        name = f"*{m['metric']}* (null control)" if m["null_control"] else m["metric"]
        if m["survives_bonferroni"]:
            name = f"**{m['metric']}**"
        L.append(f"| {name} | {m['all']['wins']}/{m['all']['n']} | "
                 f"{m['all']['p']:.4f} | {m['pre_topup']['wins']}/{m['pre_topup']['n']} | "
                 f"{m['new_only']['wins']}/{m['new_only']['n']} | "
                 f"{m['new_only']['p']:.3f} |")
    L += ["",
          f"**Survives Bonferroni: {', '.join(survivors) if survivors else 'none'}.**", "",
          "The `all` column is **not** an independent replication of `original 17` — "
          "those 17 parents are inside the 28. The `new 11 only` column is the "
          "independent test; it is small, so a real metric should agree in *direction* "
          "there even where it misses significance.", "",
          "> The null control must land at chance. Within a parent the antigen is "
          "identical between the retained and destroyed mutant (the mutation is in the "
          "heavy or light chain), so it cannot discriminate. It changed the conclusion "
          "in both EXPERIMENT_02 and EXPERIMENT_03.", "",
          "## Why the Bonferroni column matters here", "",
          "This benchmark taught the lesson directly. At 17 parents, shape "
          "complementarity and buried SASA were both 13/17 (p=0.025) and were reported "
          "as significant findings. At 28 parents both fell to 16/28, scoring 5/11 and "
          "4/11 on the new parents — chance. An uncorrected p≈0.025 across seven "
          "metrics was never a finding.", ""]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_MD}\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
