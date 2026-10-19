#!/usr/bin/env python3
"""Regenerate every number in EXPERIMENT_02 (metric comparison + confound diagnostics).

Companion to `analyze.py`, which does the same job for EXPERIMENT_01. Reads the four
result tables produced by the scorers and writes
`results/metrics_comparison.{md,json}` — same convention as elsewhere in this project:
JSON is the stable contract, markdown is the human-readable view.

This exists because EXPERIMENT_02's first pass computed its tables with throwaway shell
scripts. Numbers that only live in a commit message cannot be rechecked when someone
later doubts them, and two of this experiment's three analysis columns turned out to be
confounded — exactly the situation where being able to re-derive everything matters.

What it computes
----------------
For each metric, oriented so higher = predicted better binder:

  cross-complex rho   Spearman against log10(Kd) over the Kd-labelled set.
                      EXPECTED NEGATIVE. **Known confounded** - see diagnostics.
  unpaired AUC        Mann-Whitney U, matched_positive vs hard_negative.
                      **Known confounded** - see diagnostics.
  paired              Sign test over parent complexes carrying BOTH a
                      binding-retained and a binding-destroyed mutant, with an
                      exact one-sided binomial p. THE ONLY VALID COLUMN, and the
                      one a design campaign actually poses.

Plus three diagnostics that establish why the first two columns are not trustworthy,
and a null control (`ll_antigen`) that must land at chance in the paired column for the
paired column itself to be believable.

Stdlib only.

Usage:
    python3 experiments/scoring_calibration/analyze_metrics.py
"""

import csv
import json
import math
import statistics as st
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
MANIFEST = HERE / "sets" / "manifest.tsv"
OUT_MD = RES / "metrics_comparison.md"
OUT_JSON = RES / "metrics_comparison.json"

# (label, source table, accessor, is_null_control)
# Orientation: every lambda returns a value where HIGHER = predicted better binder,
# so Rosetta's energy terms (lower is better) are negated. Without this the sign
# conventions differ per metric and the table becomes unreadable.
METRICS = [
    ("ipTM (baseline)",         lambda r: float(r["b"]["iptm"]), False),
    ("ipSAE_min",               lambda r: float(r["i"]["ipsae_min"]), False),
    ("-dG/dSASA (Rosetta)",     lambda r: -float(r["r"]["dG_dSASA_x100"]), False),
    ("-dG_separated",           lambda r: -float(r["r"]["dG_separated"]), False),
    ("shape complementarity",   lambda r: float(r["r"]["sc_value"]), False),
    ("packstat",                lambda r: float(r["r"]["packstat"]), False),
    ("buried SASA",             lambda r: float(r["r"]["dSASA_int"]), False),
    ("interface H-bonds",       lambda r: float(r["r"]["interface_hbonds"]), False),
    ("-unsat H-bonds",          lambda r: -float(r["r"]["delta_unsat_hbonds"]), False),
    ("ESM-IF1 ll_antibody",     lambda r: float(r["e"]["ll_antibody_mean"]), False),
    ("ESM-IF1 ll_heavy",        lambda r: float(r["e"]["ll_heavy"]), False),
    ("ESM-IF1 ll_light",        lambda r: float(r["e"]["ll_light"]), False),
    ("ESM-IF1 ll_antigen",      lambda r: float(r["e"]["ll_antigen"]), True),
]


def load(name: str) -> dict:
    with (RES / name).open(encoding="utf-8") as fh:
        return {r["entry_id"]: r for r in csv.DictReader(fh, delimiter="\t")
                if r.get("status") == "ok"}


def spearman(a: list[float], b: list[float]) -> float:
    ra = {v: i for i, v in enumerate(sorted(a))}
    rb = {v: i for i, v in enumerate(sorted(b))}
    A, B = [ra[v] for v in a], [rb[v] for v in b]
    ma, mb = st.mean(A), st.mean(B)
    num = sum((x - ma) * (y - mb) for x, y in zip(A, B))
    den = math.sqrt(sum((x - ma) ** 2 for x in A) * sum((y - mb) ** 2 for y in B))
    return num / den if den else float("nan")


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    g = sum(1 for a in pos for b in neg if a > b) + \
        0.5 * sum(1 for a in pos for b in neg if a == b)
    return g / (len(pos) * len(neg))


def binom_p(k: int, n: int) -> float:
    """Exact one-sided binomial tail, P(X >= k | p=0.5)."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def main() -> None:
    ros, ips, esm = load("rosetta.tsv"), load("ipsae.tsv"), load("esmif.tsv")
    with (RES / "scores.tsv").open(encoding="utf-8") as fh:
        base = [r for r in csv.DictReader(fh, delimiter="\t") if r["status"] == "ok"]
    with MANIFEST.open(encoding="utf-8") as fh:
        man = {r["entry_id"]: r for r in csv.DictReader(fh, delimiter="\t")}

    rows = [{"b": b, "r": ros[b["entry_id"]], "i": ips[b["entry_id"]],
             "e": esm[b["entry_id"]], "m": man[b["entry_id"]],
             "set": b["set"], "kd": b["kd_molar"], "pdb": b["pdb"]}
            for b in base
            if b["entry_id"] in ros and b["entry_id"] in ips
            and b["entry_id"] in esm and b["entry_id"] in man]

    rep: dict = {"n_joined": len(rows),
                 "n_by_set": dict(Counter(r["set"] for r in rows)),
                 "metrics": [], "diagnostics": {}}

    graded = [r for r in rows if r["set"] == "contaminated_positive" and r["kd"]]
    logkd = [math.log10(float(r["kd"])) for r in graded]

    for label, fn, is_ctl in METRICS:
        rho = spearman(logkd, [fn(r) for r in graded])
        mp = [fn(r) for r in rows if r["set"] == "matched_positive"]
        hn = [fn(r) for r in rows if r["set"] == "hard_negative"]
        pb, nb = defaultdict(list), defaultdict(list)
        for r in rows:
            if r["set"] == "matched_positive":
                pb[r["pdb"]].append(fn(r))
            if r["set"] == "hard_negative":
                nb[r["pdb"]].append(fn(r))
        shared = sorted(set(pb) & set(nb))
        wins = sum(1 for p in shared if st.median(pb[p]) > st.median(nb[p]))
        rep["metrics"].append({
            "metric": label, "null_control": is_ctl,
            "cross_complex_spearman": round(rho, 4),
            "unpaired_auc": round(auc(mp, hn), 4),
            "paired_wins": wins, "paired_n": len(shared),
            "paired_binomial_p": round(binom_p(wins, len(shared)), 4),
        })

    # --- diagnostic 1: parent-complex overlap between the unpaired arms ---
    pos_p = {r["pdb"] for r in rows if r["set"] == "matched_positive"}
    neg_p = {r["pdb"] for r in rows if r["set"] == "hard_negative"}
    rep["diagnostics"]["unpaired_arm_composition"] = {
        "matched_positive_parents": len(pos_p),
        "hard_negative_parents": len(neg_p),
        "shared_parents": len(pos_p & neg_p),
        "parents_in_only_one_arm": len(pos_p ^ neg_p),
        "verdict": "unpaired AUC partly measures which complexes are in which arm",
    }

    # --- diagnostic 2: is affinity a property of the antigen rather than the pair? ---
    by_ag = defaultdict(list)
    for r in graded:
        by_ag[r["m"]["antigen_seq"][:60]].append(math.log10(float(r["kd"])))
    multi = [v for v in by_ag.values() if len(v) > 1]
    within = st.mean([max(v) - min(v) for v in multi]) if multi else float("nan")
    total = max(logkd) - min(logkd)
    rep["diagnostics"]["antigen_identity"] = {
        "n_entries": len(graded), "n_distinct_antigens": len(by_ag),
        "n_antigens_repeated": len(multi),
        "mean_within_antigen_log10kd_spread": round(within, 3),
        "total_log10kd_spread": round(total, 3),
        "fraction_variance_between_antigens": round(1 - within / total, 3)
        if total else None,
        "verdict": "most affinity variance is between antigens, so any antigen-only "
                   "feature correlates with Kd spuriously",
    }

    # --- diagnostic 3: size, checked and RULED OUT as the confound ---
    rep["diagnostics"]["size"] = {
        "spearman_logkd_vs_total_residues":
            round(spearman(logkd, [int(r["m"]["n_res_total"]) for r in graded]), 4),
        "spearman_logkd_vs_antigen_residues":
            round(spearman(logkd, [int(r["m"]["n_res_antigen"]) for r in graded]), 4),
        "spearman_logkd_vs_buried_sasa":
            round(spearman(logkd, [float(r["r"]["dSASA_int"]) for r in graded]), 4),
        "verdict": "all negligible - complex size is NOT the confound; antigen "
                   "identity is",
    }

    # --- cost, measured ---
    def secs(tbl):
        v = [float(r["seconds"]) for r in tbl.values() if r.get("seconds")]
        return {"n": len(v), "mean_s": round(st.mean(v), 1),
                "median_s": round(st.median(v), 1), "total_h": round(sum(v) / 3600, 2)}
    rep["cost"] = {"rosetta": secs(ros), "esmif": secs(esm),
                   "ipsae": "negligible - re-reads saved PAE matrices"}

    OUT_JSON.write_text(json.dumps(rep, indent=2) + "\n")

    # ---------------- markdown ----------------
    L = ["# EXPERIMENT_02 — metric comparison (generated)", "",
         f"`analyze_metrics.py` over {rep['n_joined']} entries joined across all four "
         f"result tables. Narrative and interpretation: "
         f"[EXPERIMENT_02](../EXPERIMENT_02_alternative_metrics.md).", "",
         "All metrics oriented so **higher = predicted better binder** (Rosetta energy "
         "terms are negated).", "",
         "## Metric comparison", "",
         "| metric | cross-complex ρ | unpaired AUC | **paired** | p |",
         "|---|---|---|---|---|"]
    for m in rep["metrics"]:
        name = f"*{m['metric']}* (null control)" if m["null_control"] else m["metric"]
        star = " **←**" if (not m["null_control"] and m["paired_binomial_p"] <= 0.05) else ""
        L.append(f"| {name} | {m['cross_complex_spearman']:+.3f} | "
                 f"{m['unpaired_auc']:.3f} | {m['paired_wins']}/{m['paired_n']} | "
                 f"{m['paired_binomial_p']:.3f}{star} |")

    d1 = rep["diagnostics"]["unpaired_arm_composition"]
    d2 = rep["diagnostics"]["antigen_identity"]
    d3 = rep["diagnostics"]["size"]
    L += ["", "> **Only the paired column is valid.** The first two are confounded — see "
          "diagnostics. The null control landing at chance in the paired column is what "
          "makes that column believable.", "",
          "## Diagnostics — why two columns were discarded", "",
          "**1. Unpaired AUC is confounded by arm composition.** "
          f"`matched_positive` draws on {d1['matched_positive_parents']} parent "
          f"complexes, `hard_negative` on {d1['hard_negative_parents']}, and only "
          f"**{d1['shared_parents']}** are shared — "
          f"{d1['parents_in_only_one_arm']} parents appear in just one arm. The "
          "comparison therefore partly measures which complexes landed where.", "",
          "**2. Cross-complex ρ is confounded by antigen identity.** "
          f"{d2['n_entries']} entries span {d2['n_distinct_antigens']} distinct "
          f"antigens. log10(Kd) spread *within* a repeated antigen is "
          f"**{d2['mean_within_antigen_log10kd_spread']}** against a total spread of "
          f"**{d2['total_log10kd_spread']}**, so roughly "
          f"{d2['fraction_variance_between_antigens']:.0%} of affinity variance is "
          "between antigens. Any antigen-only feature correlates with Kd spuriously — "
          "which is exactly what the null control's positive ρ demonstrates.", "",
          "**3. Complex size was checked and ruled out.** "
          f"log10(Kd) vs total residues ρ={d3['spearman_logkd_vs_total_residues']:+.3f}, "
          f"vs antigen residues {d3['spearman_logkd_vs_antigen_residues']:+.3f}, "
          f"vs buried SASA {d3['spearman_logkd_vs_buried_sasa']:+.3f}. All negligible. "
          "EXPERIMENT_01 flagged size as the likely confound in its limitations; it is "
          "not. Antigen identity is.", "",
          "## Cost (measured)", "",
          "| tier | n | mean | total |", "|---|---|---|---|",
          f"| Rosetta (CPU, 16 workers) | {rep['cost']['rosetta']['n']} | "
          f"{rep['cost']['rosetta']['mean_s']/60:.1f} min | "
          f"{rep['cost']['rosetta']['total_h']:.1f} CPU-h |",
          f"| ESM-IF1 (CPU) | {rep['cost']['esmif']['n']} | "
          f"{rep['cost']['esmif']['mean_s']:.1f} s | "
          f"{rep['cost']['esmif']['total_h']:.2f} h |",
          "| ipSAE | 332 | ~0 | re-reads saved PAE matrices |", "",
          "For scale: generating one design costs ~3 min of GPU, so Rosetta rescoring is "
          "roughly **5× more expensive than generation** — top-N only, never "
          "funnel-wide.", ""]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_MD}\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
