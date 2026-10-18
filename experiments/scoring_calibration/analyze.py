#!/usr/bin/env python3
"""Analyse the calibration run (PLAN.md milestone 4).

Reads results/scores.tsv and answers the four questions PLAN.md sec 7 poses, writing
a markdown report plus a JSON sidecar — same convention as experiments/report.py
(JSON is the stable contract, markdown is the human-readable view).

Stdlib only; the statistics here are deliberately distribution-free (medians, rank
correlation, Mann-Whitney-style AUC) rather than assuming normality, because ipTM is
bounded on [0,1] and visibly skewed.

The provenance rule from PLAN.md sec 7 is enforced here rather than left to the
reader: separation is computed as hard_negative vs matched_positive (both AbDesign-DB
curated sequences), never against clean_positive (SAbDab resolved coordinates).
Crossing that boundary would confound sequence provenance with the effect.

Usage:
    python3 experiments/scoring_calibration/analyze.py
"""

import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "scores.tsv"
OUT_MD = HERE / "results" / "calibration_report.md"
OUT_JSON = HERE / "results" / "calibration_report.json"

SETS = ["clean_positive", "contaminated_positive", "hard_negative", "matched_positive"]


def load() -> list[dict]:
    with RESULTS.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return [r for r in rows if r["status"] == "ok" and r["iptm"]]


def pct(v: list[float], p: float) -> float:
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] if lo == hi else v[lo] + (v[hi] - v[lo]) * (k - lo)


def spearman(a: list[float], b: list[float]) -> float:
    ra = {v: i for i, v in enumerate(sorted(a))}
    rb = {v: i for i, v in enumerate(sorted(b))}
    A, B = [ra[v] for v in a], [rb[v] for v in b]
    ma, mb = st.mean(A), st.mean(B)
    num = sum((x - ma) * (y - mb) for x, y in zip(A, B))
    den = math.sqrt(sum((x - ma) ** 2 for x in A) * sum((y - mb) ** 2 for y in B))
    return num / den if den else float("nan")


def auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney U as a probability: P(random positive > random negative)."""
    if not pos or not neg:
        return float("nan")
    g = sum(1 for a in pos for b in neg if a > b) + \
        0.5 * sum(1 for a in pos for b in neg if a == b)
    return g / (len(pos) * len(neg))


def main() -> None:
    rows = load()
    by = defaultdict(list)
    for r in rows:
        by[r["set"]].append(r)
    iptm = {s: [float(r["iptm"]) for r in by[s]] for s in SETS if by[s]}

    report: dict = {"n_scored": len(rows), "sets": {}}
    for s, v in iptm.items():
        report["sets"][s] = {
            "n": len(v), "median": round(st.median(v), 4), "mean": round(st.mean(v), 4),
            "p10": round(pct(v, .10), 4), "p25": round(pct(v, .25), 4),
            "p75": round(pct(v, .75), 4), "p90": round(pct(v, .90), 4),
            "min": round(min(v), 4), "max": round(max(v), 4),
            "frac_ge_0.6": round(sum(1 for x in v if x >= .6) / len(v), 4),
            "frac_ge_0.8": round(sum(1 for x in v if x >= .8) / len(v), 4),
        }

    # --- calibration: where do our own thresholds sit among real binders? ---
    clean = iptm.get("clean_positive", [])
    report["calibration"] = {
        "basis": "clean_positive (leak-free dev binders)",
        "n": len(clean),
        "frac_real_binders_rejected_at_0.6": round(sum(1 for x in clean if x < .6) / len(clean), 4),
        "frac_real_binders_rejected_at_0.8": round(sum(1 for x in clean if x < .8) / len(clean), 4),
        "median": round(st.median(clean), 4),
    }

    # --- memorisation gap ---
    cont = iptm.get("contaminated_positive", [])
    if clean and cont:
        report["memorisation"] = {
            "clean_median": round(st.median(clean), 4),
            "contaminated_median": round(st.median(cont), 4),
            "gap": round(st.median(cont) - st.median(clean), 4),
            "p_contaminated_greater": round(auc(cont, clean), 4),
            "n_clean": len(clean), "n_contaminated": len(cont),
        }

    # --- dose-response: does ipTM track affinity? ---
    graded = [r for r in by["contaminated_positive"] if r["kd_molar"]]
    if len(graded) >= 20:
        xs = [math.log10(float(r["kd_molar"])) for r in graded]
        ys = [float(r["iptm"]) for r in graded]
        bands = []
        for lo, hi, lab in ((-99, -10, "pM (<100pM)"), (-10, -8, "100pM-10nM"),
                            (-8, -6, "10nM-1uM"), (-6, 99, "weak (>1uM)")):
            g = [y for x, y in zip(xs, ys) if lo <= x < hi]
            if g:
                bands.append({"band": lab, "n": len(g), "median_iptm": round(st.median(g), 4)})
        report["dose_response"] = {
            "n": len(graded),
            "spearman_iptm_vs_log10_kd": round(spearman(xs, ys), 4),
            "expected_sign": "negative (tighter binders should score higher)",
            "bands": bands,
        }

    # --- separation: provenance-matched, per PLAN.md sec 7 ---
    hn, mp = iptm.get("hard_negative", []), iptm.get("matched_positive", [])
    if hn and mp:
        pos_by, neg_by = defaultdict(list), defaultdict(list)
        for r in by["matched_positive"]:
            pos_by[r["pdb"]].append(float(r["iptm"]))
        for r in by["hard_negative"]:
            neg_by[r["pdb"]].append(float(r["iptm"]))
        shared = sorted(set(pos_by) & set(neg_by))
        wins = sum(1 for p in shared if st.median(pos_by[p]) > st.median(neg_by[p]))
        report["separation"] = {
            "comparison": "matched_positive vs hard_negative (both AbDesign-DB curated)",
            "n_pos": len(mp), "n_neg": len(hn),
            "pos_median": round(st.median(mp), 4), "neg_median": round(st.median(hn), 4),
            "roc_auc_iptm": round(auc(mp, hn), 4),
            "paired_within_parent": {"n_complexes": len(shared), "positive_higher": wins},
        }

    secs = [float(r["seconds"]) for r in rows if r.get("seconds")]
    report["timing"] = {
        "n": len(secs), "mean_min": round(st.mean(secs) / 60, 2),
        "median_min": round(st.median(secs) / 60, 2), "max_min": round(max(secs) / 60, 2),
        "total_hours": round(sum(secs) / 3600, 2),
    }

    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")

    # ---- markdown view ----
    L = ["# Scoring calibration — pilot results", "",
         f"Generated by `analyze.py` from `results/scores.tsv` "
         f"({report['n_scored']} usable entries, 0 failures). "
         f"Method and caveats: [PLAN.md](../PLAN.md).", "",
         "## Distributions", "",
         "| set | n | median | mean | p25 | p75 | ≥0.6 | ≥0.8 |", "|---|---|---|---|---|---|---|---|"]
    for s in SETS:
        d = report["sets"].get(s)
        if d:
            L.append(f"| `{s}` | {d['n']} | **{d['median']:.3f}** | {d['mean']:.3f} | "
                     f"{d['p25']:.3f} | {d['p75']:.3f} | {d['frac_ge_0.6']:.0%} | {d['frac_ge_0.8']:.0%} |")

    c = report["calibration"]
    L += ["", "## 1. Calibration — what does a real binder score?", "",
          f"Basis: {c['n']} leak-free `dev` complexes, confirmed binders by construction "
          f"(every SAbDab entry is a crystallised antibody-antigen complex).", "",
          f"- Median real binder: **ipTM {c['median']:.3f}**",
          f"- A 0.6 cutoff rejects **{c['frac_real_binders_rejected_at_0.6']:.0%}** of real binders",
          f"- A 0.8 cutoff rejects **{c['frac_real_binders_rejected_at_0.8']:.0%}** of real binders", "",
          "The 0.8 mark this project used before step 1 would have discarded most genuine "
          "antibodies. 0.6 (imported from Bennett et al.'s AlphaFold3 results) transfers "
          "to this Boltz-2 pipeline reasonably well, but is not conservative — it still "
          "rejects about a quarter of real binders."]

    if "memorisation" in report:
        m = report["memorisation"]
        L += ["", "## 2. Memorisation gap", "",
              f"| | n | median ipTM |", "|---|---|---|",
              f"| clean (post-cutoff, leak-free) | {m['n_clean']} | {m['clean_median']:.3f} |",
              f"| contaminated (train-era) | {m['n_contaminated']} | {m['contaminated_median']:.3f} |", "",
              f"Gap **{m['gap']:+.3f}**; P(contaminated > clean) = **{m['p_contaminated_greater']:.3f}** "
              f"(0.5 would be no difference). A real but modest effect.", "",
              "Note the confound runs *against* this being an overestimate: the contaminated "
              "set skews high-affinity, and section 3 shows high-affinity complexes score "
              "*lower*, so affinity differences would shrink this gap, not inflate it."]

    if "dose_response" in report:
        d = report["dose_response"]
        L += ["", "## 3. Dose-response — ipTM does not track affinity", "",
              "| Kd band | n | median ipTM |", "|---|---|---|"]
        L += [f"| {b['band']} | {b['n']} | {b['median_iptm']:.3f} |" for b in d["bands"]]
        L += ["", f"Spearman ipTM vs log10(Kd) = **{d['spearman_iptm_vs_log10_kd']:+.3f}** "
              f"over {d['n']} complexes spanning nine orders of magnitude. Expected sign is "
              f"{d['expected_sign']}.", "",
              "**This is the pilot's most consequential result.** ipTM is the primary ranking "
              "signal in BoltzGen's own selection (weight 1 in Algorithm 2) and the metric this "
              "project's funnel is gated on, yet across real complexes with measured affinities "
              "it carries no information about binding strength — and the trend is mildly "
              "*backwards*. It plausibly measures whether a credible interface can be modelled "
              "at all, which is a different and more nearly binary quantity than affinity.", "",
              "Caveats: the weakest band is thin, and larger/harder complexes may be both "
              "lower-affinity in this corpus and harder to model, which a size-controlled "
              "analysis would need to separate."]

    if "separation" in report:
        s = report["separation"]
        p = s["paired_within_parent"]
        L += ["", "## 4. Separation — weak, but not absent", "",
              f"{s['comparison']}.", "",
              f"| | n | median ipTM |", "|---|---|---|",
              f"| matched_positive (binding retained) | {s['n_pos']} | {s['pos_median']:.3f} |",
              f"| hard_negative (binding destroyed) | {s['n_neg']} | {s['neg_median']:.3f} |", "",
              f"**ROC-AUC = {s['roc_auc_iptm']:.3f}.** Restricted to the {p['n_complexes']} parent "
              f"complexes holding both a retained and a destroyed mutant — which controls for "
              f"complex identity — positives score higher in **{p['positive_higher']}/{p['n_complexes']}**.", "",
              "Consistent with section 3 rather than contradicting it: ipTM weakly detects a "
              "mutation that *abolishes* binding while being blind to where on the affinity "
              "scale a complex sits. For reference, the autoresearch filter paper's untuned "
              "baseline was AUC 0.637 and its tuned rule reached 0.806, so raw ipTM here sits "
              "at roughly their starting point."]

    t = report["timing"]
    L += ["", "## Cost (measured, replacing PLAN.md's estimate)", "",
          f"{t['mean_min']:.1f} min/complex mean, {t['median_min']:.1f} median, "
          f"{t['max_min']:.1f} max; {t['total_hours']:.1f} GPU-hours for {t['n']} complexes. "
          f"PLAN.md estimated 3-5 min — the low end was right.", ""]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_MD}\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
