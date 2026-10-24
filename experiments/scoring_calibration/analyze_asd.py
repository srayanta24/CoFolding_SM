#!/usr/bin/env python3
"""Regenerate every number in EXPERIMENT_03 (within-antigen affinity ranking).

Third analysis script in this folder, after `analyze.py` (EXPERIMENT_01) and
`analyze_metrics.py` (EXPERIMENT_02). Same convention: JSON is the stable contract,
markdown is the human view, and nothing is hand-transcribed.

The question
------------
EXPERIMENT_02 could not test affinity ranking: on that benchmark 78% of log10(Kd)
variance sat *between* antigens, so an antigen-only null control scored +0.341 and
invalidated the column. This benchmark removes that confound by construction — each
target group is ONE antigen with many antibodies, only the heavy chain varies, and
affinity is -log KD (higher = tighter).

The null control, again
-----------------------
`ll_antigen` is ESM-IF1's score for the *antigen* chain. Within a target the antigen
is identical across every entry, so it cannot carry affinity information. Any
correlation it shows is target-specific leakage that every structure-reading metric
inherits. It is reported alongside the real metrics, not buried, because on this
benchmark it fires.

Usage:
    python3 experiments/scoring_calibration/analyze_asd.py
"""

import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
MANIFEST = HERE / "sets" / "asd_manifest.tsv"
OUT_MD = RES / "asd_comparison.md"
OUT_JSON = RES / "asd_comparison.json"

# (column, label, is_null_control). Oriented so higher = predicted tighter binder,
# matching -log KD's own direction, so every expected correlation is POSITIVE.
METRICS = [
    ("iptm", "ipTM", False),
    ("ptm", "pTM", False),
    ("complex_plddt", "complex pLDDT", False),
    ("confidence_score", "confidence_score", False),
    ("ll_heavy", "ESM-IF1 ll_heavy", False),
    ("ll_light", "ESM-IF1 ll_light", False),
    ("ll_antibody_mean", "ESM-IF1 ll_antibody", False),
    ("ll_antigen", "ESM-IF1 ll_antigen", True),
]

ABBIBENCH_BASELINES = {"ProteinMPNN": 0.30, "ESM-IF1": 0.28}


def spearman(a: list[float], b: list[float]) -> float:
    ra = {v: i for i, v in enumerate(sorted(a))}
    rb = {v: i for i, v in enumerate(sorted(b))}
    A, B = [ra[v] for v in a], [rb[v] for v in b]
    ma, mb = st.mean(A), st.mean(B)
    num = sum((x - ma) * (y - mb) for x, y in zip(A, B))
    den = math.sqrt(sum((x - ma) ** 2 for x in A) * sum((y - mb) ** 2 for y in B))
    return num / den if den else float("nan")


def main() -> None:
    with (RES / "asd_esmif.tsv").open(encoding="utf-8") as fh:
        esm = {r["entry_id"]: r for r in csv.DictReader(fh, delimiter="\t")
               if r["status"] == "ok"}
    with (RES / "asd_scores.tsv").open(encoding="utf-8") as fh:
        base = [r for r in csv.DictReader(fh, delimiter="\t") if r["status"] == "ok"]
    with MANIFEST.open(encoding="utf-8") as fh:
        man = {r["entry_id"]: r for r in csv.DictReader(fh, delimiter="\t")}

    rows = []
    for r in base:
        e = esm.get(r["entry_id"])
        if not e:
            continue
        rows.append({**r, **{k: e[k] for k in
                             ("ll_heavy", "ll_light", "ll_antibody_mean", "ll_antigen")}})

    by = defaultdict(list)
    for r in rows:
        by[r["target_pdb"]].append(r)

    rep = {"n": len(rows), "targets": {}, "pooled": {},
           "abbibench_baselines": ABBIBENCH_BASELINES}

    for t, g in sorted(by.items()):
        y = [float(r["neg_log_kd"]) for r in g]
        per = {}
        for col, lab, ctl in METRICS:
            try:
                per[lab] = round(spearman(y, [float(r[col]) for r in g]), 4)
            except (ValueError, KeyError):
                per[lab] = None
        rep["targets"][t] = {
            "n": len(g),
            "neg_log_kd_range": [round(min(y), 2), round(max(y), 2)],
            "spearman": per,
        }

    # Pooled: z-score within each target before combining, so a target with a
    # different absolute metric scale cannot dominate.
    for col, lab, ctl in METRICS:
        zs, ys = [], []
        for t, g in by.items():
            try:
                v = [float(r[col]) for r in g]
            except (ValueError, KeyError):
                continue
            y = [float(r["neg_log_kd"]) for r in g]
            m, s = st.mean(v), st.pstdev(v) or 1
            my, sy = st.mean(y), st.pstdev(y) or 1
            zs += [(x - m) / s for x in v]
            ys += [(q - my) / sy for q in y]
        rep["pooled"][lab] = {"spearman": round(spearman(ys, zs), 4),
                              "n": len(zs), "null_control": ctl}

    # Diversity diagnostic: tests (and here refutes) "4fqi's variants are too similar"
    div = {}
    for t, g in sorted(by.items()):
        seqs = [man[r["entry_id"]]["heavy_seq"] for r in g if r["entry_id"] in man]
        L = len(seqs[0])
        same_len = [s for s in seqs if len(s) == L]
        varpos = sum(1 for i in range(L) if len({s[i] for s in same_len}) > 1)
        pid = []
        for i in range(0, len(same_len), 7):
            for j in range(i + 1, min(i + 7, len(same_len))):
                pid.append(sum(a == b for a, b in zip(same_len[i], same_len[j])) / L)
        div[t] = {"heavy_len": L, "variable_positions": varpos,
                  "mean_pairwise_identity": round(st.mean(pid), 4) if pid else None}
    rep["heavy_chain_diversity"] = div

    OUT_JSON.write_text(json.dumps(rep, indent=2) + "\n")

    null_pooled = rep["pooled"]["ESM-IF1 ll_antigen"]["spearman"]
    L = ["# EXPERIMENT_03 — within-antigen affinity ranking (generated)", "",
         f"`analyze_asd.py` over {rep['n']} entries. Narrative: "
         f"[EXPERIMENT_03](../EXPERIMENT_03_within_antigen_ranking.md).", "",
         "Each target is **one antigen, many antibodies, heavy chain only varying**. "
         "Affinity is −log KD, higher = tighter, so **every expected correlation is "
         "positive**.", "",
         "## Per target", "",
         "| target | n | " + " | ".join(l for _, l, _ in METRICS) + " |",
         "|---" * (len(METRICS) + 2) + "|"]
    for t, d in rep["targets"].items():
        cells = [f"{d['spearman'][l]:+.3f}" if d["spearman"][l] is not None else "—"
                 for _, l, _ in METRICS]
        L.append(f"| {t} | {d['n']} | " + " | ".join(cells) + " |")
    L += ["", "## Pooled (z-scored within target)", "",
          "| metric | ρ | n |", "|---|---|---|"]
    for lab, d in rep["pooled"].items():
        name = f"*{lab}* **(null control)**" if d["null_control"] else lab
        L.append(f"| {name} | {d['spearman']:+.3f} | {d['n']} |")
    L += ["", f"> **The null control scores {null_pooled:+.3f} pooled.** The antigen is "
          "identical across every entry within a target, so it cannot carry affinity "
          "information. Where it correlates, something target-specific is leaking into "
          "every metric that reads the structure, and those columns cannot be "
          "interpreted at face value.", "",
          "AbBiBench's published baselines on this same data: "
          + ", ".join(f"{k} ≈{v}" for k, v in ABBIBENCH_BASELINES.items()) + ".", "",
          "## Heavy-chain diversity", "",
          "| target | length | variable positions | mean pairwise identity |",
          "|---|---|---|---|"]
    for t, d in div.items():
        L.append(f"| {t} | {d['heavy_len']} | {d['variable_positions']} | "
                 f"{d['mean_pairwise_identity']} |")
    L += ["", "Included to test the hypothesis that a null target's variants are simply "
          "too similar to resolve. It **refutes** it: all three targets are comparably "
          "diverse, and the target with the fewest variable positions shows the "
          "strongest apparent signal.", ""]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_MD}\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
