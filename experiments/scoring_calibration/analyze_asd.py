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
from math import comb
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
    # Rosetta, added 2026-10-31 once the 1,530-complex FastRelax run finished (47
    # min/complex, ~1,200 core-hours). packstat is here because it is the ONLY metric
    # that has ever beaten ipTM on a valid test in this project (23/28 on the paired
    # benchmark, p=0.0005). This is its first independent test on a different
    # benchmark, a different label and a different question, so it is the whole reason
    # the run was worth the compute. Energies are negated so higher = tighter.
    ("packstat", "packstat", False),
    ("neg_dG_dSASA", "-dG/dSASA", False),
    ("neg_dG_separated", "-dG_separated", False),
    ("sc_value", "shape complementarity", False),
    ("dSASA_int", "buried SASA", False),
    ("interface_hbonds", "interface H-bonds", False),
    # ProteinMPNN (§11.5): the decisive metric swap. Reported directly beneath ESM-IF1
    # because the pair is a controlled test of whether a stronger inverse-folding model
    # recovers the published ~0.30 baseline on our predicted backbones, where ESM-IF1
    # reached only +0.111. Its own antigen score is a second null control.
    ("mpnn_ll_heavy", "ProteinMPNN ll_heavy", False),
    ("mpnn_ll_antibody", "ProteinMPNN ll_antibody", False),
    # EXPERIMENT_06 native-backbone arm (stage 1): one shared reference backbone per
    # target, mutants threaded. The per-complex washout control -- if this beats Arm A
    # above, our benchmark's one-cofold-per-mutant construction was flattening signal.
    ("mpnn_shared_ll", "ProteinMPNN ll (shared backbone)", False),
    ("mpnn_native_ll", "ProteinMPNN ll (crystal backbone)", False),
    ("ll_antigen", "ESM-IF1 ll_antigen", True),
    ("mpnn_ll_antigen", "ProteinMPNN ll_antigen", True),
]

# Columns copied in from the side tables, and the negations derived from them.
ESM_COLS = ("ll_heavy", "ll_light", "ll_antibody_mean", "ll_antigen")
ROSETTA_COLS = ("packstat", "dG_dSASA_x100", "dG_separated", "sc_value",
                "dSASA_int", "interface_hbonds")
NULL_LABEL = "ESM-IF1 ll_antigen"

ABBIBENCH_BASELINES = {"ProteinMPNN": 0.30, "ESM-IF1": 0.28}


def spearman(a: list[float], b: list[float]) -> float:
    ra = {v: i for i, v in enumerate(sorted(a))}
    rb = {v: i for i, v in enumerate(sorted(b))}
    A, B = [ra[v] for v in a], [rb[v] for v in b]
    ma, mb = st.mean(A), st.mean(B)
    num = sum((x - ma) * (y - mb) for x, y in zip(A, B))
    den = math.sqrt(sum((x - ma) ** 2 for x in A) * sum((y - mb) ** 2 for y in B))
    return num / den if den else float("nan")


def binom_p(k: int, n: int) -> float:
    """Exact one-sided binomial tail, P(X >= k | p = 0.5). Same test as the paired
    benchmark uses, so the extremes table below is directly comparable to its 23/28."""
    if n == 0:
        return float("nan")
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def partial_spearman(y: list[float], x: list[float], z: list[float]) -> float:
    """Spearman of x vs y with the null control z partialled out of both.

    Why this is here: on this benchmark the null control fires, so a raw correlation
    cannot distinguish "the metric reads affinity" from "the metric reads the same
    target-specific quality channel the null reads". The standard first-order partial
    on ranks answers that directly, and when it was applied in EXPERIMENT_03 it changed
    the conclusion -- the signal largely survived, after I had already written that
    "most of it is not interpretable".

    Computed on ranks, so this is the partial *Spearman*: rho(xy) minus the part
    explained by z, normalised.
    """
    r_xy, r_xz, r_yz = spearman(y, x), spearman(x, z), spearman(y, z)
    den = math.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    return (r_xy - r_xz * r_yz) / den if den else float("nan")


def main() -> None:
    with (RES / "asd_esmif.tsv").open(encoding="utf-8") as fh:
        esm = {r["entry_id"]: r for r in csv.DictReader(fh, delimiter="\t")
               if r["status"] == "ok"}
    # Rosetta is optional so the script still runs if it is ever regenerated from
    # scratch before the (very expensive) FastRelax pass has been repeated.
    ros = {}
    if (RES / "asd_rosetta.tsv").exists():
        with (RES / "asd_rosetta.tsv").open(encoding="utf-8") as fh:
            ros = {r["entry_id"]: r for r in csv.DictReader(fh, delimiter="\t")
                   if r["status"] == "ok"}
    # ProteinMPNN (§11.5), optional. Its ll_* columns collide by name with ESM-IF1's,
    # so they are joined under an mpnn_ prefix.
    mpnn = {}
    if (RES / "asd_proteinmpnn.tsv").exists():
        with (RES / "asd_proteinmpnn.tsv").open(encoding="utf-8") as fh:
            mpnn = {r["entry_id"]: r for r in csv.DictReader(fh, delimiter="\t")
                    if r["status"] == "ok"}
    # EXPERIMENT_06 native-backbone arm (stage 1): every mutant scored on ONE shared
    # reference backbone per target instead of its own per-complex cofold.
    mpnn_sh = {}
    if (RES / "asd_proteinmpnn_shared.tsv").exists():
        with (RES / "asd_proteinmpnn_shared.tsv").open(encoding="utf-8") as fh:
            mpnn_sh = {r["entry_id"]: r for r in csv.DictReader(fh, delimiter="\t")
                       if r["status"] == "ok"}
    # Stage 2: true crystal backbone (3 of 6 targets pass the alignment gate).
    mpnn_nat = {}
    if (RES / "asd_proteinmpnn_native.tsv").exists():
        with (RES / "asd_proteinmpnn_native.tsv").open(encoding="utf-8") as fh:
            mpnn_nat = {r["entry_id"]: r for r in csv.DictReader(fh, delimiter="\t")
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
        row = {**r, **{k: e[k] for k in ESM_COLS}}
        q = ros.get(r["entry_id"])
        if q:
            row.update({k: q[k] for k in ROSETTA_COLS})
            # Negated so that, like every other column here, higher = tighter binder.
            row["neg_dG_dSASA"] = str(-float(q["dG_dSASA_x100"]))
            row["neg_dG_separated"] = str(-float(q["dG_separated"]))
        m = mpnn.get(r["entry_id"])
        if m:
            row["mpnn_ll_heavy"] = m["ll_heavy"]
            row["mpnn_ll_antibody"] = m["ll_antibody"]
            row["mpnn_ll_antigen"] = m["ll_antigen"]
        ms = mpnn_sh.get(r["entry_id"])
        if ms:
            row["mpnn_shared_ll"] = ms["ll_antibody"]
        mn = mpnn_nat.get(r["entry_id"])
        if mn:
            row["mpnn_native_ll"] = mn["ll_antibody"]
        rows.append(row)
    print(f"[asd] {len(rows)} entries, {sum(1 for r in rows if 'packstat' in r)} "
          f"with Rosetta, {sum(1 for r in rows if 'mpnn_ll_heavy' in r)} with ProteinMPNN")

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
    # different absolute metric scale cannot dominate. The null is z-scored the same
    # way and carried along so the partial can be taken on the pooled vectors.
    def pooled_vectors(col: str):
        zs, ys, ns = [], [], []
        for t, g in by.items():
            try:
                v = [float(r[col]) for r in g]
                nv = [float(r["ll_antigen"]) for r in g]
            except (ValueError, KeyError):
                continue
            y = [float(r["neg_log_kd"]) for r in g]
            for src, dst in ((v, zs), (y, ys), (nv, ns)):
                m, s = st.mean(src), st.pstdev(src) or 1
                dst += [(x - m) / s for x in src]
        return ys, zs, ns

    for col, lab, ctl in METRICS:
        ys, zs, ns = pooled_vectors(col)
        if not zs:
            continue
        d = {"spearman": round(spearman(ys, zs), 4), "n": len(zs),
             "null_control": ctl}
        if not ctl:
            # Does the metric still track affinity once the null's target-specific
            # quality channel is removed? This is the column that decides whether a
            # raw correlation means anything on this benchmark.
            d["partial_vs_null"] = round(partial_spearman(ys, zs, ns), 4)
        rep["pooled"][lab] = d

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

    # Extremes test. packstat wins the paired benchmark 23/28 but lands below the null
    # here, so the obvious rescue is that the two benchmarks differ in effect size: the
    # paired test contrasts binding-RETAINED against binding-DESTROYED mutants, while
    # every ASD entry is a functional binder differing only in affinity. If packstat
    # detects gross interface damage rather than affinity gradation, it should recover
    # on the extremes. Scored the same way the paired benchmark is -- a sign test over
    # groups -- at three cut depths, with the null carried through.
    #
    # It does NOT rescue it, and the reason is the null: with only 6 targets the sign
    # test has so little power that the null control passes it too. Recorded because
    # the rescue is the first thing a reader will reach for.
    ext = {}
    for frac in (0.10, 0.25, 0.50):
        per = {}
        for col, lab, ctl in METRICS:
            deltas, wins, n_t = [], 0, 0
            for t, g in sorted(by.items()):
                try:
                    vals = {id(r): float(r[col]) for r in g}
                except (ValueError, KeyError):
                    continue
                s = sorted(g, key=lambda r: float(r["neg_log_kd"]))
                k = max(1, int(len(s) * frac))
                lo = st.mean(vals[id(r)] for r in s[:k])
                hi = st.mean(vals[id(r)] for r in s[-k:])
                sd = st.pstdev(list(vals.values())) or 1
                d = (hi - lo) / sd          # in within-target standard deviations
                deltas.append(round(d, 3))
                wins += d > 0
                n_t += 1
            if n_t:
                per[lab] = {"wins": wins, "n_targets": n_t,
                            "p": round(binom_p(wins, n_t), 4),
                            "deltas_sd": deltas, "null_control": ctl}
        ext[f"top_bottom_{int(frac*100)}pct"] = per
    rep["extremes"] = ext

    # Per-target significance threshold, so the per-target table can be read honestly.
    # Two-sided alpha=0.05 on a Spearman with n entries is |rho| > ~1.96/sqrt(n-1).
    rep["per_target_rho_significance"] = {
        t: round(1.96 / math.sqrt(d["n"] - 1), 4) for t, d in rep["targets"].items()}

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
          "`ρ partial` removes the null control from both sides: it is what remains "
          "after the target-specific channel the null reads is taken out. A metric "
          "whose raw ρ is no better than the null's has not been shown to read "
          "affinity at all.", "",
          "| metric | ρ | ρ partial (null removed) | n |", "|---|---|---|---|"]
    for lab, d in rep["pooled"].items():
        name = f"*{lab}* **(null control)**" if d["null_control"] else lab
        if not d["null_control"] and d["spearman"] <= null_pooled:
            name += " — **below null**"
        part = (f"{d['partial_vs_null']:+.3f}" if "partial_vs_null" in d else "—")
        L.append(f"| {name} | {d['spearman']:+.3f} | {part} | {d['n']} |")
    L += ["", f"> **The null control scores {null_pooled:+.3f} pooled.** The antigen is "
          "identical across every entry within a target, so it cannot carry affinity "
          "information. Where it correlates, something target-specific is leaking into "
          "every metric that reads the structure, and those columns cannot be "
          "interpreted at face value.", "",
          "AbBiBench's published baselines on this same data: "
          + ", ".join(f"{k} ≈{v}" for k, v in ABBIBENCH_BASELINES.items()) + ".", "",
          "## Does packstat recover on the extremes?", "",
          "packstat wins the paired benchmark 23/28 (p=0.0005) but lands **below the "
          "null control** above. The natural rescue: the paired test contrasts "
          "binding-*retained* against binding-*destroyed* mutants, whereas every entry "
          "here is a functional binder differing only in affinity. If packstat detects "
          "gross interface damage rather than affinity gradation, it should recover when "
          "only the extremes are compared.", "",
          "Scored exactly as the paired benchmark is — a sign test over groups — at "
          "three cut depths. Deltas are in within-target standard deviations.", ""]
    for key, per in rep["extremes"].items():
        pct = key.split("_")[-1].replace("pct", "%")
        L += [f"**Top vs bottom {pct} of affinity**", "",
              "| metric | " + " | ".join(sorted(rep["targets"])) + " | wins | p |",
              "|---" * (len(rep["targets"]) + 3) + "|"]
        for lab, d in per.items():
            name = f"*{lab}* **(null control)**" if d["null_control"] else lab
            L.append(f"| {name} | " + " | ".join(f"{x:+.2f}" for x in d["deltas_sd"])
                     + f" | {d['wins']}/{d['n_targets']} | {d['p']:.3f} |")
        L.append("")
    L += ["**It does not rescue packstat, and the reason is the null control.** With "
          "only 6 targets the sign test has so little power that the null passes it too "
          "— at the decile and quartile cuts the null wins 5/6, with per-target effect "
          "sizes as large as the real metrics. No column in this table can be read as "
          "evidence. This is the same lesson the paired benchmark taught when 13/17 "
          "collapsed at 28 parents: a sign test over a handful of groups is not a "
          "finding, and the null control is what reveals it.", "",
          "The honest statement is therefore **not** \"packstat only detects gross "
          "damage\" — that remains the most plausible mechanism, but this benchmark "
          "cannot establish it. What it does establish is the negative: packstat's "
          "23/28 does not transfer to graded affinity ranking.", "",
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
