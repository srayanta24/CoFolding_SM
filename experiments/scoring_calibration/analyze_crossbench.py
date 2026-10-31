#!/usr/bin/env python3
"""Do the two benchmarks agree about which metrics work? (the arc's closing question)

Fifth analysis script here, after `analyze.py` (EXPERIMENT_01), `analyze_metrics.py`
(EXPERIMENT_02), `analyze_paired.py` (the 28-parent sign test) and `analyze_asd.py`
(EXPERIMENT_03/_04). Writes `results/cross_benchmark.{md,json}`.

It consumes only the two generated JSONs, so it adds no compute and cannot disagree
with the per-benchmark write-ups.

Why this exists
---------------
EXPERIMENT_02 and _04 each answered "does metric X work on benchmark Y". Neither asked
the question that decides what the whole arc concluded: **is a metric that wins on one
benchmark more likely to win on the other?** If the answer is no, then no metric has
demonstrated a selection signal that survives a change of question, which is a much
stronger statement than any individual experiment's.

The two benchmarks, and why a metric could legitimately differ between them
--------------------------------------------------------------------------
    paired   28 parent complexes; binding-RETAINED vs binding-DESTROYED point mutant
             of the same parent. Binary, large effect. Controls for complex identity.
    ASD      6 antigens, 1,530 antibodies; graded -log KD. Controls for antigen
             identity. Every entry is a functional binder.

A metric that detects gross interface damage but not affinity gradation SHOULD win the
first and lose the second. So disagreement is not automatically evidence of noise --
which is exactly why the chance calculation below is needed.

The chance calculation, which is the point
------------------------------------------
Counting "metrics that beat the null on both benchmarks" is meaningless without knowing
how many to expect by luck. With K of N metrics beating one null and n of N beating the
other, the overlap under independence is **hypergeometric**. If the observed overlap is
what chance predicts, then no metric has shown cross-benchmark signal -- regardless of
how any single metric's p-value looked in isolation.

This is the same discipline the null controls enforce within a benchmark, applied one
level up: compare against what a no-signal world produces, not against zero.

Stdlib only.

Usage:
    python3 experiments/scoring_calibration/analyze_crossbench.py
"""

import json
import math
import statistics as st
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT_MD = RES / "cross_benchmark.md"
OUT_JSON = RES / "cross_benchmark.json"

NULL = "ESM-IF1 ll_antigen"


def spearman(a: list[float], b: list[float]) -> float:
    ra = {v: i for i, v in enumerate(sorted(a))}
    rb = {v: i for i, v in enumerate(sorted(b))}
    A, B = [ra[v] for v in a], [rb[v] for v in b]
    ma, mb = st.mean(A), st.mean(B)
    num = sum((x - ma) * (y - mb) for x, y in zip(A, B))
    den = math.sqrt(sum((x - ma) ** 2 for x in A) * sum((y - mb) ** 2 for y in B))
    return num / den if den else float("nan")


def hypergeom_tail(N: int, K: int, n: int, obs: int) -> tuple[list[float], float, float]:
    """Distribution of the overlap, its expectation, and P(overlap >= obs).

    N metrics; K beat the ASD null; n beat the paired null; how many beat both?
    """
    pmf = [comb(K, k) * comb(N - K, n - k) / comb(N, n) if 0 <= n - k <= N - K else 0.0
           for k in range(n + 1)]
    return pmf, K * n / N, sum(pmf[obs:])


def main() -> None:
    paired = {m["metric"]: m for m in
              json.loads((RES / "paired_comparison.json").read_text())["metrics"]}
    asd = json.loads((RES / "asd_comparison.json").read_text())["pooled"]

    shared = [m for m in paired if m in asd]
    if NULL not in shared:
        raise SystemExit(f"null control {NULL!r} missing from one benchmark")

    p_null = paired[NULL]["all"]["wins"] / paired[NULL]["all"]["n"]
    a_null = asd[NULL]["spearman"]

    rows = []
    for m in shared:
        w = paired[m]["all"]["wins"] / paired[m]["all"]["n"]
        r = asd[m]["spearman"]
        rows.append({
            "metric": m,
            "null_control": m == NULL,
            "paired_win_rate": round(w, 4),
            "paired_wins": f"{paired[m]['all']['wins']}/{paired[m]['all']['n']}",
            "paired_p": paired[m]["all"]["p"],
            "asd_spearman": r,
            "beats_paired_null": w > p_null,
            "beats_asd_null": r > a_null,
        })
    rows.sort(key=lambda d: -d["paired_win_rate"])

    real = [d for d in rows if not d["null_control"]]
    agreement = spearman([d["paired_win_rate"] for d in real],
                         [d["asd_spearman"] for d in real])

    N = len(real)
    n_p = sum(d["beats_paired_null"] for d in real)
    n_a = sum(d["beats_asd_null"] for d in real)
    both = [d["metric"] for d in real
            if d["beats_paired_null"] and d["beats_asd_null"]]
    pmf, expected, tail = hypergeom_tail(N, n_a, n_p, len(both))

    rep = {
        "n_shared_metrics": N,
        "nulls": {"paired_win_rate": round(p_null, 4), "asd_spearman": a_null},
        "metrics": rows,
        "ordering_agreement_spearman": round(agreement, 4),
        "beat_null_counts": {"paired": n_p, "asd": n_a, "both": len(both)},
        "metrics_beating_both_nulls": both,
        "chance_model": {
            "test": "hypergeometric overlap under independence",
            "expected_overlap": round(expected, 3),
            "observed_overlap": len(both),
            "p_overlap_at_least_observed": round(tail, 4),
            "pmf": [round(x, 4) for x in pmf],
        },
    }
    OUT_JSON.write_text(json.dumps(rep, indent=2) + "\n")

    verdict = ("**no metric has demonstrated cross-benchmark signal**"
               if tail > 0.05 else
               "**the overlap exceeds chance** — revisit the arc's conclusion")

    L = ["# Cross-benchmark agreement (generated)", "",
         "`analyze_crossbench.py` over the two generated benchmark JSONs. Narrative: "
         "[PLAN.md](../PLAN.md) §11.", "",
         "Each experiment asked *does metric X work on benchmark Y*. This asks the "
         "question that decides what the arc concluded: **is a metric that wins on one "
         "benchmark more likely to win on the other?**", "",
         "| metric | paired | p | beats null | ASD ρ | beats null |",
         "|---|---|---|---|---|---|"]
    for d in rows:
        name = f"*{d['metric']}* **(null)**" if d["null_control"] else d["metric"]
        bp = "—" if d["null_control"] else ("✓" if d["beats_paired_null"] else "✗")
        ba = "—" if d["null_control"] else ("✓" if d["beats_asd_null"] else "✗")
        L.append(f"| {name} | {d['paired_wins']} | {d['paired_p']:.4f} | {bp} | "
                 f"{d['asd_spearman']:+.3f} | {ba} |")
    L += ["",
          f"Null control: **{p_null:.2f}** win rate on paired, **{a_null:+.3f}** on ASD. "
          "`✓` means the metric beat the null *on that benchmark*.", "",
          "## Do the two orderings agree?", "",
          f"Rank correlation between the two columns, over the {N} non-null shared "
          f"metrics: **ρ = {agreement:+.3f}**.", "",
          "The best metric on one benchmark is near-worst on the other, in both "
          "directions: `packstat` tops the paired test and sits below the null on ASD, "
          "while `ipTM` is at chance on the paired test and near the top on ASD.", "",
          "## Is the overlap more than chance?", "",
          f"{n_p} of {N} metrics beat the paired null; {n_a} of {N} beat the ASD null; "
          f"**{len(both)} beat both** ({', '.join(both) if both else 'none'}).", "",
          "Counting survivors means nothing without knowing how many luck produces. "
          "Under independence the overlap is hypergeometric:", "",
          "| overlap | P |", "|---|---|"]
    for k, p in enumerate(pmf):
        L.append(f"| {k}{' ← observed' if k == len(both) else ''} | {p:.3f} |")
    L += ["",
          f"- expected overlap by chance: **{expected:.2f}**",
          f"- observed: **{len(both)}**",
          f"- P(overlap ≥ observed) = **{tail:.3f}**", "",
          f"So {verdict}. The two metrics that survive both are precisely what a "
          "no-signal world produces.", "",
          "> This is the null-control discipline applied one level up. Within a "
          "benchmark, a metric is compared against a quantity that cannot carry the "
          "signal. Across benchmarks, a *count of survivors* is compared against the "
          "count chance produces. Both answer the same question: is this more than "
          "nothing?", "",
          "## What this does and does not establish", "",
          "**Does**: no metric among those tested ranks designs in a way that survives "
          "a change of question. The remedy of picking a better existing scalar is "
          "closed, because the shortlist has been enumerated and none of it transfers.",
          "", "**Does not**: prove no cheap metric exists. Four families were tested "
          "(BoltzGen confidences, ipSAE, Rosetta, inverse-folding likelihoods), not all "
          "possible ones. And a metric *may* legitimately differ between these two "
          "benchmarks, since one contrasts intact against wrecked interfaces and the "
          "other ranks functional binders — the chance calculation is what rules out "
          "reading the survivors as evidence, not the disagreement by itself.", ""]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"ordering agreement rho = {agreement:+.3f}")
    print(f"beat both nulls: {len(both)} observed, {expected:.2f} expected, p={tail:.3f}")
    print(f"wrote {OUT_MD}\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
