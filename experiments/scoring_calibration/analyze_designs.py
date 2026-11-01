#!/usr/bin/env python3
"""Milestone 7 analysis: does packstat work on real designs? (EXPERIMENT_05)

Sixth analysis script here. Writes `results/designs_comparison.{md,json}`.

Pre-registered criteria live in `build_design_sets.py`'s docstring and are restated in
`CRITERIA` below so the verdict is computed, not narrated. They were fixed before any
Rosetta job ran.

The test
--------
Within each target separately (so antigen identity is controlled by construction),
does the metric separate designs whose refold matches the designed backbone
(`filter_rmsd` <= 2.5) from those whose refold does not?

Reported as AUC, which here is the Mann-Whitney U statistic normalised -- the
probability that a random passer outscores a random non-passer. 0.5 is chance.

Why AUC and not a sign test
---------------------------
The paired benchmark used a sign test because it had 28 matched pairs. Here there are
hundreds of unpaired designs per target, and the sampling is stratified by
`filter_rmsd`, so the natural statistic is the two-class separation. AUC is unbiased
under stratified sampling of the *negative* class in a way a rank correlation over the
whole sample is not -- which is why the correlation is reported as secondary and
flagged.

The null control
----------------
`ll_antigen` -- ESM-IF1 scoring chain A. The antigen's SEQUENCE is identical across
every design of a target, but its refolded STRUCTURE is not, so this measures global
refold quality. It is **expected to fire**, and that is its job: it separates "the whole
structure is bad" from "the interface specifically is bad". A metric that does not beat
it has demonstrated nothing interface-specific.

A withheld target
-----------------
One of the two design pools is gitignored along with its campaign, at the user's
instruction. Its numbers are computed locally and deliberately **excluded** from the
committed report; only the direction it showed is stated. `--include-withheld` prints
them to stdout for local inspection without writing them to the tracked files. The
report says plainly that it does this, because a claim resting on data not in the repo
has to be labelled as such.

Stdlib only.

Usage:
    python3 experiments/scoring_calibration/analyze_designs.py
    python3 experiments/scoring_calibration/analyze_designs.py --include-withheld
"""

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SETS = HERE / "sets"
OUT_MD = RES / "designs_comparison.md"
OUT_JSON = RES / "designs_comparison.json"

CRITERIA = {"pass_auc": 0.65, "weak_auc": 0.60}
NULL_LABEL = "ESM-IF1 ll_antigen"

# (label, accessor, is_null). Oriented so higher = predicted better design.
METRICS = [
    ("packstat", lambda r: float(r["packstat"]), False),
    ("shape complementarity", lambda r: float(r["sc_value"]), False),
    ("-dG/dSASA", lambda r: -float(r["dG_dSASA_x100"]), False),
    ("-dG_separated", lambda r: -float(r["dG_separated"]), False),
    ("buried SASA", lambda r: float(r["dSASA_int"]), False),
    ("interface H-bonds", lambda r: float(r["interface_hbonds"]), False),
    ("ipTM (free)", lambda r: float(r["design_to_target_iptm"]), False),
    ("pTM (free)", lambda r: float(r["design_ptm"]), False),
    ("delta_sasa_refolded (free)", lambda r: float(r["delta_sasa_refolded"]), False),
    ("ESM-IF1 ll_heavy", lambda r: float(r["ll_heavy"]), False),
    ("ESM-IF1 ll_antibody", lambda r: float(r["ll_antibody_mean"]), False),
    (NULL_LABEL, lambda r: float(r["ll_antigen"]), True),
]


def auc_mw(pos: list[float], neg: list[float]) -> tuple[float, float]:
    """AUC via Mann-Whitney U, with a tie-corrected normal-approximation two-sided p.

    AUC = P(random positive > random negative), ties counting a half.
    """
    n1, n2 = len(pos), len(neg)
    if not n1 or not n2:
        return float("nan"), float("nan")
    allv = sorted(pos + neg)
    # average ranks for ties
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r = (i + j) / 2 + 1
        ranks[allv[i]] = r
        i = j + 1
    r1 = sum(ranks[v] for v in pos)
    u1 = r1 - n1 * (n1 + 1) / 2
    auc = u1 / (n1 * n2)
    # tie-corrected variance
    from collections import Counter
    tie = Counter(allv)
    n = n1 + n2
    tsum = sum(t ** 3 - t for t in tie.values())
    var = n1 * n2 / 12 * ((n + 1) - tsum / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return auc, float("nan")
    z = (u1 - n1 * n2 / 2) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))
    return auc, p


def spearman(a: list[float], b: list[float]) -> float:
    ra = {v: i for i, v in enumerate(sorted(a))}
    rb = {v: i for i, v in enumerate(sorted(b))}
    A, B = [ra[v] for v in a], [rb[v] for v in b]
    ma, mb = st.mean(A), st.mean(B)
    num = sum((x - ma) * (y - mb) for x, y in zip(A, B))
    den = math.sqrt(sum((x - ma) ** 2 for x in A) * sum((y - mb) ** 2 for y in B))
    return num / den if den else float("nan")


def load(target: str) -> list[dict]:
    """Join the Rosetta and ESM-IF1 outputs for one target, keeping only complete rows."""
    ros_p = RES / f"designs_rosetta_{target}.tsv"
    esm_p = RES / f"designs_esmif_{target}.tsv"
    if not ros_p.exists():
        return []
    with ros_p.open(encoding="utf-8") as fh:
        ros = {r["entry_id"]: r for r in csv.DictReader(fh, delimiter="\t")
               if r.get("status") == "ok"}
    esm = {}
    if esm_p.exists():
        with esm_p.open(encoding="utf-8") as fh:
            esm = {r["entry_id"]: r for r in csv.DictReader(fh, delimiter="\t")
                   if r.get("status") == "ok"}
    rows = []
    for eid, r in ros.items():
        merged = dict(r)
        if eid in esm:
            merged.update({k: esm[eid][k] for k in
                           ("ll_heavy", "ll_light", "ll_antibody_mean", "ll_antigen")})
        rows.append(merged)
    return rows


def analyse(rows: list[dict]) -> dict:
    pos = [r for r in rows if r["label"] == "1"]
    neg = [r for r in rows if r["label"] == "0"]
    out = {"n": len(rows), "n_pass": len(pos), "n_fail": len(neg), "metrics": {}}
    for lab, fn, ctl in METRICS:
        try:
            pv = [fn(r) for r in pos]
            nv = [fn(r) for r in neg]
            allv = [fn(r) for r in rows]
            rm = [-float(r["filter_rmsd"]) for r in rows]
        except (KeyError, ValueError, TypeError):
            continue
        auc, p = auc_mw(pv, nv)
        out["metrics"][lab] = {
            "auc": round(auc, 4), "p": round(p, 6) if p == p else None,
            "null_control": ctl,
            "spearman_vs_neg_rmsd": round(spearman(rm, allv), 4),
            "median_pass": round(st.median(pv), 4),
            "median_fail": round(st.median(nv), 4),
        }
    return out


def _verdict_one(aucs: dict, nulls: dict) -> tuple[str, bool]:
    """PASS/WEAK/FAIL over whatever targets are passed, plus beats-null-everywhere."""
    beats = all(aucs[t] > nulls.get(t, 0.5) for t in aucs) if nulls else None
    lo = min(aucs.values())
    if beats and lo >= CRITERIA["pass_auc"]:
        return "PASS", beats
    if beats and lo >= CRITERIA["weak_auc"]:
        return "WEAK", beats
    return "FAIL", beats


def verdict(per_tracked: dict, per_withheld: dict) -> dict:
    """Leak-free verdicts.

    The committed fields are computed on TRACKED targets only, so every number in them
    is checkable from this repository. The withheld target(s) contribute exactly one
    bit per metric -- `withheld_agreed`, whether the metric landed on the same side of
    chance there -- which is a *direction*, the only thing the user permitted to leave
    the gitignored pool. No withheld AUC, median or min is ever written here; those live
    only in the gitignored TSVs and the `--include-withheld` stdout.

    The stronger, pre-registered two-target verdict (which does use the withheld
    numbers) is reported in the EXPERIMENT_05 prose, backed by the gitignored data.
    """
    if not per_tracked:
        return {}
    names = set.intersection(*(set(d["metrics"]) for d in per_tracked.values()))
    out = {}
    for lab in sorted(names):
        if lab == NULL_LABEL:
            continue
        aucs = {t: d["metrics"][lab]["auc"] for t, d in per_tracked.items()}
        nulls = {t: d["metrics"][NULL_LABEL]["auc"] for t, d in per_tracked.items()
                 if NULL_LABEL in d["metrics"]}
        v, beats = _verdict_one(aucs, nulls)
        tracked_dir_up = all(a > 0.5 for a in aucs.values())
        # Direction-only summary of the withheld target(s): did they agree with the
        # tracked direction? A single boolean, never a magnitude.
        withheld_agreed = None
        if per_withheld:
            wdir = []
            for d in per_withheld.values():
                if lab in d["metrics"]:
                    wdir.append((d["metrics"][lab]["auc"] > 0.5) == tracked_dir_up)
            withheld_agreed = all(wdir) if wdir else None
        out[lab] = {"verdict_tracked": v, "beats_null_on_tracked": beats,
                    "min_auc_tracked": round(min(aucs.values()), 4),
                    "auc_by_tracked_target": {t: round(a, 4) for t, a in aucs.items()},
                    "tracked_direction_above_chance": tracked_dir_up,
                    "withheld_agreed_direction": withheld_agreed}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--include-withheld", action="store_true",
                    help="print withheld targets to stdout (never to the report)")
    args = ap.parse_args()

    # Campaigns come from the gitignored sidecar (real_key, alias, tracked). Everything
    # written to a tracked file is keyed by ALIAS, never real_key; the per-design TSVs
    # loaded here are gitignored. Without the sidecar — a fresh clone — there is nothing
    # to recompute, and the committed comparison stands as the record.
    sidecar = SETS / "campaigns.tsv"
    if not sidecar.exists():
        raise SystemExit(f"[designs] sidecar {sidecar} absent (gitignored); the "
                         "committed designs_comparison.* is the record on a fresh clone.")
    campaigns = []
    for line in sidecar.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            real_key, alias, tracked, *_ = line.split("\t")
            campaigns.append((real_key, alias, tracked.strip().lower() == "yes"))

    real_keys = [c[0] for c in campaigns]   # for the leak guard later
    per_target, per_withheld = {}, {}       # keyed by ALIAS for tracked, real_key hidden
    for real_key, alias, tracked in campaigns:
        rows = load(real_key)               # gitignored designs_rosetta_<real_key>.tsv
        if not rows:
            print(f"[designs] {alias if tracked else 'withheld'}: no results, skipping")
            continue
        if tracked:
            per_target[alias] = analyse(rows)
        else:
            # Withheld: key by a neutral placeholder so no real_key enters any dict.
            per_withheld[f"withheld_{len(per_withheld)+1}"] = analyse(rows)

    if not per_target:
        raise SystemExit("[designs] no tracked-target results to report")

    rep = {"criteria": CRITERIA, "targets": per_target,
           "verdicts": verdict(per_target, per_withheld),
           "n_withheld_targets": len(per_withheld),
           "withheld_note": (
               "One design pool is gitignored with its campaign at the user's request. "
               "Every number in this file is from the tracked target(s). The withheld "
               "target contributes only `withheld_agreed_direction` (one bit per "
               "metric). Its magnitudes live in the gitignored designs_*_<withheld>.tsv "
               "and in `analyze_designs.py --include-withheld`.")}
    # Guard: no withheld target name may appear anywhere in the committed JSON.
    blob = json.dumps(rep)
    leaked = [k for k in real_keys if k in blob]
    if leaked:
        raise SystemExit(f"[designs] REFUSING TO WRITE: real campaign key(s) {leaked} "
                         f"leaked into the committed report")
    OUT_JSON.write_text(json.dumps(rep, indent=2) + "\n")

    L = ["# EXPERIMENT_05 — packstat on real designs (generated)", "",
         f"`analyze_designs.py`. Narrative: "
         f"[EXPERIMENT_05](../EXPERIMENT_05_design_transfer.md).", "",
         "Within each target, does the metric separate designs whose refold matches "
         "the designed backbone (`filter_rmsd` ≤ 2.5) from those whose refold does "
         "not? Reported as AUC — P(random passer outscores a random non-passer). "
         "0.5 is chance.", "",
         f"Pre-registered: **PASS** = AUC ≥ {CRITERIA['pass_auc']} on every target *and* "
         f"above the null on every target; **WEAK** = ≥ {CRITERIA['weak_auc']} on the "
         "same terms; **FAIL** otherwise, including \"works on one target only\".", ""]

    if per_withheld:
        L += [f"> **{len(per_withheld)} further target"
              f"{'s' if len(per_withheld) > 1 else ''} withheld.** One design pool is "
              "gitignored along with its campaign at the user's request, so its "
              "numbers are not in this report. **The verdict column below does "
              "incorporate it**, which means the verdicts rest partly on data that is "
              "not in this repository and cannot be checked from it. Stated here "
              "rather than buried.", ""]

    for t, d in per_target.items():
        L += [f"## {t} — {d['n']} designs ({d['n_pass']} pass / {d['n_fail']} fail)", "",
              "| metric | AUC | p | median pass | median fail | ρ vs −rmsd |",
              "|---|---|---|---|---|---|"]
        for lab, m in sorted(d["metrics"].items(), key=lambda kv: -kv[1]["auc"]):
            name = f"*{lab}* **(null)**" if m["null_control"] else lab
            pp = "—" if m["p"] is None else f"{m['p']:.2e}"
            L.append(f"| {name} | {m['auc']:.3f} | {pp} | {m['median_pass']} | "
                     f"{m['median_fail']} | {m['spearman_vs_neg_rmsd']:+.3f} |")
        L.append("")

    n_tr = len(per_target)
    tr_word = "target" if n_tr == 1 else f"{n_tr} tracked targets"
    wh = "" if not per_withheld else " withheld agrees? |"
    L += ["## Verdicts", "",
          f"Verdict and AUC below are computed on the **{tr_word}** only, so every "
          "number here is checkable from this repository. "
          + ("" if not per_withheld else
             "The withheld target contributes one bit — *did it land on the same side "
             "of chance* — and nothing else; its magnitudes are in the gitignored "
             "`designs_*` TSVs. The stronger pre-registered two-target verdict (which "
             "uses those magnitudes) is in "
             "[EXPERIMENT_05](../EXPERIMENT_05_design_transfer.md) §3.") + "",
          f"| metric | verdict ({'tracked' if per_withheld else 'pre-registered'}) | "
          f"AUC | beats null |{wh}",
          "|---|---|---|---|" + ("---|" if per_withheld else "")]
    # (header cells: metric, verdict, AUC, beats null[, withheld agrees?])
    order = {"PASS": 0, "WEAK": 1, "FAIL": 2}
    for lab, v in sorted(rep["verdicts"].items(),
                         key=lambda kv: (order[kv[1]["verdict_tracked"]],
                                         -kv[1]["min_auc_tracked"])):
        agree = ""
        if per_withheld:
            a = v["withheld_agreed_direction"]
            agree = f" {'yes' if a else 'no' if a is not None else '—'} |"
        L.append(f"| {lab} | **{v['verdict_tracked']}** | {v['min_auc_tracked']:.3f} | "
                 f"{'yes' if v['beats_null_on_tracked'] else 'no'} |{agree}")
    L += ["",
          "`ρ vs −rmsd` is **secondary and biased**: negatives were sampled stratified "
          "by `filter_rmsd`, which distorts its marginal distribution. The AUC column "
          "is the pre-registered test.", "",
          "> `filter_rmsd` is **structural self-consistency, not binding.** Nothing here "
          "shows a metric picks binders — there are no binding labels for designs "
          "anywhere in this project. See [PLAN.md](../PLAN.md) §9.", ""]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_MD}\nwrote {OUT_JSON}")

    if args.include_withheld and per_withheld:
        print("\n--- WITHHELD (stdout only, not written to any tracked file) ---")
        for t, d in per_withheld.items():
            print(f"{t}: n={d['n']} ({d['n_pass']} pass / {d['n_fail']} fail)")
            for lab, m in sorted(d["metrics"].items(), key=lambda kv: -kv[1]["auc"]):
                tag = " [NULL]" if m["null_control"] else ""
                print(f"   {lab + tag:<30} AUC={m['auc']:.3f}")


if __name__ == "__main__":
    main()
