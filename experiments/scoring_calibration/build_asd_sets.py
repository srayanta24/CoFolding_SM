#!/usr/bin/env python3
"""Build a WITHIN-ANTIGEN affinity benchmark from ASD (EXPERIMENT_03 manifest).

EXPERIMENT_02 concluded that cross-complex affinity ranking is untestable on the
EXPERIMENT_01 benchmark: 78% of log10(Kd) variance there sits *between* antigens, so an
antigen-only null control scored +0.341 — stronger than any real metric. The fix is a
corpus where many antibodies are measured against the *same* antigen, making the
antigen-identity confound impossible by construction.

`databases/asd/` turns out to contain exactly that, and better than expected: its
`dataset == "abbd"` subset is **AbBiBench** itself (155,853 rows, confirmed by row
count and by `source_url` pointing at
huggingface.co/datasets/AbBibench/Antibody_Binding_Benchmark_Dataset). That is the same
benchmark IMPROVE_DESIGN.md §4 cites for "inverse-folding models are the best available
affinity predictors," so our numbers become directly comparable to its published
baselines (ProteinMPNN ρ≈0.30, ESM-IF1 ≈0.28) rather than floating free.

Why this is the right shape where EXPERIMENT_01's benchmark was not
-------------------------------------------------------------------
- **One antigen per target group.** Ranking happens within an antigen, so antigen
  identity cannot confound it. This is the specific defect EXPERIMENT_02 diagnosed.
- **Only the heavy chain varies**; every target has exactly one light chain. The
  comparison is therefore controlled to the CDR-bearing chain.
- **Affinity is `-log KD`, higher = tighter.** Unambiguous direction, unlike the
  `alphaseq` scale also present in ASD.
- **It matches the task a design campaign actually poses** (rank many candidates
  against one target), which EXPERIMENT_02 established is the only comparison worth
  measuring.

Target selection
----------------
Applies the same scope rules as `build_sets.py` so results stay comparable:
protein antigens only (>= MIN_ANTIGEN_RESIDUES, which excludes AbBiBench's two
14-residue peptide targets `aayl49`/`aayl51`), Fab format, and a total-size cap.

Within each target, antibodies are sampled **stratified across affinity deciles** rather
than uniformly: a uniform draw would concentrate in the dense middle of the distribution
and leave the tails — the part that actually tests ranking — underpopulated.

Sampling is deterministic (SHA-256 of the entry id, same convention as
`databases/src/build_splits.py`), so a rebuild reproduces the identical set.

Leakage note, stated because it matters for interpretation
-----------------------------------------------------------
These are *natural* antibody variants against well-studied antigens, several with
deposited structures (1mlc lysozyme, 2fjg VEGF, 1n8z HER2, 4fqi/3gbn influenza HA).
They are almost certainly inside Boltz-2's training window. This benchmark therefore
measures **ranking ability**, not generalisation to novel targets, and a strong result
here would not by itself transfer to de novo designs. EXPERIMENT_01's clean/contaminated
split exists for the generalisation question; this one is deliberately a different
instrument.

Usage:
    .venvs/epitope-prediction/bin/python3 \\
        experiments/scoring_calibration/build_asd_sets.py --per-target 150
"""

import argparse
import ast
import csv
import glob
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ASD = REPO / "databases" / "asd"
OUT_DIR = HERE / "sets"

# Same guards as build_sets.py, so the two benchmarks stay comparable.
MAX_TOTAL_RESIDUES = 1200
MIN_ANTIGEN_RESIDUES = 25

# Affinity-plausibility gate, added 2026-10-20 after a real failure.
#
# ASD labels every abbd row `affinity_type == "-log KD"`, but that is a blanket label
# over heterogeneous source datasets and it is WRONG for some targets. Checked across
# the full subset: 2fjg spans -3.01..0.50 with 46% negative values and a median of
# 0.03, and 1mlc spans -4.10..10.18 with 8% negative. A -log KD of -3 implies
# Kd = 10^3 M, which is not a binding constant -- those columns are almost certainly
# a ddG-like or log-fold-change quantity centred on the wild type, where the SIGN
# CONVENTION IS INVERTED relative to -log KD.
#
# This mattered: the first build sorted candidates cheapest-first and took three, and
# the three cheapest targets are exactly the mislabelled ones. Selecting on cost
# without validating the label silently produced a benchmark whose ground truth
# pointed the wrong way for two of three targets. Any target whose values cannot be
# -log KD is now rejected rather than ranked.
MIN_PLAUSIBLE_NEG_LOG_KD = 0.0   # Kd <= 1 M; anything negative is not a -log KD
MAX_PLAUSIBLE_NEG_LOG_KD = 14.0  # Kd >= 10 fM; tighter than any measured antibody

FIELDS = ["entry_id", "set", "source", "target_pdb", "neg_log_kd", "affinity_decile",
          "n_res_heavy", "n_res_light", "n_res_antigen", "n_res_total",
          "heavy_seq", "light_seq", "antigen_seq", "note"]


def rank_key(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def load_abbd() -> list[dict]:
    import pyarrow.parquet as pq
    import pyarrow as pa
    files = sorted(glob.glob(str(ASD / "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files under {ASD}")
    d = pa.concat_tables([pq.read_table(f) for f in files]).to_pydict()

    def num(x):
        try:
            v = float(x)
            return v if v == v else None  # reject NaN
        except (TypeError, ValueError):
            return None

    def meta(m):
        try:
            return ast.literal_eval(m) if isinstance(m, str) else (m or {})
        except Exception:
            return {}

    out = []
    for ag, aff, ds, h, l, md in zip(d["antigen_sequence"], d["affinity"],
                                     d["dataset"], d["heavy_sequence"],
                                     d["light_sequence"], d["metadata"]):
        if ds != "abbd":
            continue
        a = num(aff)
        if a is None or not (ag and h and l):
            continue
        out.append({"antigen": ag.strip(), "heavy": h.strip(), "light": l.strip(),
                    "neg_log_kd": a, "target_pdb": meta(md).get("target_pdb", "")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-target", type=int, default=150,
                    help="antibodies sampled per target (default 150)")
    ap.add_argument("--max-targets", type=int, default=3,
                    help="how many targets to include (default 3)")
    ap.add_argument("--deepen", default=None,
                    help="comma-separated target:n overrides, e.g. '3gbn:300,4fqi:300'. "
                         "Deterministic hash ordering means a larger n is a SUPERSET of "
                         "a smaller one, so already-scored entries are reused, not "
                         "resampled.")
    args = ap.parse_args()

    rows = load_abbd()
    by_t = defaultdict(list)
    for r in rows:
        by_t[r["target_pdb"]].append(r)

    # Characterise every target, then pick within the scope rules.
    cand, rejected = [], []
    for pdb, v in by_t.items():
        agl = len(v[0]["antigen"])
        size = agl + len(v[0]["heavy"]) + len(v[0]["light"])
        aff = [r["neg_log_kd"] for r in v]
        info = {"target_pdb": pdb, "n": len(v), "antigen_res": agl,
                "approx_total_res": size, "spread": round(max(aff) - min(aff), 2),
                "min": round(min(aff), 2), "max": round(max(aff), 2)}
        n_neg = sum(1 for x in aff if x < MIN_PLAUSIBLE_NEG_LOG_KD)
        info["frac_implausible"] = round(n_neg / len(aff), 3)
        if agl < MIN_ANTIGEN_RESIDUES:
            info["rejected"] = f"antigen < {MIN_ANTIGEN_RESIDUES} res (peptide target)"
            rejected.append(info)
        elif size > MAX_TOTAL_RESIDUES:
            info["rejected"] = f"complex > {MAX_TOTAL_RESIDUES} res (cost guard)"
            rejected.append(info)
        elif n_neg or max(aff) > MAX_PLAUSIBLE_NEG_LOG_KD:
            info["rejected"] = (
                f"affinity not plausible as -log KD ({info['min']}..{info['max']}, "
                f"{info['frac_implausible']:.0%} negative) - label is unreliable, "
                "sign convention unknown")
            rejected.append(info)
        else:
            cand.append(info)

    # Among targets that survive the plausibility gate, prefer the ones with the most
    # antibodies (deeper sampling across the affinity range) rather than the cheapest.
    # Cost-first selection is what produced the mislabelled first build; a target that
    # is cheap but whose ground truth points the wrong way is worth nothing.
    cand.sort(key=lambda c: -c["n"])
    chosen = cand[: args.max_targets]

    deepen = {}
    if args.deepen:
        for part in args.deepen.split(","):
            k, v = part.split(":")
            deepen[k.strip()] = int(v)

    entries = []
    for c in chosen:
        per_target = deepen.get(c["target_pdb"], args.per_target)
        # Aggregate repeated measurements of the SAME heavy chain before sampling.
        # A heavy can appear many times (4fqi has 130,629 rows over 65,536 distinct
        # heavies), and treating replicates as independent both inflates n and, once
        # deduplicated after stratifying, silently shrinks the sample. Median is used
        # rather than mean so a single outlying replicate cannot move the label.
        agg = defaultdict(list)
        for r in by_t[c["target_pdb"]]:
            agg[r["heavy"]].append(r)
        pool = []
        for h, g in agg.items():
            vals = sorted(x["neg_log_kd"] for x in g)
            med = vals[len(vals) // 2] if len(vals) % 2 else \
                (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
            r0 = dict(g[0]); r0["neg_log_kd"] = med; r0["n_replicates"] = len(g)
            pool.append(r0)
        pool_sorted = sorted(pool, key=lambda r: r["neg_log_kd"])
        n = len(pool_sorted)
        per_decile = max(1, per_target // 10)
        for dec in range(10):
            lo, hi = dec * n // 10, (dec + 1) * n // 10
            band = pool_sorted[lo:hi]
            if not band:
                continue
            band = sorted(band, key=lambda r: rank_key(
                f"{c['target_pdb']}|{r['heavy']}"))[:per_decile]
            for r in band:
                total = len(r["heavy"]) + len(r["light"]) + len(r["antigen"])
                if total > MAX_TOTAL_RESIDUES:
                    continue
                eid = f"asd__{c['target_pdb']}__{rank_key(r['heavy'])[:10]}"
                entries.append({
                    "entry_id": eid, "set": f"asd_{c['target_pdb']}",
                    "source": "asd/abbd (AbBiBench)", "target_pdb": c["target_pdb"],
                    "neg_log_kd": f"{r['neg_log_kd']:.4f}", "affinity_decile": dec,
                    "n_res_heavy": len(r["heavy"]), "n_res_light": len(r["light"]),
                    "n_res_antigen": len(r["antigen"]), "n_res_total": total,
                    "heavy_seq": r["heavy"], "light_seq": r["light"],
                    "antigen_seq": r["antigen"],
                    "note": f"higher neg_log_kd = tighter binder; "
                            f"{r.get('n_replicates',1)} replicate(s) aggregated",
                })

    # dedupe (the same heavy can recur across deciles only if affinities tie)
    seen, uniq = set(), []
    for e in entries:
        if e["entry_id"] in seen:
            continue
        seen.add(e["entry_id"])
        uniq.append(e)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_tsv = OUT_DIR / "asd_manifest.tsv"
    with out_tsv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t",
                           extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(uniq, key=lambda e: (e["set"], e["entry_id"])))

    summary = {
        "total_abbd_rows": len(rows),
        "targets_found": len(by_t),
        "targets_in_scope": cand,
        "targets_rejected": rejected,
        "targets_chosen": [c["target_pdb"] for c in chosen],
        "entries": len(uniq),
        "entries_by_target": dict(sorted(
            {c["target_pdb"]: sum(1 for e in uniq if e["target_pdb"] == c["target_pdb"])
             for c in chosen}.items())),
        "total_residues": sum(int(e["n_res_total"]) for e in uniq),
        "scope": {"max_total_residues": MAX_TOTAL_RESIDUES,
                  "min_antigen_residues": MIN_ANTIGEN_RESIDUES},
    }
    (OUT_DIR / "asd_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_tsv}\nwrote {OUT_DIR/'asd_summary.json'}")


if __name__ == "__main__":
    main()
