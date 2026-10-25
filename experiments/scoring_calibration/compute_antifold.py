#!/usr/bin/env python3
"""AntiFold sequence log-likelihood over the benchmarks (EXPERIMENT_04, tier 2b).

Why this is the cleanest available test of IMPROVE_DESIGN.md §1
----------------------------------------------------------------
§1 established that nothing in the BoltzGen stack was ever trained on antibodies: the
`antibody-anything` and `nanobody-anything` protocols are byte-identical config dicts,
no checkpoint differs by modality, no training config mentions antibodies, and the
inverse-folding stage is (per the paper) "largely similar to SolubleMPNN" and performs
"similar to ProteinMPNN and SolubleMPNN".

**AntiFold is ESM-IF1 fine-tuned on antibodies** — SAbDab solved structures plus OAS
predicted structures. Same architecture, same inverse-folding task, same scoring API.
We already have ESM-IF1 scored on both benchmarks. So AntiFold vs ESM-IF1 is a
controlled A/B where the *only* material difference is antibody-specific training:

    ESM-IF1 (generic)    paired 13/28 (chance)   ASD pooled rho +0.078
    AntiFold (antibody)  <- this script

If antibody specialisation matters at the inverse-folding stage, it should show up
here. If it does not, that is evidence against the cheapest version of the §1 fix and
should temper expectations for the expensive versions (fine-tuning, a different base
model).

Isolation, and why it is not optional
--------------------------------------
AntiFold pins `torch==2.3.1`, which has **no aarch64 CUDA wheel**. Installing it into
an existing venv silently downgraded torch from the sm_121-capable nightly, took
`cuda.is_available()` from True to False, and orphaned the torch_scatter/sparse/cluster
extensions built against the nightly ABI — exactly the failure DESIGN.md §2 records for
Protenix's identical pin. Recovery was only possible because uv had the unpacked
nightly in its archive cache; the wheel itself had been purged from the index.

So this runs in its own `.venvs/antifold`, CPU-only, and must stay there. Check any
future install into a shared venv with `--dry-run` first.

Scoring
-------
Two conditionings, because the AntiFold paper reports that adding the antigen chain
*improves* AntiFold while it *degrades* ProteinMPNN and AbMPNN — an antigen-awareness
claim worth testing directly rather than assuming:

    ll_antifold      H + L only (AntiFold's default mode)
    ll_antifold_ag   H + L + antigen (custom_chain_mode)

Each is the mean log-probability of the *observed* residue at each designed position,
so it is the same quantity as ESM-IF1's `ll_*` columns and directly comparable.

AntiFold reads `{pdb_dir}/{pdb}.cif`, so the benchmark's scattered per-entry structures
are staged into a flat directory of symlinks named by entry_id. Symlinks, not copies —
the structures are ~4GB.

Usage:
    .venvs/antifold/bin/python3 experiments/scoring_calibration/compute_antifold.py \
        --manifest .../sets/manifest.tsv --results .../results/scores.tsv \
        --out .../results/antifold.tsv --limit 4
"""

import argparse
import csv
import glob
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCORES_DIR = HERE / ".scores"
STAGE = HERE / ".antifold_stage"

CARRY = ["entry_id", "set", "label", "kd_molar", "ddg_kcal", "elisa_ratio", "iptm",
         "target_pdb", "neg_log_kd"]
METRICS = ["ll_antifold", "ll_antifold_ag", "ll_af_heavy", "ll_af_light"]
FIELDS = CARRY + METRICS + ["status", "seconds"]

AA = "ACDEFGHIKLMNPQRSTVWY"


def stage_structures(entry_ids: list[str]) -> dict[str, str]:
    """Symlink each entry's predicted complex into one flat dir AntiFold can read."""
    STAGE.mkdir(parents=True, exist_ok=True)
    ok = {}
    for eid in entry_ids:
        hits = glob.glob(str(SCORES_DIR / eid / "**" / "*_model_0.cif"), recursive=True)
        if not hits:
            continue
        link = STAGE / f"{eid}.cif"
        if not link.exists():
            link.symlink_to(Path(hits[0]).resolve())
        ok[eid] = str(link)
    return ok


def mean_logprob(df_lp, chains: tuple[str, ...]) -> dict:
    """Mean log-prob of the observed residue, per chain and overall.

    df_logits_to_logprobs returns one row per residue with a column per amino acid
    plus `pdb_chain` and the observed residue. We read the observed residue's own
    column, which is the model's log-probability of what is actually there.
    """
    out = {}
    seq_col = next((c for c in ("pdb_res", "aa", "seq") if c in df_lp.columns), None)
    chain_col = next((c for c in ("pdb_chain", "chain") if c in df_lp.columns), None)
    if seq_col is None or chain_col is None:
        raise ValueError(f"unexpected AntiFold columns: {list(df_lp.columns)[:12]}")
    vals_all = []
    for ch in chains:
        sub = df_lp[df_lp[chain_col] == ch]
        vals = [row[row[seq_col]] for _, row in sub.iterrows()
                if row[seq_col] in df_lp.columns]
        if vals:
            out[ch] = sum(vals) / len(vals)
            vals_all += vals
    out["_all"] = sum(vals_all) / len(vals_all) if vals_all else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import pandas as pd
    from antifold.main import load_model, get_pdbs_logits, df_logits_to_logprobs

    with open(args.manifest, encoding="utf-8") as fh:
        man = {r["entry_id"]: r for r in csv.DictReader(fh, delimiter="\t")}
    with open(args.results, encoding="utf-8") as fh:
        scored = [r for r in csv.DictReader(fh, delimiter="\t") if r["status"] == "ok"]

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            if hdr != FIELDS:
                raise SystemExit(
                    f"REFUSING TO APPEND: {out_path} has {len(hdr)} columns, this "
                    f"build writes {len(FIELDS)}. Rewrite it or use a new --out.")
        with out_path.open(encoding="utf-8") as fh:
            done = {r["entry_id"] for r in csv.DictReader(fh, delimiter="\t")}

    todo = [r for r in scored if r["entry_id"] not in done and r["entry_id"] in man]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[antifold] {len(scored)} scored, {len(done)} done, {len(todo)} to do",
          flush=True)
    if not todo:
        return

    staged = stage_structures([r["entry_id"] for r in todo])
    print(f"[antifold] staged {len(staged)} structures into {STAGE}", flush=True)

    print("[antifold] loading model (CPU)", flush=True)
    model = load_model()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()
    t0 = time.time()
    with out_path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t",
                           extrasaction="ignore", lineterminator="\n")
        if new_file:
            w.writeheader()
            fh.flush()
        for i, rec in enumerate(todo, 1):
            eid = rec["entry_id"]
            row = {k: rec.get(k, "") for k in CARRY}
            row.update({k: "" for k in METRICS})
            t = time.time()
            try:
                if eid not in staged:
                    raise FileNotFoundError("no structure")
                # Variant 1: antibody alone (AntiFold's default H/L mode)
                df1 = pd.DataFrame([{"pdb": eid, "Hchain": "B", "Lchain": "C"}])
                lg1 = get_pdbs_logits(model, df1, str(STAGE), save_flag=False)
                # get_pdbs_logits returns a LIST of per-PDB DataFrames (verified, not
                # assumed: one entry in, one DataFrame out, columns pdb_chain/pdb_res
                # plus one column per amino acid).
                lp1 = df_logits_to_logprobs(lg1[0] if isinstance(lg1, (list, tuple)) else lg1)
                m1 = mean_logprob(lp1, ("B", "C"))
                row["ll_antifold"] = f"{m1['_all']:.6f}" if m1.get("_all") else ""
                row["ll_af_heavy"] = f"{m1['B']:.6f}" if m1.get("B") else ""
                row["ll_af_light"] = f"{m1['C']:.6f}" if m1.get("C") else ""
                # Variant 2: antigen-aware (the paper's claimed advantage over PMPNN)
                df2 = pd.DataFrame([{"pdb": eid, "Hchain": "B", "Lchain": "C",
                                     "antigen_chain": "A"}])
                lg2 = get_pdbs_logits(model, df2, str(STAGE), save_flag=False,
                                      custom_chain_mode=True)
                lp2 = df_logits_to_logprobs(lg2[0] if isinstance(lg2, (list, tuple)) else lg2)
                m2 = mean_logprob(lp2, ("B", "C"))
                row["ll_antifold_ag"] = f"{m2['_all']:.6f}" if m2.get("_all") else ""
                row["status"] = "ok"
            except Exception as exc:
                row["status"] = f"error:{type(exc).__name__}"
                (SCORES_DIR / eid).mkdir(parents=True, exist_ok=True)
                (SCORES_DIR / eid / "antifold_error.txt").write_text(
                    traceback.format_exc())
            row["seconds"] = f"{time.time()-t:.1f}"
            w.writerow(row)
            fh.flush()
            el = time.time() - t0
            print(f"[{i}/{len(todo)}] {eid:<40} {row['status']:<18} "
                  f"ll={row['ll_antifold'] or '-':>10} {row['seconds']:>6}s "
                  f"| ETA {(len(todo)-i)*el/i/60:.0f}m", flush=True)
    print(f"[antifold] done in {(time.time()-t0)/60:.1f}m -> {out_path}")


if __name__ == "__main__":
    main()
