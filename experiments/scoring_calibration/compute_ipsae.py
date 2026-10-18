#!/usr/bin/env python3
"""ipSAE over the calibration set's saved PAE matrices (EXPERIMENT_02, tier 0).

ipSAE ("interprotein Score from Aligned Errors", Dunbrack lab) is a drop-in
replacement for ipTM that fixes three specific defects: it counts only residue pairs
with confident interchain PAE, derives d0 from the number of such residues rather
than from total chain length, and uses the PAE value directly instead of a
probability distribution over aligned errors. It is the single best-performing
feature in the 3,766-binder meta-analysis (Overath et al. 2025), where ipTM ranks
only fourth.

This costs no GPU time: Boltz-2 already wrote a `pae_*.npz` per entry during
EXPERIMENT_01, and all 332 are on disk. We are re-reading data already paid for.

Algorithm, transcribed from the reference implementation
---------------------------------------------------------
github.com/DunbrackLab/IPSAE. Transcribed rather than recalled, because a
plausible-looking wrong implementation would produce a number that looks fine and
means nothing.

    d0(L)      = 1.24 * (L - 15)**(1/3) - 1.8   if L > 27, else 1.0, floored at 1.0
    ptm(x, d0) = 1 / (1 + (x / d0)**2)

For a directed group pair (align on group1, score group2), for each residue i in
group1: select j in group2 with PAE[i, j] < pae_cutoff, take the mean of ptm over
those j, and the directed score is the **max** over i. Three d0 variants differ only
in what length feeds d0:

    ipsae_d0chn  d0 from len(group1) + len(group2)              — all pairs
    ipsae_d0dom  d0 from the count of residues in either group
                 having any interchain PAE < cutoff             — subset
    ipsae_d0res  d0 computed per-residue i from its own count
                 of passing partners                            — per-residue

`ipsae_min` is min(group1→group2, group2→group1), which is the asymmetry fix and the
variant the meta-analysis reports.

One deliberate deviation, for comparability
--------------------------------------------
Stock ipsae.py scores **chain pairs**. Our complexes have three chains — antigen (A),
heavy (B), light (C) — and an antibody paratope spans both B and C. The metric this
has to be comparable with is `design_to_target_iptm`, which treats the whole designed
antibody as one unit against the target. So the primary score here groups **B+C as
one "binder"** against A as "target". The per-chain-pair values (A-B, A-C) are also
emitted for reference, but the grouped number is the one to compare.

Chain boundaries come from the manifest's `n_res_antigen`/`n_res_heavy`/`n_res_light`
rather than from parsing CIFs: `run_design.py`'s `build_boltz_config` always emits the
target as chain A first, then the partner chains as B and C, so the PAE row order is
antigen → heavy → light. Verified directly on `pdb_000013bf` (51 + 221 + 215 = 487,
matching the PAE matrix shape).

Requires numpy, so this runs under `.venvs/epitope-prediction` rather than bare
python3 — a necessary exception to the stdlib-only orchestration convention, since
`.npz` cannot be read without it.

Usage:
    .venvs/epitope-prediction/bin/python3 experiments/scoring_calibration/compute_ipsae.py
"""

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCORES_DIR = HERE / ".scores"
MANIFEST = HERE / "sets" / "manifest.tsv"
RESULTS = HERE / "results" / "scores.tsv"
OUT = HERE / "results" / "ipsae.tsv"

PAE_CUTOFF = 10.0  # Dunbrack's stated operating point: interchain PAE < 10 A

FIELDS = [
    "entry_id", "set", "label", "kd_molar", "ddg_kcal", "elisa_ratio", "iptm",
    "ipsae_min", "ipsae_d0chn_min", "ipsae_d0dom_min",
    "ipsae_ab_to_ag", "ipsae_ag_to_ab",
    "ipsae_heavy_only", "ipsae_light_only", "n_pae", "status",
]


def d0(L: float, minimum: float = 1.0) -> float:
    L = float(L)
    if L > 27.0:
        return max(minimum, 1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8)
    return minimum


def ptm(x: np.ndarray, d: float) -> np.ndarray:
    return 1.0 / (1.0 + (x / d) ** 2.0)


def directed(pae: np.ndarray, rows: np.ndarray, cols: np.ndarray,
             cutoff: float) -> dict[str, float]:
    """Align on `rows`, score `cols`. Returns the three d0 variants."""
    sub = pae[np.ix_(rows, cols)]
    valid = sub < cutoff
    if not valid.any():
        return {"d0chn": 0.0, "d0dom": 0.0, "d0res": 0.0}

    # d0chn: both group lengths
    d_chn = d0(len(rows) + len(cols))
    # d0dom: residues in EITHER group with any interchain pair under the cutoff
    n_dom = int((valid.any(axis=1)).sum() + (valid.any(axis=0)).sum())
    d_dom = d0(n_dom)

    out = {}
    for name, dval in (("d0chn", d_chn), ("d0dom", d_dom)):
        per_res = np.where(valid.any(axis=1),
                           np.divide((ptm(sub, dval) * valid).sum(axis=1),
                                     np.maximum(valid.sum(axis=1), 1)),
                           0.0)
        out[name] = float(per_res.max())

    # d0res: per-residue d0 from that residue's own count of passing partners
    counts = valid.sum(axis=1)
    per_res = np.zeros(len(rows))
    for i in range(len(rows)):
        if counts[i] == 0:
            continue
        per_res[i] = ptm(sub[i][valid[i]], d0(counts[i])).mean()
    out["d0res"] = float(per_res.max())
    return out


def find_pae(entry_id: str) -> Path | None:
    hits = glob.glob(str(SCORES_DIR / entry_id / "**" / "pae_*.npz"), recursive=True)
    return Path(hits[0]) if hits else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pae-cutoff", type=float, default=PAE_CUTOFF)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    man = {r["entry_id"]: r for r in
           csv.DictReader(MANIFEST.open(encoding="utf-8"), delimiter="\t")}
    scored = list(csv.DictReader(RESULTS.open(encoding="utf-8"), delimiter="\t"))
    if args.limit:
        scored = scored[: args.limit]

    rows, skipped = [], 0
    for rec in scored:
        eid = rec["entry_id"]
        m = man.get(eid)
        out = {k: "" for k in FIELDS}
        out.update({k: rec.get(k, "") for k in
                    ("entry_id", "set", "label", "kd_molar", "ddg_kcal",
                     "elisa_ratio", "iptm")})
        pae_path = find_pae(eid)
        if m is None or pae_path is None:
            out["status"] = "no_pae"
            rows.append(out)
            skipped += 1
            continue

        pae = np.load(pae_path)["pae"]
        n_ag = int(m["n_res_antigen"])
        n_h = int(m["n_res_heavy"])
        n_l = int(m["n_res_light"])
        if pae.shape[0] != n_ag + n_h + n_l:
            # Boltz tokenisation can diverge from our residue counts; don't guess.
            out["status"] = f"shape_mismatch_{pae.shape[0]}_vs_{n_ag+n_h+n_l}"
            rows.append(out)
            skipped += 1
            continue

        ag = np.arange(0, n_ag)
        hv = np.arange(n_ag, n_ag + n_h)
        lt = np.arange(n_ag + n_h, n_ag + n_h + n_l)
        ab = np.concatenate([hv, lt])

        fwd = directed(pae, ab, ag, args.pae_cutoff)   # align on antibody, score antigen
        rev = directed(pae, ag, ab, args.pae_cutoff)   # align on antigen, score antibody

        out["ipsae_ab_to_ag"] = f"{fwd['d0res']:.6f}"
        out["ipsae_ag_to_ab"] = f"{rev['d0res']:.6f}"
        out["ipsae_min"] = f"{min(fwd['d0res'], rev['d0res']):.6f}"
        out["ipsae_d0chn_min"] = f"{min(fwd['d0chn'], rev['d0chn']):.6f}"
        out["ipsae_d0dom_min"] = f"{min(fwd['d0dom'], rev['d0dom']):.6f}"
        out["ipsae_heavy_only"] = f"{min(directed(pae, hv, ag, args.pae_cutoff)['d0res'], directed(pae, ag, hv, args.pae_cutoff)['d0res']):.6f}"
        out["ipsae_light_only"] = f"{min(directed(pae, lt, ag, args.pae_cutoff)['d0res'], directed(pae, ag, lt, args.pae_cutoff)['d0res']):.6f}"
        out["n_pae"] = pae.shape[0]
        out["status"] = "ok"
        rows.append(out)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t",
                           extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}  ({len(rows)} rows, {skipped} without usable PAE, "
          f"pae_cutoff={args.pae_cutoff})")


if __name__ == "__main__":
    main()
