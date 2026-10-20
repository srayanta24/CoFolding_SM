#!/usr/bin/env python3
"""ESM-IF1 inverse-folding log-likelihood over the calibration set (EXPERIMENT_02, tier 2).

The third and last independent signal in this experiment. Tier 0 (ipSAE) failed because
it reweights the same PAE matrix ipTM comes from. Tier 1 (Rosetta) scores physics on
the coordinates. This scores a *learned sequence-given-structure* likelihood, which is
a third kind of evidence again.

Why this one is worth testing even after two disappointments: AbBiBench (155,853 mutated
antibodies, 9 antigens) found structure-conditioned inverse-folding likelihoods to be
the **best available affinity predictors** it benchmarked — ProteinMPNN avg Spearman
0.30, ESM-IF1 0.28. Those are weak in absolute terms, but they are the right sign and
the right quantity, which is exactly what ipTM and Rosetta both failed to be on this
benchmark.

Correction to IMPROVE_DESIGN.md's original framing
---------------------------------------------------
That plan claimed tier 2 was nearly free because "BoltzGen already runs an
inverse-folding model as pipeline stage 2 and discards the likelihood." Reading the
code, that overstates it. `InverseFoldingDecoder.forward()` does expose a teacher-forced
logits path, but it masks edge visibility at random (`torch.rand`, inverse_fold.py:622),
so a single pass is stochastic and would need averaging — and driving it at all requires
constructing BoltzGen's full feature dict (k-NN graph, backbone frames, res_type_clone,
masks), which is the data pipeline, not a one-line extraction. ESM-IF1 is both less work
*and* the model AbBiBench actually measured, so it is the better instrument here.

Method
------
`score_sequence_in_complex(model, alphabet, coords, target_chain_id, target_seq)` scores
one chain's sequence conditioned on the backbone of the whole complex, including the
other chains. That is the right conditioning: a mutation that destroys binding should
look less likely *given the antigen is there*, which is information a single-chain score
would not have.

Both antibody chains are scored separately (heavy given A+C, light given A+B) and
reported alongside their mean. The antigen is also scored as a control: it is identical
between a binding-retained and a binding-destroyed mutant of the same parent, so any
apparent discrimination on the antigen column is measuring noise, not signal.

Backbone coordinates (N, CA, C per residue) are parsed from the CIF directly rather than
through `esm.inverse_folding.util.load_structure`, which avoids biotite's file readers
entirely — they changed API between biotite 0.x (which fair-esm 2.0.0 targets) and the
1.2.0 built here. One shim is still required for the module to import at all:
`filter_backbone` was renamed `filter_peptide_backbone`. Patching one name is contained;
downgrading biotite would have risked dragging numpy, and therefore torch, with it — a
failure mode this project has already hit once (DESIGN.md sec 2).

Runs under `.venvs/epiformer`: the only venv with the full GVP stack
(torch_scatter/sparse/cluster, built from source during the EpiFormer work) *and* the
sm_121-capable torch nightly.

Usage:
    TORCH_HOME=$PWD/weights/esm .venvs/epiformer/bin/python3 \
        experiments/scoring_calibration/compute_esmif.py --limit 4
"""

import argparse
import csv
import glob
import os
import time
import traceback
from pathlib import Path

import biotite.structure as _bs

# biotite 1.x rename; see module docstring. Must precede the esm import.
if not hasattr(_bs, "filter_backbone"):
    _bs.filter_backbone = _bs.filter_peptide_backbone

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
SCORES_DIR = HERE / ".scores"
RESULTS = HERE / "results" / "scores.tsv"
OUT = HERE / "results" / "esmif.tsv"

# Superset across both benchmarks; absent columns come through blank.
CARRY = ["entry_id", "set", "label", "kd_molar", "ddg_kcal", "elisa_ratio", "iptm",
         "target_pdb", "neg_log_kd", "affinity_decile"]
METRICS = ["ll_heavy", "ll_light", "ll_antibody_mean", "ll_antigen",
           "n_heavy", "n_light", "n_antigen"]
FIELDS = CARRY + METRICS + ["status", "seconds"]

BACKBONE = ("N", "CA", "C")


def backbone_coords_by_chain(cif_path: Path) -> dict[str, np.ndarray]:
    """auth_asym_id -> [L, 3, 3] array of N, CA, C coordinates, ordered by label_seq_id.

    Header-driven column mapping rather than fixed positions, matching the defensive
    style used elsewhere in this project. Residues missing any backbone atom get NaN,
    which is what ESM-IF1 expects for unresolved positions.
    """
    lines = cif_path.read_text().splitlines()
    cols: list[str] = []
    for line in lines:
        if line.startswith("_atom_site."):
            cols.append(line.split(".", 1)[1].strip())
        elif cols:
            break
    idx = {n: i for i, n in enumerate(cols)}

    per_chain: dict[str, dict[int, dict[str, tuple]]] = {}
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        f = line.split()
        alt = f[idx["label_alt_id"]] if "label_alt_id" in idx else "."
        if alt not in (".", "?", "A"):
            continue
        atom = f[idx["label_atom_id"]]
        if atom not in BACKBONE:
            continue
        chain = f[idx.get("auth_asym_id", idx["label_asym_id"])]
        try:
            seq = int(f[idx["label_seq_id"]])
            xyz = (float(f[idx["Cartn_x"]]), float(f[idx["Cartn_y"]]),
                   float(f[idx["Cartn_z"]]))
        except (ValueError, KeyError):
            continue
        per_chain.setdefault(chain, {}).setdefault(seq, {})[atom] = xyz

    out = {}
    for chain, residues in per_chain.items():
        arr = np.full((len(residues), 3, 3), np.nan, dtype=np.float32)
        for i, seq in enumerate(sorted(residues)):
            for j, atom in enumerate(BACKBONE):
                if atom in residues[seq]:
                    arr[i, j] = residues[seq][atom]
        out[chain] = arr
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--manifest", default=str(HERE / "sets" / "manifest.tsv"))
    ap.add_argument("--results", default=str(RESULTS),
                    help="scored table whose entries to re-score")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import esm
    import esm.inverse_folding as inv

    print(f"[esmif] loading ESM-IF1 on {args.device}", flush=True)
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().to(args.device)

    man = {r["entry_id"]: r for r in
           csv.DictReader(Path(args.manifest).open(encoding="utf-8"), delimiter="\t")}
    scored = [r for r in csv.DictReader(Path(args.results).open(encoding="utf-8"),
                                        delimiter="\t") if r["status"] == "ok"]
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        done = {r["entry_id"] for r in
                csv.DictReader(out_path.open(encoding="utf-8"), delimiter="\t")
                if r.get("entry_id")}
    todo = [r for r in scored if r["entry_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[esmif] {len(scored)} entries, {len(done)} done, {len(todo)} to score",
          flush=True)

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
            out = {k: rec.get(k, "") for k in CARRY}
            out.update({k: "" for k in METRICS})
            t = time.time()
            try:
                m = man[eid]
                hits = glob.glob(str(SCORES_DIR / eid / "**" / "*_model_0.cif"),
                                 recursive=True)
                if not hits:
                    raise FileNotFoundError("no structure")
                coords = backbone_coords_by_chain(Path(hits[0]))
                seqs = {"A": m["antigen_seq"], "B": m["heavy_seq"], "C": m["light_seq"]}
                for ch in ("A", "B", "C"):
                    if ch not in coords or len(coords[ch]) != len(seqs[ch]):
                        raise ValueError(
                            f"chain {ch} coords {len(coords.get(ch, []))} != "
                            f"seq {len(seqs[ch])}")
                with torch.no_grad():
                    ll_h, _ = inv.multichain_util.score_sequence_in_complex(
                        model, alphabet, coords, "B", seqs["B"])
                    ll_l, _ = inv.multichain_util.score_sequence_in_complex(
                        model, alphabet, coords, "C", seqs["C"])
                    ll_a, _ = inv.multichain_util.score_sequence_in_complex(
                        model, alphabet, coords, "A", seqs["A"])
                out["ll_heavy"] = f"{ll_h:.6f}"
                out["ll_light"] = f"{ll_l:.6f}"
                out["ll_antibody_mean"] = f"{(ll_h + ll_l) / 2:.6f}"
                out["ll_antigen"] = f"{ll_a:.6f}"
                out["n_heavy"], out["n_light"] = len(seqs["B"]), len(seqs["C"])
                out["n_antigen"] = len(seqs["A"])
                out["status"] = "ok"
            except Exception as exc:
                out["status"] = f"error:{type(exc).__name__}"
                (SCORES_DIR / eid).mkdir(parents=True, exist_ok=True)
                (SCORES_DIR / eid / "esmif_error.txt").write_text(
                    traceback.format_exc())
            out["seconds"] = f"{time.time()-t:.1f}"
            w.writerow(out)
            fh.flush()
            el = time.time() - t0
            print(f"[{i}/{len(todo)}] {eid:<42} {out['status']:<16} "
                  f"ll_ab={out['ll_antibody_mean'] or '-':>10} {out['seconds']:>6}s "
                  f"| ETA {(len(todo)-i)*el/i/60:.0f}m", flush=True)

    print(f"[esmif] done in {(time.time()-t0)/60:.1f}m -> {out_path}")


if __name__ == "__main__":
    main()
