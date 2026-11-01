#!/usr/bin/env python3
"""ProteinMPNN inverse-folding log-likelihood on the ASD within-antigen benchmark.

The §11.5 check (scoring_calibration/PLAN.md). The arc concluded no cheap metric ranks
antibody designs by binding, but that rested on a benchmark whose resolving power was
BORROWED: AbBiBench reports ProteinMPNN rho~0.30 and ESM-IF1 ~0.28 on the same affinity
data where our best inverse-folding column (ESM-IF1 ll_heavy) reached only +0.111.

Two readings, and this run discriminates them:
  * ProteinMPNN here ~0.30  -> the metric mattered; ESM-IF1 was just weak; we have a
    usable ranker, and the bottleneck was never the structures.
  * ProteinMPNN here ~0.14  -> a better inverse-folding model does NOT help on our
    predicted backbones; the limit is the structure (or the predicted-backbone
    washout), not the scorer -- which points the programme at structure quality.

Same scorer family as ESM-IF1 (inverse folding), same structures, same benchmark, same
null control -- so this is a clean metric swap. Higher ll = more native-like given the
backbone = (per AbBiBench) predicted tighter binder, matching -log KD's direction.

What this is NOT
----------------
This scores each antibody sequence against ITS OWN per-complex predicted backbone, the
same structures ESM-IF1 saw. It is therefore directly comparable to ESM-IF1 (+0.111),
but it is NOT AbBiBench's protocol, which threads every mutant onto ONE native backbone
per target. If this comes back low, the per-complex predicted backbone is a confound
that a native-backbone arm (not done here) would isolate. Stated in the write-up.

Scoring, copied from ProteinMPNN's own score_only path (protein_mpnn_run.py):
    log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_enc, randn)
    score     = _scores(S, log_probs, mask*chain_M*chain_M_pos)   # NLL over designed
We average `score` over N_ORDER random decoding orders (fixed per-entry seed) to damp
the autoregressive-order variance, then report ll = -mean(score).

Three quantities per complex:
    ll_antibody   designed = heavy+light (B,C), antigen (A) fixed as context
    ll_heavy      designed = heavy (B) only, (A,C) fixed
    ll_antigen    designed = antigen (A), (B,C) fixed  -- the NULL control, as in esmif

Runs in the boltz2 venv (torch + gemmi), model loaded once. CPU is fine (1.7M params).

Usage:
    .venvs/boltz2/bin/python3 experiments/scoring_calibration/compute_proteinmpnn.py
    .venvs/boltz2/bin/python3 experiments/scoring_calibration/compute_proteinmpnn.py --limit 2
"""

import argparse
import csv
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MPNN_DIR = ROOT / "src" / "proteinmpnn"
SCORES_DIR = HERE / ".scores"
RESULTS = HERE / "results" / "asd_scores.tsv"
OUT = HERE / "results" / "asd_proteinmpnn.tsv"
WEIGHTS = MPNN_DIR / "vanilla_model_weights" / "v_48_020.pt"
N_ORDER = 8  # random decoding orders averaged per score

CARRY = ["entry_id", "set", "label", "kd_molar", "ddg_kcal", "elisa_ratio", "iptm",
         "target_pdb", "neg_log_kd"]
METRICS = ["ll_antibody", "ll_heavy", "ll_antigen", "n_antigen", "n_heavy", "n_light"]
FIELDS = CARRY + METRICS + ["status", "seconds"]

sys.path.insert(0, str(MPNN_DIR))


def load_model(device):
    from protein_mpnn_utils import ProteinMPNN
    ckpt = torch.load(WEIGHTS, map_location=device)
    m = ProteinMPNN(num_letters=21, node_features=128, edge_features=128,
                    hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                    k_neighbors=ckpt["num_edges"], augment_eps=0.0)
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(device).eval()


def cif_to_pdb(cif_path: str, pdb_path: str) -> None:
    import gemmi
    st = gemmi.read_structure(cif_path)
    st.setup_entities()
    st.write_pdb(pdb_path)


def score(model, pdb_dict, designed, fixed, device, seed) -> float:
    """Mean autoregressive NLL of `designed` chains given `fixed`, over N_ORDER orders.

    Replicates protein_mpnn_run.py's score_only block; returns the NLL (lower = more
    native-like), which the caller negates so higher = better.
    """
    from protein_mpnn_utils import tied_featurize, _scores
    name = pdb_dict["name"]
    chain_dict = {name: (list(designed), list(fixed))}
    (X, S, mask, _l, chain_M, chain_enc, _cl, _vl, _ml, _mcl, chain_M_pos, _oa,
     residue_idx, _dm, _tp, _pc, _pb, _plo, _br, _tb) = tied_featurize(
        [pdb_dict], device, chain_dict, None, None, None, None, None, ca_only=False)
    mask_for_loss = mask * chain_M * chain_M_pos
    g = torch.Generator(device="cpu").manual_seed(seed)
    vals = []
    with torch.no_grad():
        for _ in range(N_ORDER):
            randn = torch.randn(chain_M.shape, generator=g).to(device)
            log_probs = model(X, S, mask, chain_M * chain_M_pos, residue_idx,
                              chain_enc, randn)
            vals.append(float(_scores(S, log_probs, mask_for_loss).cpu().numpy()[0]))
    return float(np.mean(vals))


def find_structure(entry_id: str) -> str | None:
    hits = glob.glob(str(SCORES_DIR / entry_id / "**" / "*_model_0.cif"), recursive=True)
    return hits[0] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu",
                    help="cpu is plenty (1.7M params); the box's GPU is often busy")
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    import tempfile

    device = torch.device(args.device)
    model = load_model(device)
    from protein_mpnn_utils import parse_PDB
    print(f"[mpnn] loaded v_48_020 on {device}", flush=True)

    with open(args.results, encoding="utf-8") as fh:
        base = [r for r in csv.DictReader(fh, delimiter="\t")
                if r.get("status") == "ok" and str(r.get("set", "")).startswith("asd")]
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            done = {r["entry_id"] for r in csv.DictReader(fh, delimiter="\t")
                    if r.get("entry_id")}
    todo = [r for r in base if r["entry_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[mpnn] {len(base)} asd entries, {len(done)} done, {len(todo)} to score",
          flush=True)
    if not todo:
        return

    new_file = not out_path.exists()
    if not new_file:
        with out_path.open(encoding="utf-8") as fh:
            if fh.readline().rstrip("\n").split("\t") != FIELDS:
                raise SystemExit(f"[mpnn] {out_path} has a different schema; "
                                 "rewrite it or use a new --out.")
    t0 = time.time()
    with out_path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t",
                           extrasaction="ignore", lineterminator="\n")
        if new_file:
            w.writeheader()
            fh.flush()
        with tempfile.TemporaryDirectory() as td:
            for i, rec in enumerate(todo, 1):
                eid = rec["entry_id"]
                out = {k: rec.get(k, "") for k in CARRY}
                out.update({k: "" for k in METRICS})
                t = time.time()
                try:
                    cif = find_structure(eid)
                    if not cif:
                        raise FileNotFoundError("no structure")
                    pdb = str(Path(td) / "c.pdb")
                    cif_to_pdb(cif, pdb)
                    d = parse_PDB(pdb)[0]
                    d["name"] = eid
                    chains = {c for c in "ABC" if f"seq_chain_{c}" in d}
                    if chains != {"A", "B", "C"}:
                        raise ValueError(f"chains={sorted(chains)}")
                    seed = abs(hash(eid)) % (2 ** 31)
                    out["ll_antibody"] = f"{-score(model, d, ['B','C'], ['A'], device, seed):.6f}"
                    out["ll_heavy"] = f"{-score(model, d, ['B'], ['A','C'], device, seed):.6f}"
                    out["ll_antigen"] = f"{-score(model, d, ['A'], ['B','C'], device, seed):.6f}"
                    out["n_antigen"] = len(d["seq_chain_A"])
                    out["n_heavy"] = len(d["seq_chain_B"])
                    out["n_light"] = len(d["seq_chain_C"])
                    out["status"] = "ok"
                except Exception as exc:
                    out["status"] = f"error:{type(exc).__name__}"
                out["seconds"] = f"{time.time()-t:.1f}"
                w.writerow(out)
                fh.flush()
                if i % 25 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  [{i}/{len(todo)}] {out['status']:<18} "
                          f"ll_ab={out['ll_antibody'] or '-':>10} "
                          f"{el/i:.1f}s/ea ETA {(len(todo)-i)*el/i/60:.0f}m", flush=True)
    print(f"[mpnn] done in {(time.time()-t0)/60:.1f}m -> {out_path}")


if __name__ == "__main__":
    main()
