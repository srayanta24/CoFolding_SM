#!/usr/bin/env python3
"""EXPERIMENT_06, native-backbone arm (stage 1: shared predicted backbone).

Arm A (compute_proteinmpnn.py) scored each antibody against ITS OWN per-complex
predicted backbone and got rho=+0.097 -- near ESM-IF1, far under the published 0.30.
A low result there has two non-exclusive causes (EXPERIMENT_06 §6):
  (a) predicted-backbone quality, and
  (b) per-complex-backbone WASHOUT -- scoring a sequence against the backbone folded
      from it is partly circular, flattening the mutation signal.

This stage isolates (b). For each target it fixes ONE backbone -- the best-modelled
cofold (highest ipTM) -- and threads every mutant's heavy+light sequence onto it, then
scores with ProteinMPNN conditioned on the antigen. One fixed backbone + threaded
mutants is the structural essence of AbBiBench's protocol, minus the crystal.

  * If this is much higher than Arm A's +0.097, the per-complex washout was the problem:
    our BENCHMARK CONSTRUCTION (one cofold per mutant) destroyed signal, independent of
    metric or absolute structure quality. That is actionable for every future benchmark.
  * If it stays ~0.10, washout is not the cause, and the remaining suspects are
    predicted-structure quality (needs the true-crystal arm) or non-transfer.

Zero alignment risk, which is the whole point of doing this before the crystal arm:
every cofold of a target shares the manifest's exact heavy/light lengths, so threading
is a sequence swap with coordinates untouched -- no renumbering, no VH-region matching,
no missing-residue mismatch. (The crystal arm must align manifest VH to a Fab backbone;
1n8z and 4fqi crystals include CH1, so that arm carries the mis-registration risk this
one does not.)

Threading mechanism: ProteinMPNN reads each chain's sequence AND coordinates from the
parsed dict. Replacing seq_chain_B/C with the mutant's heavy/light while keeping
coords_chain_B/C (the reference backbone) scores the mutant sequence on the reference
fold. Verified by a length assertion that refuses any mismatch.

Usage:
    .venvs/boltz2/bin/python3 experiments/scoring_calibration/compute_proteinmpnn_shared.py
"""

import argparse
import collections
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
SCORES_TSV = HERE / "results" / "asd_scores.tsv"
MANIFEST = HERE / "sets" / "asd_manifest.tsv"
OUT = HERE / "results" / "asd_proteinmpnn_shared.tsv"
WEIGHTS = MPNN_DIR / "vanilla_model_weights" / "v_48_020.pt"
N_ORDER = 8

CARRY = ["entry_id", "set", "label", "iptm", "target_pdb", "neg_log_kd"]
FIELDS = CARRY + ["ll_antibody", "ref_entry", "status", "seconds"]

sys.path.insert(0, str(MPNN_DIR))


def load_model(device):
    from protein_mpnn_utils import ProteinMPNN
    ckpt = torch.load(WEIGHTS, map_location=device)
    m = ProteinMPNN(num_letters=21, node_features=128, edge_features=128,
                    hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                    k_neighbors=ckpt["num_edges"], augment_eps=0.0)
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(device).eval()


def cif_to_dict(cif_path: str, name: str, tmp: str):
    import gemmi
    from protein_mpnn_utils import parse_PDB
    st = gemmi.read_structure(cif_path)
    st.setup_entities()
    st.write_pdb(tmp)
    d = parse_PDB(tmp)[0]
    d["name"] = name
    return d


def score_threaded(model, ref_dict, heavy, light, device, seed) -> float:
    """NLL of (heavy,light) threaded onto ref_dict's backbone, antigen fixed."""
    from protein_mpnn_utils import tied_featurize, _scores
    if len(heavy) != len(ref_dict["seq_chain_B"]) or \
       len(light) != len(ref_dict["seq_chain_C"]):
        raise ValueError("length mismatch vs reference backbone")
    d = dict(ref_dict)
    d["seq_chain_B"], d["seq_chain_C"] = heavy, light
    d["seq"] = d["seq_chain_A"] + heavy + light      # A,B,C concat order
    chain_dict = {d["name"]: (["B", "C"], ["A"])}
    (X, S, mask, _l, chain_M, chain_enc, _cl, _vl, _ml, _mcl, chain_M_pos, _oa,
     residue_idx, _dm, _tp, _pc, _pb, _plo, _br, _tb) = tied_featurize(
        [d], device, chain_dict, None, None, None, None, None, ca_only=False)
    mask_for_loss = mask * chain_M * chain_M_pos
    g = torch.Generator(device="cpu").manual_seed(seed)
    vals = []
    with torch.no_grad():
        for _ in range(N_ORDER):
            randn = torch.randn(chain_M.shape, generator=g).to(device)
            lp = model(X, S, mask, chain_M * chain_M_pos, residue_idx, chain_enc, randn)
            vals.append(float(_scores(S, lp, mask_for_loss).cpu().numpy()[0]))
    return float(np.mean(vals))


def find_structure(entry_id: str) -> str | None:
    hits = glob.glob(str(SCORES_DIR / entry_id / "**" / "*_model_0.cif"), recursive=True)
    return hits[0] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit-targets", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import tempfile
    device = torch.device(args.device)
    model = load_model(device)
    print(f"[mpnn-shared] loaded v_48_020 on {device}", flush=True)

    man = {r["entry_id"]: r for r in
           csv.DictReader(MANIFEST.open(encoding="utf-8"), delimiter="\t")}
    with SCORES_TSV.open(encoding="utf-8") as fh:
        base = [r for r in csv.DictReader(fh, delimiter="\t")
                if r.get("status") == "ok" and str(r.get("set", "")).startswith("asd")]
    by = collections.defaultdict(list)
    for r in base:
        by[r["target_pdb"]].append(r)
    targets = sorted(by)
    if args.limit_targets:
        targets = targets[: args.limit_targets]

    done = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as fh:
            if fh.readline().rstrip("\n").split("\t") != FIELDS:
                raise SystemExit(f"[mpnn-shared] {OUT} has a different schema")
            done = {r.split("\t")[0] for r in fh}

    new_file = not OUT.exists()
    t0 = time.time()
    n_done = 0
    with OUT.open("a", newline="", encoding="utf-8") as fh, \
         tempfile.TemporaryDirectory() as td:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t",
                           extrasaction="ignore", lineterminator="\n")
        if new_file:
            w.writeheader(); fh.flush()
        for t in targets:
            g = [r for r in by[t] if r["entry_id"] not in done]
            if not g:
                continue
            # Reference = best-modelled backbone in this target (highest ipTM).
            ref = max(by[t], key=lambda r: float(r["iptm"] or 0))
            ref_cif = find_structure(ref["entry_id"])
            try:
                ref_dict = cif_to_dict(ref_cif, ref["entry_id"], str(Path(td) / "r.pdb"))
            except Exception as exc:
                print(f"[{t}] reference load failed: {exc}", flush=True)
                continue
            print(f"[{t}] ref={ref['entry_id']} ipTM={ref['iptm']} "
                  f"VH={len(ref_dict.get('seq_chain_B',''))} "
                  f"VL={len(ref_dict.get('seq_chain_C',''))}, {len(g)} to score", flush=True)
            for r in g:
                eid = r["entry_id"]
                out = {k: r.get(k, "") for k in CARRY}
                out["ref_entry"] = ref["entry_id"]
                tt = time.time()
                try:
                    m = man[eid]
                    seed = abs(hash(eid)) % (2 ** 31)
                    s = score_threaded(model, ref_dict, m["heavy_seq"], m["light_seq"],
                                       device, seed)
                    out["ll_antibody"] = f"{-s:.6f}"
                    out["status"] = "ok"
                except Exception as exc:
                    out["status"] = f"error:{type(exc).__name__}"
                out["seconds"] = f"{time.time()-tt:.1f}"
                w.writerow(out); fh.flush()
                n_done += 1
                if n_done % 50 == 0:
                    el = time.time() - t0
                    print(f"  [{n_done}] {el/n_done:.1f}s/ea", flush=True)
    print(f"[mpnn-shared] done in {(time.time()-t0)/60:.1f}m -> {OUT}")


if __name__ == "__main__":
    main()
