#!/usr/bin/env python3
"""EXPERIMENT_06 native-backbone arm, stage 2: the TRUE crystal backbone.

Stage 1 (compute_proteinmpnn_shared.py) fixed one PREDICTED cofold per target and lifted
ProteinMPNN's affinity rho from +0.08 to +0.247 -- but the reference was arbitrary, and
two targets inverted (1n8z -0.44, 1mhp -0.19), which could be reference-backbone
pathologies. This stage replaces the predicted reference with the actual CRYSTAL complex
(SAbDab) for each PDB-backed target, threads the mutant antibody sequences onto the
crystal variable-domain backbone, and re-scores. It (a) removes the arbitrary-reference
concern and (b) verifies AbBiBench's absolute number in our hands.

The risk this arm carries, and the gate against it
---------------------------------------------------
The crystal antibody is the WT parent; mutants are same-length variants. But crystals are
Fabs/Fvs with their own residue numbering, missing loops, and constant regions the
VH/VL-only manifest sequences do not have (1n8z, 4fqi crystal H are Fabs). Threading the
wrong residues would silently fabricate a correlation. So every target passes an
ALIGNMENT GATE before any scoring: slide the manifest heavy/light over the crystal
chain's resolved sequence, require the shorter fully contained at >= MIN_ID identity with
no internal gap (high contiguous identity implies no unresolved break in the span). A
target failing the gate is SKIPPED and reported, never threaded on a guess. Verified
2026-11-01: 1mhp/1n8z/4fqi pass (H id 0.91-0.99, L 1.00); 3gbn's light chain is 0.74
(a gap or a divergent chain) and is excluded.

Threading: take the crystal H/L resolved backbone over the matched span, keep its
coordinates, replace its sequence with the mutant's heavy/light, score designed = H,L
conditioned on the (fixed) crystal antigen chain(s). Scores the paratope variable domain
against the real interface -- AbBiBench's protocol.

Only the 4 PDB-backed targets apply (aayl49/aayl51 are AlphaSeq, no crystal).

Usage:
    .venvs/boltz2/bin/python3 experiments/scoring_calibration/compute_proteinmpnn_native.py
"""

import argparse
import collections
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MPNN_DIR = ROOT / "src" / "proteinmpnn"
SABDAB = ROOT / "databases" / "sabdab" / "structures"
SCORES_TSV = HERE / "results" / "asd_scores.tsv"
MANIFEST = HERE / "sets" / "asd_manifest.tsv"
OUT = HERE / "results" / "asd_proteinmpnn_native.tsv"
WEIGHTS = MPNN_DIR / "vanilla_model_weights" / "v_48_020.pt"
N_ORDER = 8
MIN_ID = 0.85   # alignment gate: manifest vs crystal resolved, over the contained span

# Verified crystal chain mapping (SAbDab summary): target -> (Hchain, Lchain, [antigen]).
# 3gbn is intentionally absent: its light chain aligns at only 0.74 (gap/divergence), so
# it fails the gate and is excluded rather than threaded on a guess.
CHAINS = {
    "1mhp": ("H", "L", ["A"]),
    "1n8z": ("B", "A", ["C"]),
    "4fqi": ("H", "L", ["A", "B"]),
    "3gbn": ("H", "L", ["A"]),   # kept so the gate logs WHY it is skipped
}

CARRY = ["entry_id", "set", "label", "iptm", "target_pdb", "neg_log_kd"]
FIELDS = CARRY + ["ll_antibody", "status", "seconds"]

sys.path.insert(0, str(MPNN_DIR))


def load_model(device):
    from protein_mpnn_utils import ProteinMPNN
    ckpt = torch.load(WEIGHTS, map_location=device)
    m = ProteinMPNN(num_letters=21, node_features=128, edge_features=128,
                    hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                    k_neighbors=ckpt["num_edges"], augment_eps=0.0)
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(device).eval()


def best_offset(short: str, long: str) -> tuple[float, int]:
    best = (0.0, -1)
    for off in range(0, len(long) - len(short) + 1):
        idn = sum(a == b for a, b in zip(short, long[off:off + len(short)])) / len(short)
        if idn > best[0]:
            best = (idn, off)
    return best


AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
       'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
       'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
       'TYR': 'Y', 'VAL': 'V'}


def parse_crystal(cif: str, _tmp: str) -> dict:
    """GAP-FREE backbone per chain, straight from gemmi.

    ProteinMPNN's own parse_PDB inserts '-' placeholders for unresolved residues and
    insertion codes, which frame-shifts a contiguous sequence alignment (it tanked the
    gate to id~0.4 on structures that are actually 0.9-1.0). Here we keep ONLY resolved
    standard residues, in order, with no placeholders -- so the sequence aligns cleanly
    to the manifest and each kept residue carries real N/CA/C/O (NaN for a missing
    backbone atom, which ProteinMPNN masks). Mirrors parse_PDB's dict layout
    (seq_chain_X, coords_chain_X[{N,CA,C,O}_chain_X]) so the rest of the script is
    unchanged.
    """
    import gemmi
    st = gemmi.read_structure(cif)
    out = {}
    for model in st:
        for chain in model:
            ch = chain.name
            seq, coords = [], {a: [] for a in ("N", "CA", "C", "O")}
            for res in chain:
                if res.name not in AA3 or res.find_atom("CA", "*") is None:
                    continue
                seq.append(AA3[res.name])
                for a in ("N", "CA", "C", "O"):
                    at = res.find_atom(a, "*")
                    coords[a].append([at.pos.x, at.pos.y, at.pos.z] if at
                                     else [float("nan")] * 3)
            if seq:
                out[f"seq_chain_{ch}"] = "".join(seq)
                out[f"coords_chain_{ch}"] = {f"{a}_chain_{ch}": coords[a]
                                             for a in ("N", "CA", "C", "O")}
        break  # first model only
    return out


def slice_chain(pdb: dict, src: str, dst: str, off: int, length: int, seq: str) -> dict:
    """Relabel crystal chain `src` to `dst`, sliced to [off:off+length], seq replaced."""
    c = pdb[f"coords_chain_{src}"]
    coords = {f"{a}_chain_{dst}": c[f"{a}_chain_{src}"][off:off + length]
              for a in ("N", "CA", "C", "O")}
    return {"seq": seq, "coords": coords}


def score(model, parts: dict, designed: list, fixed: list, name: str, device, seed):
    """parts: {letter: {'seq','coords'}}. NLL of designed chains given fixed."""
    from protein_mpnn_utils import tied_featurize, _scores
    order = fixed + designed
    d = {"name": name, "num_of_chains": len(order),
         "seq": "".join(parts[c]["seq"] for c in order)}
    for c in order:
        d[f"seq_chain_{c}"] = parts[c]["seq"]
        d[f"coords_chain_{c}"] = parts[c]["coords"]
    chain_dict = {name: (list(designed), list(fixed))}
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    import tempfile

    device = torch.device(args.device)
    model = load_model(device)
    print(f"[mpnn-native] loaded v_48_020 on {device}", flush=True)

    man = {r["entry_id"]: r for r in
           csv.DictReader(MANIFEST.open(encoding="utf-8"), delimiter="\t")}
    with SCORES_TSV.open(encoding="utf-8") as fh:
        base = [r for r in csv.DictReader(fh, delimiter="\t")
                if r.get("status") == "ok" and str(r.get("set", "")).startswith("asd")]
    by = collections.defaultdict(list)
    for r in base:
        by[r["target_pdb"]].append(r)

    done = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as fh:
            if fh.readline().rstrip("\n").split("\t") != FIELDS:
                raise SystemExit(f"[mpnn-native] {OUT} schema mismatch")
            done = {ln.split("\t")[0] for ln in fh}

    new_file = not OUT.exists()
    t0 = time.time(); n = 0
    with OUT.open("a", newline="", encoding="utf-8") as fh, \
         tempfile.TemporaryDirectory() as td:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t",
                           extrasaction="ignore", lineterminator="\n")
        if new_file:
            w.writeheader(); fh.flush()
        for t, (hc, lc, ag) in CHAINS.items():
            g = [r for r in by.get(t, []) if r["entry_id"] not in done]
            if not g:
                continue
            cif = SABDAB / f"pdb_0000{t}" / f"pdb_0000{t}_sabdab.cif"
            pdb = parse_crystal(str(cif), str(Path(td) / f"{t}.pdb"))
            qh, ql = g[0]["heavy_seq"] if False else man[g[0]["entry_id"]]["heavy_seq"], None
            # representative manifest VH/VL for the alignment gate
            rep = man[g[0]["entry_id"]]
            qh, ql = rep["heavy_seq"], rep["light_seq"]
            Hc, Lc = pdb.get(f"seq_chain_{hc}"), pdb.get(f"seq_chain_{lc}")
            idh, offh = best_offset(qh, Hc) if Hc and len(Hc) >= len(qh) else best_offset(Hc or "", qh)
            idl, offl = best_offset(ql, Lc) if Lc and len(Lc) >= len(ql) else best_offset(Lc or "", ql)
            # Which sequence is the window (shorter), and the span length scored.
            h_short_is_manifest = len(qh) <= len(Hc)
            l_short_is_manifest = len(ql) <= len(Lc)
            if idh < MIN_ID or idl < MIN_ID:
                print(f"[{t}] SKIP (gate): H id={idh:.2f} L id={idl:.2f} < {MIN_ID}",
                      flush=True)
                continue
            # Antigen chains, fixed context.
            ag_parts = {}
            ag_letters = []
            for i, a in enumerate(ag):
                letter = "A" if i == 0 else chr(ord("D") + i - 1)  # A, D, E...
                aseq = pdb.get(f"seq_chain_{a}")
                if not aseq:
                    continue
                ag_parts[letter] = slice_chain(pdb, a, letter, 0, len(aseq), aseq)
                ag_letters.append(letter)
            print(f"[{t}] threading: H id={idh:.2f} L id={idl:.2f}, "
                  f"antigen chains={ag_letters}, {len(g)} mutants", flush=True)
            for r in g:
                eid = r["entry_id"]
                out = {k: r.get(k, "") for k in CARRY}
                tt = time.time()
                try:
                    m = man[eid]
                    mh, mlt = m["heavy_seq"], m["light_seq"]
                    # Thread onto the crystal span. Window = shorter seq; the crystal
                    # backbone span is [off:off+span], threaded seq = the mutant residues
                    # over that same span.
                    if h_short_is_manifest:
                        Hpart = slice_chain(pdb, hc, "B", offh, len(mh), mh)
                    else:
                        Hpart = slice_chain(pdb, hc, "B", 0, len(Hc), mh[offh:offh + len(Hc)])
                    if l_short_is_manifest:
                        Lpart = slice_chain(pdb, lc, "C", offl, len(mlt), mlt)
                    else:
                        Lpart = slice_chain(pdb, lc, "C", 0, len(Lc), mlt[offl:offl + len(Lc)])
                    parts = {**ag_parts, "B": Hpart, "C": Lpart}
                    seed = abs(hash(eid)) % (2 ** 31)
                    s = score(model, parts, ["B", "C"], ag_letters, t, device, seed)
                    out["ll_antibody"] = f"{-s:.6f}"
                    out["status"] = "ok"
                except Exception as exc:
                    out["status"] = f"error:{type(exc).__name__}"
                out["seconds"] = f"{time.time()-tt:.1f}"
                w.writerow(out); fh.flush()
                n += 1
                if n % 50 == 0:
                    print(f"  [{n}] {(time.time()-t0)/n:.1f}s/ea", flush=True)
    print(f"[mpnn-native] done in {(time.time()-t0)/60:.1f}m -> {OUT}")


if __name__ == "__main__":
    main()
