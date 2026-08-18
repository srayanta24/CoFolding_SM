#!/usr/bin/env python3
"""Shared per-residue node-feature construction for both Model A (geometric-only) and
Model B (geometric + ESM2) — this module builds the geometric half; esm2_features.py
adds the language-model half for Model B.

Features per antigen residue: relative SASA (freesasa, standalone antigen chain only —
we don't have a bound complex for a novel target, so this approximates what's
accessible on the apo/predicted surface), a lightweight backbone-dihedral secondary
structure call (helix/sheet/loop, not DSSP — a supplementary feature, not the primary
signal, so an approximate geometric rule is an acceptable simplification per PLAN.md
§4), and local heavy-atom packing density. Graph: k-NN over CA coordinates, topk=30,
matching BoltzGen's own InverseFoldingEncoder choice
(src/boltzgen/src/boltzgen/model/modules/inverse_fold.py:357) for consistency.
"""

import math
import sys
from pathlib import Path

import freesasa
import numpy as np
import torch

freesasa.setVerbosity(freesasa.nowarnings)  # atom-name-guessing warnings are expected
# noise here (freesasa's classifier doesn't know every atom name it sees) and would
# otherwise print thousands of lines across a full corpus run.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from interface_labels import Atom, get_chain_atoms  # noqa: E402

KNN_K = 30
DENSITY_RADIUS = 10.0  # Angstroms


def group_residues(atoms: list[Atom]) -> list[tuple[tuple[str, str, str], list[Atom]]]:
    """Groups atoms by (auth_asym_id, auth_seq_id, auth_comp_id), sorted by chain then
    numeric residue number (falls back to string sort if non-numeric, e.g. insertion
    codes)."""
    groups: dict[tuple[str, str, str], list[Atom]] = {}
    for a in atoms:
        key = (a.auth_asym_id, a.auth_seq_id, a.auth_comp_id)
        groups.setdefault(key, []).append(a)

    def sort_key(key):
        chain, seq_id, _ = key
        try:
            return (chain, 0, int(seq_id))
        except ValueError:
            return (chain, 1, seq_id)

    return [(k, groups[k]) for k in sorted(groups, key=sort_key)]


def _dihedral(p0, p1, p2, p3) -> float:
    """Standard dihedral angle (radians) from four points, via the cross-product
    formula (no external geometry library needed)."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 /= np.linalg.norm(b1) + 1e-8
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return math.atan2(y, x)


def compute_secondary_structure(residues: list[tuple[tuple[str, str, str], list[Atom]]]) -> list[str]:
    """Backbone phi/psi -> {"H"elix, "E"heet, "L"oop} via broad Ramachandran-region
    rules. Not DSSP-quality (no H-bond pattern analysis) -- a supplementary feature,
    approximate by design (PLAN.md sec 4)."""
    # Per-residue N/CA/C coordinates, None where missing (chain breaks, non-standard residues).
    backbone = []
    for _, res_atoms in residues:
        coords = {a.atom_name: np.array(a.xyz) for a in res_atoms}
        backbone.append((coords.get("N"), coords.get("CA"), coords.get("C")))

    ss = ["L"] * len(residues)
    for i in range(len(residues)):
        chain_i = residues[i][0][0]
        prev_ok = i > 0 and residues[i - 1][0][0] == chain_i and all(x is not None for x in backbone[i - 1])
        next_ok = i < len(residues) - 1 and residues[i + 1][0][0] == chain_i and all(x is not None for x in backbone[i + 1])
        if not (prev_ok and next_ok) or any(x is None for x in backbone[i]):
            continue
        n_i, ca_i, c_i = backbone[i]
        c_prev = backbone[i - 1][2]
        n_next = backbone[i + 1][0]
        phi = math.degrees(_dihedral(c_prev, n_i, ca_i, c_i))
        psi = math.degrees(_dihedral(n_i, ca_i, c_i, n_next))
        if -100 <= phi <= -30 and -77 <= psi <= -17:
            ss[i] = "H"  # alpha-helix region
        elif -170 <= phi <= -50 and 90 <= psi <= 180:
            ss[i] = "E"  # beta-sheet region
    return ss


def compute_sasa(pdb_id: str, antigen_atoms: list[Atom]) -> dict[tuple[str, str, str], float]:
    """Relative SASA per residue, computed on the standalone antigen chain(s) only
    (freesasa.Structure built directly from parsed atoms, not re-reading the CIF —
    freesasa's file loader wants classic PDB format, ours is mmCIF)."""
    struct = freesasa.Structure()
    added = set()
    for a in antigen_atoms:
        # freesasa skips atoms its classifier can't recognize (rare non-standard atom
        # names) rather than raising -- acceptable for a supplementary feature.
        try:
            struct.addAtom(a.atom_name, a.auth_comp_id, a.auth_seq_id, a.auth_asym_id, *a.xyz)
            added.add((a.auth_asym_id, a.auth_seq_id, a.auth_comp_id))
        except Exception:
            continue
    if not added:
        return {}

    result = freesasa.calc(struct)
    areas = result.residueAreas()
    sasa = {}
    for (chain, seq_id, comp_id) in added:
        chain_areas = areas.get(chain, {})
        residue_area = chain_areas.get(seq_id)
        if residue_area is None:
            continue
        # freesasa's hasRelativeAreas flag does NOT guarantee relativeTotal is a real
        # number -- verified directly: 1,409 of 3,520 training structures (40%) had
        # NaN in this column, all traced to residues (non-standard/modified, or ends
        # of chains) where freesasa has no reference max-area to normalize against,
        # even though hasRelativeAreas reports True. This produced NaN training loss
        # across every ensemble member on the first real training run -- caught before
        # trusting the checkpoints, not after. Fall back to raw total/100 (a rough
        # absolute-SASA-in-hundreds-of-A^2 proxy, not a true relative fraction, but a
        # real finite number) whenever the relative value isn't actually usable.
        rel = residue_area.relativeTotal if residue_area.hasRelativeAreas else float("nan")
        if rel != rel or math.isinf(rel):  # NaN != NaN is the standard NaN check
            rel = residue_area.total / 100.0
        sasa[(chain, seq_id, comp_id)] = rel
    return sasa


def compute_local_density(residues: list[tuple[tuple[str, str, str], list[Atom]]], radius: float = DENSITY_RADIUS) -> np.ndarray:
    """Heavy-atom neighbor count within `radius` of each residue's CA, normalized by
    the max count in the structure (0-1 range, comparable across structures of
    different sizes)."""
    ca_coords = []
    all_heavy = []
    for _, res_atoms in residues:
        ca = next((a.xyz for a in res_atoms if a.atom_name == "CA"), None)
        ca_coords.append(ca)
        all_heavy.extend(a.xyz for a in res_atoms)

    all_heavy_xyz = np.array(all_heavy)
    counts = np.zeros(len(residues))
    for i, ca in enumerate(ca_coords):
        if ca is None:
            continue
        dists = np.linalg.norm(all_heavy_xyz - np.array(ca), axis=1)
        counts[i] = (dists <= radius).sum()
    max_count = counts.max() if counts.max() > 0 else 1.0
    return counts / max_count


def build_knn_edge_index(residues: list[tuple[tuple[str, str, str], list[Atom]]], k: int = KNN_K) -> torch.Tensor:
    """k-NN graph over CA coordinates. Implemented directly with torch (pairwise
    distances + topk) rather than torch_geometric.nn.pool.knn_graph: that function
    requires the optional compiled extension pyg-lib, not installed (verified: raises
    ImportError on this machine) -- rather than chase another possibly-aarch64-fragile
    compiled dependency (this project has hit that class of problem repeatedly, see
    DESIGN.md sec 3), a manual k-NN is ~10 lines and has zero new dependencies."""
    ca_coords = []
    for _, res_atoms in residues:
        ca = next((a.xyz for a in res_atoms if a.atom_name == "CA"), None)
        ca_coords.append(ca if ca is not None else (0.0, 0.0, 0.0))
    pos = torch.tensor(ca_coords, dtype=torch.float32)
    k_eff = min(k, len(residues) - 1) if len(residues) > 1 else 1

    dists = torch.cdist(pos, pos)  # [N, N]
    dists.fill_diagonal_(float("inf"))  # exclude self-loops
    _, nn_idx = torch.topk(dists, k_eff, dim=1, largest=False)  # [N, k_eff]

    n = pos.shape[0]
    src = torch.arange(n).unsqueeze(1).expand(-1, k_eff).reshape(-1)
    dst = nn_idx.reshape(-1)
    return torch.stack([src, dst], dim=0)


def build_geometric_features(pdb_id: str) -> dict | None:
    """Returns {"residue_keys": [...], "node_features": Tensor[N, 3], "edge_index":
    Tensor[2, E]}, or None if the structure has no usable antigen. node_features
    columns: [relative_sasa, is_helix, is_sheet] (loop is the implicit all-zero case)
    plus local_density appended -> 4 columns total."""
    chains = get_chain_atoms(pdb_id)
    if chains is None:
        return None
    _, antigen_atoms = chains
    if not antigen_atoms:
        return None

    residues = group_residues(antigen_atoms)
    if len(residues) < 2:
        return None

    sasa = compute_sasa(pdb_id, antigen_atoms)
    ss = compute_secondary_structure(residues)
    density = compute_local_density(residues)

    rows = []
    keys = []
    for i, (key, _) in enumerate(residues):
        keys.append(key)
        rel_sasa = sasa.get(key, 0.0)
        is_helix = 1.0 if ss[i] == "H" else 0.0
        is_sheet = 1.0 if ss[i] == "E" else 0.0
        rows.append([rel_sasa, is_helix, is_sheet, density[i]])

    node_features = torch.tensor(rows, dtype=torch.float32)
    edge_index = build_knn_edge_index(residues)
    return {"residue_keys": keys, "node_features": node_features, "edge_index": edge_index}


if __name__ == "__main__":
    import sys as _sys
    pdb_id = _sys.argv[1] if len(_sys.argv) > 1 else "pdb_00001a14"
    feats = build_geometric_features(pdb_id)
    if feats is None:
        print(f"{pdb_id}: no usable antigen")
    else:
        print(f"{pdb_id}: {len(feats['residue_keys'])} residues, "
              f"node_features {tuple(feats['node_features'].shape)}, "
              f"edge_index {tuple(feats['edge_index'].shape)}")
        print("feature means (sasa, helix, sheet, density):", feats["node_features"].mean(dim=0).tolist())
