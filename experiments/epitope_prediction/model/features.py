#!/usr/bin/env python3
"""Shared per-residue node-feature construction for both Model A (geometric-only) and
Model B (geometric + ESM2) — this module builds the geometric half; esm2_features.py
adds the language-model half for Model B.

Features per antigen residue (7 columns, GEOMETRIC_DIM below is the single source of
truth for this count — gnn.py imports it rather than hardcoding): relative SASA
(freesasa, standalone antigen chain only — we don't have a bound complex for a novel
target, so this approximates what's accessible on the apo/predicted surface), a
lightweight backbone-dihedral secondary structure call (helix/sheet/loop, not DSSP — a
supplementary feature, not the primary signal, so an approximate geometric rule is an
acceptable simplification per PLAN.md §4), local heavy-atom packing density,
hydrophobicity (Kyte-Doolittle), a protrusion/curvature proxy, and distance to the
antigen's own centroid. Graph: k-NN over CA coordinates, topk=30, matching BoltzGen's
own InverseFoldingEncoder choice
(src/boltzgen/src/boltzgen/model/modules/inverse_fold.py:357) for consistency.
"""

import math
import sys
from pathlib import Path

import freesasa
import numpy as np
import torch
from scipy.spatial import cKDTree

freesasa.setVerbosity(freesasa.nowarnings)  # atom-name-guessing warnings are expected
# noise here (freesasa's classifier doesn't know every atom name it sees) and would
# otherwise print thousands of lines across a full corpus run.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from interface_labels import Atom, get_chain_atoms  # noqa: E402

KNN_K = 30
DENSITY_RADIUS = 10.0  # Angstroms
PROTRUSION_RADIUS = 15.0  # Angstroms
GEOMETRIC_DIM = 7  # [rel_sasa, is_helix, is_sheet, density, hydrophobicity, protrusion, dist_to_centroid]

# Theoretical max solvent-accessible area per residue type (Tien et al. 2013, Table 1,
# "Theoretical" column, A^2) -- used both to compute relative SASA's NaN-fallback (see
# compute_sasa's docstring) and, implicitly, as a real physicochemical constant rather
# than an arbitrary scaling factor.
MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
    "MSE": 224.0,  # selenomethionine -> MET's value
    "SEC": 167.0,  # selenocysteine -> CYS's value
    "PYL": 236.0,  # pyrrolysine -> closest to LYS
}
DEFAULT_MAX_ASA = sum(MAX_ASA[k] for k in (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL")) / 20  # ~200.6, used for UNK/unrecognized

# Kyte & Doolittle (1982) hydrophobicity scale.
HYDROPHOBICITY = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
    "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
    "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6,
    "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
    "MSE": 1.9, "SEC": 2.5, "PYL": -3.9,
}
DEFAULT_HYDROPHOBICITY = 0.0  # neutral, for UNK/unrecognized


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
            # Was `residue_area.total / 100.0` -- an arbitrary absolute-SASA-in-hundreds
            # proxy that silently mixed scales with the relative-SASA column for the
            # 40% of residues that hit this path. Fixed: divide by the same residue
            # type's real theoretical max area (Tien et al. 2013) so the fallback is a
            # genuine relative-accessibility estimate, on the same 0-ish-to-1-ish scale
            # as the primary (non-fallback) values, clamped since a raw total can
            # occasionally exceed the theoretical max for unusual conformations.
            rel = min(residue_area.total / MAX_ASA.get(comp_id, DEFAULT_MAX_ASA), 2.0)
        sasa[(chain, seq_id, comp_id)] = rel
    return sasa


def compute_protrusion(residues: list[tuple[tuple[str, str, str], list[Atom]]], radius: float = PROTRUSION_RADIUS) -> np.ndarray:
    """Curvature/protrusion proxy (the docstring in an earlier version of this module
    claimed a curvature feature that was never actually implemented -- this is that
    feature, done for real): for each residue's CA, the distance from CA to the
    centroid of neighboring CAs within `radius`. A residue on a convex bump sits
    offset from its neighbors' centroid (large distance); a residue in a concave
    pocket or on a flat patch sits close to it (small distance). Normalized by the
    max value in the structure (0-1), matching compute_local_density's convention."""
    ca_coords = [next((a.xyz for a in res_atoms if a.atom_name == "CA"), None) for _, res_atoms in residues]
    valid_idx = [i for i, c in enumerate(ca_coords) if c is not None]
    if not valid_idx:
        return np.zeros(len(residues))

    valid_xyz = np.array([ca_coords[i] for i in valid_idx])
    protrusion = np.zeros(len(residues))
    for local_i, i in enumerate(valid_idx):
        dists = np.linalg.norm(valid_xyz - valid_xyz[local_i], axis=1)
        neighbor_mask = (dists > 0) & (dists <= radius)
        if not neighbor_mask.any():
            continue
        neighbor_centroid = valid_xyz[neighbor_mask].mean(axis=0)
        protrusion[i] = np.linalg.norm(valid_xyz[local_i] - neighbor_centroid)

    max_val = protrusion.max() if protrusion.max() > 0 else 1.0
    return protrusion / max_val


def compute_dist_to_centroid(residues: list[tuple[tuple[str, str, str], list[Atom]]]) -> np.ndarray:
    """CA distance from the whole-antigen-chain centroid, normalized by the structure's
    own max such distance (0-1) -- a cheap proxy for "how deep in the fold vs. how far
    out toward the surface periphery" a residue sits, complementary to SASA (which
    captures local exposure, not overall position relative to the fold's bulk)."""
    ca_coords = [next((a.xyz for a in res_atoms if a.atom_name == "CA"), None) for _, res_atoms in residues]
    valid = [c for c in ca_coords if c is not None]
    if not valid:
        return np.zeros(len(residues))

    centroid = np.array(valid).mean(axis=0)
    dist = np.array([np.linalg.norm(np.array(c) - centroid) if c is not None else 0.0 for c in ca_coords])
    max_val = dist.max() if dist.max() > 0 else 1.0
    return dist / max_val


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


## --- Model C (EpiFormer-inspired multi-relational EGNN encoder) support below ---
## Antigen-only: no antibody input, no cross-attention (see PLAN.md's EpiFormer
## write-up) -- this only borrows EpiFormer's antigen-side EGNN-R encoder idea.

RESIDUE_TYPES = [  # same 20-residue order as MAX_ASA/HYDROPHOBICITY above, for a stable one-hot index
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
RESIDUE_TYPE_IDX = {r: i for i, r in enumerate(RESIDUE_TYPES)}
RESIDUE_TYPE_DIM = len(RESIDUE_TYPES) + 1  # +1 "other" bucket for MSE/SEC/PYL/UNK/unrecognized

SHORT_RANGE_MAX_SEP = 3  # relation rho2: |i-j| in [2, SHORT_RANGE_MAX_SEP]
MEDIUM_RANGE_RADIUS = 8.0  # Angstroms, relation rho4
N_RELATIONS = 4  # rho1 sequential, rho2 short-range, rho3 k-NN, rho4 medium-range spatial


def residue_type_onehot(residues: list[tuple[tuple[str, str, str], list[Atom]]]) -> torch.Tensor:
    """[N, RESIDUE_TYPE_DIM] one-hot residue identity -- EGNN-R's node scalars need
    "residue type" explicitly (PLAN.md's EpiFormer write-up); nothing in the existing
    7-column geometric feature set encodes identity, only local geometry/chemistry."""
    rows = []
    for (_, _, comp_id), _ in residues:
        vec = [0.0] * RESIDUE_TYPE_DIM
        vec[RESIDUE_TYPE_IDX.get(comp_id, RESIDUE_TYPE_DIM - 1)] = 1.0
        rows.append(vec)
    return torch.tensor(rows, dtype=torch.float32)


def _virtual_cbeta(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Idealized virtual C-beta position from N/CA/C (standard tetrahedral-geometry
    construction, used for glycine which has no real side chain / C-beta atom).
    Coefficients are the commonly-used ones for this construction (e.g. as used in
    Rosetta-family tools) -- not derived here, just applied."""
    b = ca - n
    cc = c - ca
    a = np.cross(b, cc)
    return ca - 0.58273431 * a + 0.56802827 * b - 0.54067466 * cc


def build_backbone_coords(residues: list[tuple[tuple[str, str, str], list[Atom]]]) -> torch.Tensor:
    """[N, 4, 3] backbone coordinate matrix (N, CA, C-beta, O) per residue, matching
    EGNN-R's node coordinate input. Real C-beta used when present; glycine (no C-beta)
    and any other residue missing it get the virtual placement above. A residue
    missing N/CA/C entirely (chain break, unresolved backbone) gets its CA repeated
    across all 4 slots -- keeps the tensor dense/rectangular without fabricating a
    geometrically meaningless position; such residues contribute a ~zero coordinate
    spread instead of NaNs propagating into the equivariant updates."""
    rows = []
    for _, res_atoms in residues:
        coords = {a.atom_name: np.array(a.xyz) for a in res_atoms}
        n, ca, c, o = coords.get("N"), coords.get("CA"), coords.get("C"), coords.get("O")
        if ca is None:
            ca = np.zeros(3)
        if n is None or c is None:
            n = c = ca
        cb = coords.get("CB")
        if cb is None:
            cb = _virtual_cbeta(n, ca, c) if not (n is ca) else ca
        if o is None:
            o = ca
        rows.append(np.stack([n, ca, cb, o]))
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def build_multirelational_edges(residues: list[tuple[tuple[str, str, str], list[Atom]]], k: int = KNN_K,
                                 short_range: int = SHORT_RANGE_MAX_SEP, radius: float = MEDIUM_RANGE_RADIUS) -> dict[int, torch.Tensor]:
    """Returns {relation_id: edge_index [2, E]} for the 4 EGNN-R relations:
    1=sequential (covalent, |i-j|==1, same chain), 2=short-range (2<=|i-j|<=short_range,
    same chain), 3=k-NN (reuses build_knn_edge_index's CA-based k-NN), 4=medium-range
    spatial (CA-CA within `radius`, same convention as interface_labels.py's KD-tree
    fix for large structures, excluding pairs already in relations 1-3 to keep the
    relations a genuine partition rather than heavily overlapping).

    Verified/real finding, not a bug: with KNN_K=30 (this project's existing k, matched
    to BoltzGen's own InverseFoldingEncoder), relation 4 comes back empty on every
    structure checked (15 to 388 residues) -- at typical CA-CA spacing, 30 nearest
    neighbors already covers essentially everything within 8A, so the k-NN relation
    (3) structurally subsumes the medium-range relation (4) at this k. Left in per the
    plan's literal 4-relation spec (matching EpiFormer's own ratio set) rather than
    tuning KNN_K down to force non-overlap -- a smaller k would deviate from this
    project's established KNN_K=30 convention for no clearly better reason."""
    n = len(residues)
    chains = [key[0] for key, _ in residues]
    ca_coords = np.array([
        next((a.xyz for a in res_atoms if a.atom_name == "CA"), (0.0, 0.0, 0.0))
        for _, res_atoms in residues
    ])

    seq_edges = [[], []]
    short_edges = [[], []]
    for i in range(n):
        for j in range(n):
            if i == j or chains[i] != chains[j]:
                continue
            sep = abs(i - j)
            if sep == 1:
                seq_edges[0].append(i)
                seq_edges[1].append(j)
            elif 2 <= sep <= short_range:
                short_edges[0].append(i)
                short_edges[1].append(j)

    knn_edge_index = build_knn_edge_index(residues, k=k)
    knn_pairs = {(int(s), int(d)) for s, d in zip(knn_edge_index[0].tolist(), knn_edge_index[1].tolist())}
    seq_pairs = set(zip(seq_edges[0], seq_edges[1]))
    short_pairs = set(zip(short_edges[0], short_edges[1]))

    tree = cKDTree(ca_coords)
    pairs = tree.query_pairs(r=radius)  # unordered {(i, j), ...}, i < j
    medium_edges = [[], []]
    for i, j in pairs:
        for a, b in ((i, j), (j, i)):
            if (a, b) not in seq_pairs and (a, b) not in short_pairs and (a, b) not in knn_pairs:
                medium_edges[0].append(a)
                medium_edges[1].append(b)

    def to_tensor(edges):
        if not edges[0]:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long)

    return {
        1: to_tensor(seq_edges),
        2: to_tensor(short_edges),
        3: knn_edge_index,
        4: to_tensor(medium_edges),
    }


def build_multirelational_features(pdb_id: str) -> dict | None:
    """Model C's input: same residue_keys/labels convention as build_geometric_features,
    plus node_scalars (7-col geometric features + RESIDUE_TYPE_DIM-col one-hot, for
    EGNN-R's node scalar input), backbone_coords ([N,4,3], for its equivariant
    coordinate input), and edges_by_relation ({1..4: edge_index})."""
    chains = get_chain_atoms(pdb_id)
    if chains is None:
        return None
    _, antigen_atoms = chains
    if not antigen_atoms:
        return None

    residues = group_residues(antigen_atoms)
    if len(residues) < 2:
        return None

    geo = build_geometric_features(pdb_id)
    if geo is None:
        return None

    onehot = residue_type_onehot(residues)
    node_scalars = torch.cat([geo["node_features"], onehot], dim=-1)
    backbone_coords = build_backbone_coords(residues)
    edges_by_relation = build_multirelational_edges(residues)

    return {
        "residue_keys": geo["residue_keys"],
        "node_scalars": node_scalars,
        "backbone_coords": backbone_coords,
        "edges_by_relation": edges_by_relation,
    }


def build_geometric_features(pdb_id: str) -> dict | None:
    """Returns {"residue_keys": [...], "node_features": Tensor[N, GEOMETRIC_DIM],
    "edge_index": Tensor[2, E]}, or None if the structure has no usable antigen.
    node_features columns: [relative_sasa, is_helix, is_sheet (loop is the implicit
    all-zero case), local_density, hydrophobicity, protrusion, dist_to_centroid]."""
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
    protrusion = compute_protrusion(residues)
    dist_to_centroid = compute_dist_to_centroid(residues)

    rows = []
    keys = []
    for i, (key, _) in enumerate(residues):
        keys.append(key)
        comp_id = key[2]
        rel_sasa = sasa.get(key, 0.0)
        is_helix = 1.0 if ss[i] == "H" else 0.0
        is_sheet = 1.0 if ss[i] == "E" else 0.0
        hydrophobicity = HYDROPHOBICITY.get(comp_id, DEFAULT_HYDROPHOBICITY)
        rows.append([rel_sasa, is_helix, is_sheet, density[i], hydrophobicity, protrusion[i], dist_to_centroid[i]])

    node_features = torch.tensor(rows, dtype=torch.float32)
    assert node_features.shape[1] == GEOMETRIC_DIM
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
        cols = ["sasa", "helix", "sheet", "density", "hydrophobicity", "protrusion", "dist_to_centroid"]
        means = feats["node_features"].mean(dim=0).tolist()
        print("feature means:", dict(zip(cols, means)))
