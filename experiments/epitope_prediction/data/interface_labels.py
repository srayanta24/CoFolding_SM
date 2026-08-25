#!/usr/bin/env python3
"""Our own coordinate-based antibody-antigen interface labeler.

Built because AACDB's precomputed labels (databases/aacdb/interacting_res_distance/)
only cover 4 of our 851 databases/splits/{test,dev}.txt structures — this computes
interface contacts directly from databases/sabdab/structures/'s own mmCIF coordinates
for full coverage (1,437 test+dev structures have a real protein antigen and usable
coordinates).

Chain-id gotcha, verified the hard way (an earlier version of this file got this wrong
and was caught by testing before it shipped): summary.csv's Hchain/Lchain/antigen_chain
columns are **auth_asym_id**, not label_asym_id, despite label_asym_id being the mmCIF
field that superficially looks like "the" chain id. Confirmed systematically across a
30-structure random sample: `_entity_poly.pdbx_strand_id` (the field
databases/src/build_splits.py matches antigen_chain against) equals auth_asym_id in
every case where it's unambiguous (17/30), and is consistent with both in the remaining
13/30 only because label happened to equal auth for those particular structures — it
never matched label_asym_id exclusively. So both this file and build_splits.py filter
atoms by **auth_asym_id** (also directly usable as the output key — no separate lookup
needed, unlike an earlier draft of this file that tried to keep the two id systems
separate). AACDB's own interacting_res_distance files also use auth numbering (e.g.
"N:THR401"), so labels keyed this way are directly cross-referenceable for the sanity
check below.

Usage:
    python3 experiments/epitope_prediction/data/interface_labels.py <pdb_id>
    python3 experiments/epitope_prediction/data/interface_labels.py --sanity-check
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SABDAB_DIR = REPO_ROOT / "databases" / "sabdab"
AACDB_DIR = REPO_ROOT / "databases" / "aacdb"
THRESHOLD = 5.0  # Angstroms; AACDB's own files verified to cap at 5.99A

ATOM_LINE_RE = re.compile(r"^(ATOM|HETATM)\s+")

# Real bug found while expanding training coverage past AACDB's own protein-contact-only
# scope (PLAN.md-style write-up worth keeping visible): get_chain_atoms() used to filter
# only by element (type_symbol != "H"), not residue identity, so a SAbDab "antigen_chain"
# that's actually a bound ion/hapten/sugar (summary.csv's own antigen_type column, e.g.
# "ION|HAPTEN") got scored as protein epitope residues -- verified concretely on
# pdb_00001a0q, whose "3 interface residues" were 2 zinc ions + a heparin fragment, not
# amino acids. Affects 1,915/8,072 (24%) of train_era (dev.txt/test.txt are ~98% clean,
# 2/100 and 16/751 respectively). Fixed structurally here rather than by consulting
# antigen_type per-PDB, since a "PROTEIN|ION" mixed structure can still have individual
# non-protein residues on the same chain that need excluding at the residue level.
STANDARD_RESIDUES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL", "UNK",  # common non-standard-but-real polypeptide residues
})


@dataclass
class Atom:
    type_symbol: str
    atom_name: str
    label_asym_id: str
    label_seq_id: str
    auth_asym_id: str
    auth_seq_id: str
    auth_comp_id: str
    xyz: tuple[float, float, float]


def parse_atom_site(cif_path: Path) -> list[Atom]:
    """Small targeted parser for the _atom_site loop, matching the same style as
    build_splits.py's parse_entity_poly — not a general CIF parser. Column order
    verified fixed for SAbDab's own re-generated CIFs (see module docstring's source
    citation): group_PDB id type_symbol label_atom_id label_alt_id label_comp_id
    label_asym_id label_entity_id label_seq_id pdbx_PDB_ins_code Cartn_x Cartn_y
    Cartn_z occupancy B_iso_or_equiv pdbx_formal_charge auth_atom_id auth_comp_id
    auth_seq_id auth_asym_id pdbx_PDB_model_num."""
    atoms = []
    for line in cif_path.read_text().splitlines():
        if not ATOM_LINE_RE.match(line):
            continue
        f = line.split()
        if len(f) < 21:
            continue
        # Alternate conformations (label_alt_id, field 4): verified real, e.g.
        # pdb_00009jim residue SER 474 has two full sets of atoms at alt_id "A"/"B"
        # (occupancy 0.51/0.49). Keeping both produced two atoms of the same name
        # (e.g. two "CA") in one residue, which fed silently-duplicated coordinates
        # into distance calcs and made freesasa produce NaN for that residue (traced
        # to a NaN training loss across an entire ensemble run before this fix). "."
        # means no alternate; when alternates exist, keep only "A" (conventionally
        # the primary/highest-occupancy one).
        alt_id = f[4]
        if alt_id not in (".", "A"):
            continue
        atoms.append(Atom(
            type_symbol=f[2],
            atom_name=f[16],  # auth_atom_id (e.g. "CA") -- label_atom_id (f[3]) also
                               # works but auth is consistent with every other field
                               # here being auth-convention, per the module docstring
            label_asym_id=f[6],
            label_seq_id=f[8],
            auth_asym_id=f[19],
            auth_seq_id=f[18],
            auth_comp_id=f[17],
            xyz=(float(f[10]), float(f[11]), float(f[12])),
        ))
    return atoms


_summary_cache: dict[str, list[dict]] | None = None


def _load_summary_rows(pdb_id: str) -> list[dict]:
    global _summary_cache
    if _summary_cache is None:
        _summary_cache = {}
        with open(SABDAB_DIR / "summary.csv", newline="") as f:
            for row in csv.DictReader(f):
                _summary_cache.setdefault(row["PDB"], []).append(row)
    return _summary_cache.get(pdb_id, [])


def get_chain_atoms(pdb_id: str) -> tuple[list["Atom"], list["Atom"]] | None:
    """Shared by compute_interface_labels() below and data/train_labels.py (which
    needs the same antigen-residue universe but overlays AACDB's contacts instead of
    computing distances itself — reused rather than duplicated so the auth-vs-label
    chain-id logic only needs to be correct in one place). Returns
    (antibody_heavy_atoms, antigen_heavy_atoms), or None if the structure/summary row
    is missing."""
    cif_path = SABDAB_DIR / "structures" / pdb_id / f"{pdb_id}_sabdab.cif"
    rows = _load_summary_rows(pdb_id)
    if not cif_path.exists() or not rows:
        return None

    antibody_chains, antigen_chains = set(), set()
    for row in rows:
        if row["Hchain"].strip():
            antibody_chains.add(row["Hchain"].strip())
        if row["Lchain"].strip():
            antibody_chains.add(row["Lchain"].strip())
        if row["antigen_chain"].strip():
            antigen_chains.update(row["antigen_chain"].split("|"))

    atoms = parse_atom_site(cif_path)
    antibody_atoms = [a for a in atoms if a.auth_asym_id in antibody_chains and a.type_symbol != "H"
                       and a.auth_comp_id in STANDARD_RESIDUES]
    antigen_atoms = [a for a in atoms if a.auth_asym_id in antigen_chains and a.type_symbol != "H"
                      and a.auth_comp_id in STANDARD_RESIDUES]
    return antibody_atoms, antigen_atoms


def _antigen_contacts(antibody_atoms: list["Atom"], antigen_atoms: list["Atom"],
                       threshold: float) -> list[tuple["Atom", bool]]:
    """Shared distance computation behind both key conventions below -- same logic,
    just keyed differently by the two callers (auth for eval-label reporting, label_seq
    for cross-referencing against BoltzGen design outputs, which renumber sequentially
    and don't preserve auth numbering).

    Real bug hit while expanding past dev.txt/test.txt's ~1,437 structures to the full
    train_era (8,072): the original dense pairwise-distance matrix
    (n_antigen x n_antibody x 3) allocated 338 GiB on one large multi-copy asymmetric
    unit (142,020 x 106,560 atoms) and crashed. dev/test apparently never hit a
    structure this large. Fixed with a KD-tree radius query (scipy, already an
    installed transitive dep) -- same "any antibody heavy atom within threshold"
    semantics, but memory bounded regardless of structure size."""
    if not antibody_atoms or not antigen_atoms:
        return [(a, False) for a in antigen_atoms]

    antibody_xyz = np.array([a.xyz for a in antibody_atoms])
    antigen_xyz = np.array([a.xyz for a in antigen_atoms])
    antibody_tree = cKDTree(antibody_xyz)
    counts = antibody_tree.query_ball_point(antigen_xyz, r=threshold, return_length=True)
    within = counts > 0
    return list(zip(antigen_atoms, within.tolist()))


def compute_interface_labels(pdb_id: str, threshold: float = THRESHOLD) -> dict[tuple[str, str, str], bool]:
    """Returns {(auth_asym_id, auth_seq_id, auth_comp_id): is_epitope_residue}, for
    every antigen residue with at least one heavy atom in the structure (both True and
    False entries present — False means "not observed within threshold", not
    "confirmed non-epitope", matching the positive-unlabeled framing documented in
    PLAN.md)."""
    chains = get_chain_atoms(pdb_id)
    if chains is None:
        return {}
    antibody_atoms, antigen_atoms = chains

    labels: dict[tuple[str, str, str], bool] = {}
    for atom, is_contact in _antigen_contacts(antibody_atoms, antigen_atoms, threshold):
        key = (atom.auth_asym_id, atom.auth_seq_id, atom.auth_comp_id)
        labels[key] = labels.get(key, False) or is_contact
    return labels


def compute_interface_labels_by_label_seq(pdb_id: str, threshold: float = THRESHOLD) -> dict[int, bool]:
    """Same distance logic as compute_interface_labels(), keyed by antigen
    label_seq_id (entity-sequence position, 1-based) instead of auth numbering --
    needed for eval/downstream_eval.py's contact-overlap comparison against BoltzGen
    design-output CIFs. Verified empirically (not assumed): a design's antigen chain,
    included directly from the original structure via `include:`, keeps the exact same
    resolved-residue order and identity at each label_seq_id position as the original
    structure (checked pdb_000010gh: 1006/1006 positions, identical residue names in
    order) -- BoltzGen renumbers auth_seq_id sequentially from 1 in its output, which
    does NOT match the original structure's auth numbering, so auth-keyed labels can't
    be used for this comparison."""
    chains = get_chain_atoms(pdb_id)
    if chains is None:
        return {}
    antibody_atoms, antigen_atoms = chains

    labels: dict[int, bool] = {}
    for atom, is_contact in _antigen_contacts(antibody_atoms, antigen_atoms, threshold):
        if not atom.label_seq_id.isdigit():
            continue
        key = int(atom.label_seq_id)
        labels[key] = labels.get(key, False) or is_contact
    return labels


def parse_atom_site_generic(cif_path: Path) -> list["Atom"]:
    """Header-driven _atom_site parser (builds a column-name -> index map from the
    `_atom_site.*` header lines instead of hardcoding positions), for CIFs whose layout
    isn't guaranteed to match SAbDab's fixed 21-field layout that parse_atom_site()
    above targets. Needed for BoltzGen's own design-output CIFs, verified to use a
    different, shorter 19-field layout with no separate auth_atom_id/auth_comp_id
    columns (only auth_seq_id/auth_asym_id) -- those fall back to the label_
    equivalents, which are identical for a freshly-generated structure with no
    alternate residue-naming or altlocs beyond what's already filtered below."""
    lines = cif_path.read_text().splitlines()
    columns: list[str] = []
    for line in lines:
        if line.startswith("_atom_site."):
            columns.append(line.split(".", 1)[1].strip())
        elif columns:
            break  # header block ends at the first non-_atom_site. line after it starts
    col_idx = {name: i for i, name in enumerate(columns)}

    def get(f: list[str], name: str, fallback: str | None = None) -> str | None:
        idx = col_idx.get(name, col_idx.get(fallback) if fallback else None)
        return f[idx] if idx is not None else None

    atoms = []
    for line in lines:
        if not ATOM_LINE_RE.match(line):
            continue
        f = line.split()
        if len(f) < len(columns):
            continue
        alt_id = get(f, "label_alt_id")
        if alt_id not in (None, ".", "A"):
            continue
        atoms.append(Atom(
            type_symbol=get(f, "type_symbol"),
            atom_name=get(f, "auth_atom_id", "label_atom_id"),
            label_asym_id=get(f, "label_asym_id"),
            label_seq_id=get(f, "label_seq_id"),
            auth_asym_id=get(f, "auth_asym_id"),
            auth_seq_id=get(f, "auth_seq_id", "label_seq_id"),
            auth_comp_id=get(f, "auth_comp_id", "label_comp_id"),
            xyz=(float(get(f, "Cartn_x")), float(get(f, "Cartn_y")), float(get(f, "Cartn_z"))),
        ))
    return atoms


def _parse_aacdb_labels(pdb_id: str) -> dict[tuple[str, str], bool] | None:
    """For the sanity check only: AACDB files are named <PDB>_<chains>_..., PDB id is
    uppercase 4-char; returns {(chain, auth_seq_id): True} for antigen-side residues
    (the file's own "antigen" column, already self-labeled).

    A PDB entry commonly has *multiple* AACDB files — one per distinct antibody copy in
    the asymmetric unit (verified: 1,990 of 3,628 train_era/AACDB-overlapping
    structures have more than one). An earlier version of this function took only the
    first glob match, which silently compared against a single antibody instance while
    compute_interface_labels() unions antigen contacts across *every* summary.csv row
    (every instance) for that PDB — an apples-to-oranges mismatch that manifested as
    misleadingly low Jaccard scores for exactly the multi-instance structures, not a
    real labeler bug. Fixed by unioning across all matching files here too."""
    matches = list(AACDB_DIR.glob(f"interacting_res_distance/{pdb_id.upper()}_*_interacting_residues_distance.txt"))
    if not matches:
        return None
    labels = {}
    for match in matches:
        with open(match) as f:
            next(f)  # header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                _, antigen = parts[0], parts[1]
                chain, resname_num = antigen.split(":")
                num = "".join(ch for ch in resname_num if ch.isdigit())
                labels[(chain, num)] = True
    return labels


def sanity_check() -> None:
    """Compares this labeler's output against AACDB's on train_era structures where
    both exist, per PLAN.md's verification step."""
    with open(REPO_ROOT / "databases" / "splits" / "train_era.txt") as f:
        train_era = {p.replace("pdb_0000", "").upper() for p in f.read().split()}

    aacdb_pdbs = set()
    for p in AACDB_DIR.glob("interacting_res_distance/*_interacting_residues_distance.txt"):
        aacdb_pdbs.add(p.name.split("_")[0])

    overlap = sorted(train_era & aacdb_pdbs)
    print(f"[sanity_check] {len(overlap)} train_era structures with AACDB annotations", file=sys.stderr)

    jaccards = []
    for pdb in overlap[:200]:  # cap for speed; this is a sanity check, not full eval
        pdb_id_full = f"pdb_0000{pdb.lower()}"
        ours = compute_interface_labels(pdb_id_full)
        aacdb = _parse_aacdb_labels(pdb)
        if not ours or not aacdb:
            continue
        our_positive = {(c, n) for (c, n, _), v in ours.items() if v}
        aacdb_positive = set(aacdb.keys())
        if not our_positive and not aacdb_positive:
            continue
        union = our_positive | aacdb_positive
        inter = our_positive & aacdb_positive
        jaccards.append(len(inter) / len(union) if union else 0.0)

    if jaccards:
        print(f"[sanity_check] n={len(jaccards)}, mean Jaccard={sum(jaccards) / len(jaccards):.3f}, "
              f"min={min(jaccards):.3f}, max={max(jaccards):.3f}", file=sys.stderr)
    else:
        print("[sanity_check] no comparable structures found", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdb_id", nargs="?", help="e.g. pdb_0000010bt")
    parser.add_argument("--sanity-check", action="store_true")
    args = parser.parse_args()

    if args.sanity_check:
        sanity_check()
        return

    labels = compute_interface_labels(args.pdb_id)
    n_positive = sum(labels.values())
    print(f"{args.pdb_id}: {len(labels)} antigen residues, {n_positive} interface residues")
    for key, val in sorted(labels.items()):
        if val:
            print(" ", key)


if __name__ == "__main__":
    main()
