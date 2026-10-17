#!/usr/bin/env python3
"""Assemble the labelled scoring-calibration benchmark (PLAN.md milestone 2).

Produces one tidy manifest, `sets/manifest.tsv`, with one row per scoring job. The
four "sets" described in PLAN.md sec 5 are *queries over this table*, not separate
files — sets B and C are nearly the same complexes playing two analysis roles
(memorisation gap vs. dose-response), so splitting them into separate manifests
would duplicate rows and invite them drifting apart.

Stdlib only, matching the rest of this project's orchestration layer (see
experiments/README.md's note on deliberately not pulling gemmi/biotite into a
bare-python3 zero-venv path).

Sets produced
-------------
clean_positive
    Protein-antigen Fabs drawn from `databases/splits/{dev,test}.txt` — post-cutoff
    and antigen-sequence-cluster-clean, so BoltzGen and Boltz-2 did not train on
    them. These are confirmed binders by construction: every SAbDab entry is a
    crystallised antibody-antigen complex, so no Kd label is needed to know it binds.
    This is the set that sets the honest calibration.
contaminated_positive
    The Kd-labelled affinity pairs. 99.6% sit inside BoltzGen's training window, so
    these are scored to *measure the memorisation gap* against clean_positive, and
    to test dose-response against log(Kd). Never used to set a threshold directly.
hard_negative
    AbDesign-DB mutants where binding is measurably destroyed on an otherwise
    identical complex (ddg >= +2.0 kcal/mol, or ELISA mut/wt ratio <= 0.1).
matched_positive
    AbDesign-DB mutants on the *same* parent complexes where binding is retained
    (|ddg| < 0.5, or ELISA ratio >= 0.8). The control that makes hard_negative
    interpretable — without it, any separation could just be parent-complex identity.

Real bug caught while building this, worth keeping visible
----------------------------------------------------------
The first version read sequences from the mmCIF's `_entity_poly` block, reusing
`databases/src/build_splits.py`'s `parse_entity_poly`. That is the **full declared
construct** sequence, and it silently included signal peptides: `pdb_000011zw`'s
heavy chain came out as `MPMGSLQPLATLYLLGMLVASVLA|EVQLVESGGG...` and its antigen as
`MGILPSPGMPALLSLVSLLSVLLMGCVA|ETGTPELEVVQ...`. Cofolding those would have produced
disordered junk at both termini, depressed every confidence metric, and inflated the
residue count (and therefore the GPU cost) — while calibrating against something the
design pipeline never sees, since that pipeline conditions on a *structure*.

Fixed by building sequences from `_atom_site` instead: only residues with real
resolved coordinates, which is what is physically in the crystal. This is the same
conclusion `experiments/epitope_prediction/model/esm2_features.py` reached
independently, for the same reason, and its module docstring records it.

Consequence for the analysis, stated here because it constrains which comparisons
are legitimate: structural sets (clean_positive, contaminated_positive) now share
one extraction path, so the memorisation-gap comparison between them is
provenance-clean. The mutant sets necessarily use AbDesign-DB's own curated
sequences — there is no crystal structure of a point mutant — so **hard_negative
must be compared against matched_positive, not against clean_positive.** Both
mutant sets share a provenance; crossing the boundary would confound sequence
provenance with the thing being measured.

Three verified facts this script depends on
-------------------------------------------
1. `abdesign_db/datasets_mut.csv`'s heavy_sequence/light_sequence columns hold the
   **post-mutation** sequence, not the wild type. Checked directly across all 1,303
   rows: the residue at `mutation_pos` (1-based) matches `new_residue` in 1303 cases
   and `orig_residue` in 0. So no mutation needs applying here, and the usual
   PDB-numbering trap is avoided entirely. Do not "fix" this by applying the
   mutation again.
2. `summary.csv`'s Hchain/Lchain/antigen_chain are **auth_asym_id** — the gotcha
   documented at length in `experiments/epitope_prediction/PLAN.md` sec 7. The
   `_atom_site` parser below is therefore keyed by auth_asym_id to match.
3. 491 of the 493 Kd-labelled affinity pairs have a SAbDab CIF on disk at
   `structures/pdb_0000<id>/`, so the contaminated set is built from structures
   too rather than from the affinity CSV's own sequence columns. The CSV supplies
   only the Kd label.

Why a local _atom_site parser instead of importing one
------------------------------------------------------
`epitope_prediction/data/interface_labels.py` has a good one, but it imports numpy
and scipy, and `model/features.py` adds torch and freesasa — all of which live in
`.venvs/epitope-prediction/`. Importing them would force this script to run inside
that venv and break the bare-python3, stdlib-only convention the rest of the
orchestration layer keeps (experiments/README.md states this deliberately). The
parser here needs residue identity and ordering only, no coordinates, so it is a
fraction of the size.

Deliberate v1 scope limits
--------------------------
- **Fabs only** (both Hchain and Lchain present). Nanobodies are excluded even
  though BoltzGen's own wet-lab validation is nanobody-heavy, because the pipeline
  being calibrated here runs `antibody-anything` against Fab scaffolds. Add a `vhh`
  arm later if the nanobody protocol is ever adopted; don't silently pool them.
- **Protein antigens only** (`antigen_type == PROTEIN` exactly). Peptide, hapten,
  sugar, ion and mixed-type antigens are dropped: a hapten "epitope" is not the same
  scoring problem, and mixed types make the contact metrics ambiguous.
- **AB-Bind is not read directly.** 8 of its 32 complexes are not antibody-antigen
  at all and 4 more are homology models (PLAN.md sec 5). AbDesign-DB already
  re-processes the usable AB-Bind rows with sequences attached, so it is the safer
  source for the same data.

Usage
-----
    python3 experiments/scoring_calibration/build_sets.py
    python3 experiments/scoring_calibration/build_sets.py --pilot 300
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATABASES = REPO_ROOT / "databases"
SABDAB = DATABASES / "sabdab"
SPLITS = DATABASES / "splits"
OUT_DIR = Path(__file__).resolve().parent / "sets"

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    # Non-standard but real polypeptide residues, same allowlist spirit as
    # interface_labels.py's STANDARD_RESIDUES. MSE (selenomethionine) is extremely
    # common in crystallography and is chemically methionine for our purposes.
    "MSE": "M", "SEC": "U", "PYL": "O", "UNK": "X",
}

# Cost guard. Cofolding time scales superlinearly with total residues (BoltzGen paper
# Fig. 14), and a 2,000-residue antigen would dominate the pilot's GPU budget on its
# own. 1,200 total residues comfortably fits a Fab (~450) plus a large antigen.
MAX_TOTAL_RESIDUES = 1200
MIN_ANTIGEN_RESIDUES = 25

FIELDS = [
    "entry_id", "set", "source", "pdb", "split", "fmt",
    "label", "kd_molar", "ddg_kcal", "elisa_ratio",
    "n_res_heavy", "n_res_light", "n_res_antigen", "n_res_total",
    "heavy_seq", "light_seq", "antigen_seq", "note",
]


def rank_key(entry_id: str) -> str:
    """Deterministic ordering for sampling — SHA-256 of the id, not a seeded RNG.

    Same convention as build_splits.py's dev/test slice, so a rebuild on another
    machine reproduces the identical pilot subset.
    """
    return hashlib.sha256(entry_id.encode()).hexdigest()


def load_splits() -> dict[str, str]:
    out = {}
    for name in ("train_era", "dev", "test"):
        path = SPLITS / f"{name}.txt"
        for line in path.read_text().splitlines():
            if line.strip():
                out[line.strip()] = name
    return out


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def resolved_chain_sequences(cif_path: Path) -> dict[str, str]:
    """auth_asym_id -> one-letter sequence of residues that have real coordinates.

    Header-driven (builds a column-name -> index map from the `_atom_site.*` header
    lines) rather than assuming a fixed column layout, matching the defensive style
    of parse_atom_site_generic() in the epitope code.

    Deliberately NOT the declared `_entity_poly` sequence — see the module docstring's
    bug note. Residues are ordered by label_seq_id (the canonical sequential index
    within the entity), falling back to auth_seq_id where label_seq_id is absent or
    non-numeric. Alternate conformations are collapsed to the primary altloc, the
    same fix interface_labels.py's parse_atom_site had to make.
    """
    lines = cif_path.read_text().splitlines()
    columns: list[str] = []
    for line in lines:
        if line.startswith("_atom_site."):
            columns.append(line.split(".", 1)[1].strip())
        elif columns:
            break
    if not columns:
        return {}
    idx = {name: i for i, name in enumerate(columns)}

    def col(fields: list[str], name: str, fallback: str | None = None) -> str | None:
        i = idx.get(name)
        if i is None and fallback:
            i = idx.get(fallback)
        return fields[i] if i is not None and i < len(fields) else None

    # chain -> {sort_key: comp_id}; first atom of a residue wins
    seen: dict[str, dict[tuple, str]] = {}
    for line in lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        f = line.split()
        alt = col(f, "label_alt_id")
        if alt not in (None, ".", "?", "A"):
            continue  # secondary altloc — skip, keep only the primary
        comp = col(f, "label_comp_id", "auth_comp_id")
        chain = col(f, "auth_asym_id", "label_asym_id")
        if comp is None or chain is None or comp not in AA3_TO_1:
            continue
        lseq = col(f, "label_seq_id")
        aseq = col(f, "auth_seq_id")
        try:
            key = (0, int(lseq))
        except (TypeError, ValueError):
            try:
                key = (1, int(aseq))
            except (TypeError, ValueError):
                continue
        seen.setdefault(chain, {}).setdefault(key, comp)

    return {
        chain: "".join(AA3_TO_1[c] for _k, c in sorted(res.items()))
        for chain, res in seen.items()
    }


def build_clean_positives(splits: dict[str, str]) -> tuple[list[dict], Counter]:
    """Protein-antigen Fabs from dev+test, resolved sequences read out of the mmCIF."""
    stats = Counter()
    rows = read_csv(SABDAB / "summary.csv")

    # One antibody copy per PDB. A PDB can list several (multiple Fabs in the
    # asymmetric unit); they are redundant for calibration purposes and would
    # over-weight those entries.
    by_pdb: dict[str, dict] = {}
    for r in rows:
        pdb = r["PDB"].strip()
        if splits.get(pdb) not in ("dev", "test"):
            continue
        if r["antigen_type"].strip().upper() != "PROTEIN":
            stats["skip_antigen_type"] += 1
            continue
        h, l = r["Hchain"].strip(), r["Lchain"].strip()
        if h in ("", "NA") or l in ("", "NA"):
            stats["skip_not_fab"] += 1
            continue
        by_pdb.setdefault(pdb, r)

    out = []
    for pdb, r in sorted(by_pdb.items()):
        cif = SABDAB / "structures" / pdb / f"{pdb}_sabdab.cif"
        if not cif.exists():
            stats["skip_no_cif"] += 1
            continue
        try:
            chain_seq = resolved_chain_sequences(cif)
        except Exception:
            stats["skip_cif_parse_error"] += 1
            continue
        h, l = r["Hchain"].strip(), r["Lchain"].strip()
        # antigen_chain is |-delimited for multi-chain antigens (build_splits.py's
        # documented gotcha). Take the first chain only: a multi-chain antigen is a
        # different scoring problem and would not match the single-antigen-entity
        # shape the design pipeline produces.
        ag_chains = [c.strip() for c in r["antigen_chain"].split("|") if c.strip()]
        if len(ag_chains) != 1:
            stats["skip_multichain_antigen"] += 1
            continue
        hs, ls, ags = chain_seq.get(h), chain_seq.get(l), chain_seq.get(ag_chains[0])
        if not (hs and ls and ags):
            stats["skip_chain_has_no_resolved_residues"] += 1
            continue
        if len(ags) < MIN_ANTIGEN_RESIDUES:
            stats["skip_antigen_too_short"] += 1
            continue
        total = len(hs) + len(ls) + len(ags)
        if total > MAX_TOTAL_RESIDUES:
            stats["skip_too_large"] += 1
            continue
        out.append({
            "entry_id": f"clean__{pdb}", "set": "clean_positive",
            "source": "sabdab_splits", "pdb": pdb, "split": splits[pdb], "fmt": "fab",
            "label": "binder", "kd_molar": "", "ddg_kcal": "", "elisa_ratio": "",
            "n_res_heavy": len(hs), "n_res_light": len(ls), "n_res_antigen": len(ags),
            "n_res_total": total,
            "heavy_seq": hs, "light_seq": ls, "antigen_seq": ags,
            "note": f"crystallised Ab-Ag complex; antigen chain {ag_chains[0]}",
        })
        stats["kept"] += 1
    return out, stats


def build_contaminated_positives(splits: dict[str, str]) -> tuple[list[dict], Counter]:
    """Kd-labelled pairs, built from the SAbDab CIF so provenance matches set A.

    The affinity CSV carries its own sequence columns, but using them would make the
    memorisation-gap comparison (A vs B) confound sequence provenance with the thing
    being measured — the CSV's curated sequences are not the same strings the
    resolved-coordinate extraction produces. 491 of the 493 pairs have a CIF on
    disk, so the CSV is used only for the Kd label and the chain assignment comes
    from summary.csv exactly as it does for set A.
    """
    stats = Counter()
    kd_by_pdb: dict[str, tuple[float, str]] = {}
    for r in read_csv(SABDAB / "affinity" / "antibody_affinity_protein_sabdab.csv"):
        short = r["Antibody_ID"].strip().lower()
        try:
            kd_by_pdb[f"pdb_0000{short}"] = (float(r["Y"]), r.get("Antigen_ID", "")[:48])
        except Exception:
            stats["skip_unparseable_kd"] += 1

    summary_by_pdb: dict[str, dict] = {}
    for r in read_csv(SABDAB / "summary.csv"):
        pdb = r["PDB"].strip()
        if pdb in kd_by_pdb and r["Hchain"].strip() not in ("", "NA") \
                and r["Lchain"].strip() not in ("", "NA"):
            summary_by_pdb.setdefault(pdb, r)

    out = []
    for pdb, (kd, ag_name) in sorted(kd_by_pdb.items()):
        r = summary_by_pdb.get(pdb)
        if r is None:
            stats["skip_no_fab_row_in_summary"] += 1
            continue
        cif = SABDAB / "structures" / pdb / f"{pdb}_sabdab.cif"
        if not cif.exists():
            stats["skip_no_cif"] += 1
            continue
        try:
            chain_seq = resolved_chain_sequences(cif)
        except Exception:
            stats["skip_cif_parse_error"] += 1
            continue
        ag_chains = [c.strip() for c in r["antigen_chain"].split("|") if c.strip()]
        if len(ag_chains) != 1:
            stats["skip_multichain_antigen"] += 1
            continue
        hs = chain_seq.get(r["Hchain"].strip())
        ls = chain_seq.get(r["Lchain"].strip())
        ags = chain_seq.get(ag_chains[0])
        if not (hs and ls and ags):
            stats["skip_chain_has_no_resolved_residues"] += 1
            continue
        if len(ags) < MIN_ANTIGEN_RESIDUES:
            stats["skip_antigen_too_short"] += 1
            continue
        total = len(hs) + len(ls) + len(ags)
        if total > MAX_TOTAL_RESIDUES:
            stats["skip_too_large"] += 1
            continue
        split = splits.get(pdb, "not_in_splits")
        out.append({
            "entry_id": f"aff__{pdb}", "set": "contaminated_positive",
            "source": "sabdab_affinity+structure", "pdb": pdb, "split": split,
            "fmt": "fab", "label": "binder",
            "kd_molar": f"{kd:.3e}", "ddg_kcal": "", "elisa_ratio": "",
            "n_res_heavy": len(hs), "n_res_light": len(ls), "n_res_antigen": len(ags),
            "n_res_total": total,
            "heavy_seq": hs, "light_seq": ls, "antigen_seq": ags,
            "note": f"Kd-labelled; antigen={ag_name}",
        })
        stats[f"split_{split}"] += 1
        stats["kept"] += 1
    return out, stats


def build_mutant_sets() -> tuple[list[dict], Counter]:
    """Hard negatives + matched positives from AbDesign DB's reprocessed mutants.

    Sequences here are already post-mutation (verified, see module docstring).
    """
    stats = Counter()
    rows = read_csv(DATABASES / "abdesign_db" / "datasets_mut.csv")
    out = []
    for i, r in enumerate(rows):
        hs = r.get("heavy_sequence", "").strip()
        ls = r.get("light_sequence", "").strip()
        ags = r.get("antigen_sequence", "").strip()
        if not (hs and ls and ags) or len(ags) < MIN_ANTIGEN_RESIDUES:
            stats["skip_bad_sequence"] += 1
            continue
        atype = r.get("affinity_type", "").strip()
        try:
            aval = float(r["affinity"])
        except Exception:
            stats["skip_no_affinity"] += 1
            continue

        set_name = label = None
        ddg = elisa = ""
        if atype == "ddg":
            ddg = f"{aval:.3f}"
            if aval >= 2.0:
                set_name, label = "hard_negative", "binding_destroyed"
            elif abs(aval) < 0.5:
                set_name, label = "matched_positive", "binding_retained"
        elif atype == "elisa_mut_to_wt_ratio":
            elisa = f"{aval:.4f}"
            if aval <= 0.1:
                set_name, label = "hard_negative", "binding_destroyed"
            elif aval >= 0.8:
                set_name, label = "matched_positive", "binding_retained"
        if set_name is None:
            stats["skip_ambiguous_middle"] += 1
            continue

        total = len(hs) + len(ls) + len(ags)
        if total > MAX_TOTAL_RESIDUES:
            stats["skip_too_large"] += 1
            continue
        parent = r.get("pdb_name", "").strip()
        mut = (f"{r.get('mutation_chain_id','')}:{r.get('orig_residue','')}"
               f"{r.get('mutation_pos','')}{r.get('new_residue','')}")
        out.append({
            "entry_id": f"mut__{parent}__{mut}__{i}", "set": set_name,
            "source": f"abdesign_db/{r.get('dataset','')}", "pdb": parent,
            "split": "", "fmt": "fab", "label": label,
            "kd_molar": "", "ddg_kcal": ddg, "elisa_ratio": elisa,
            "n_res_heavy": len(hs), "n_res_light": len(ls), "n_res_antigen": len(ags),
            "n_res_total": total,
            "heavy_seq": hs, "light_seq": ls, "antigen_seq": ags,
            "note": f"mutant {mut} ({atype}); sequences are post-mutation",
        })
        stats[f"kept_{set_name}"] += 1
    return out, stats


def pick_pilot(entries: list[dict], per_set: int) -> list[dict]:
    """Deterministic stratified subset: `per_set` entries from each set.

    For contaminated_positive, spread across the Kd range rather than taking a flat
    hash sample — the dose-response analysis (PLAN.md sec 7) needs coverage of the
    whole affinity span, and a uniform sample would over-represent the nM mode where
    half the data sits.
    """
    by_set = defaultdict(list)
    for e in entries:
        by_set[e["set"]].append(e)

    chosen = []
    for set_name, group in sorted(by_set.items()):
        if set_name == "clean_positive":
            # Split discipline, same rule as everywhere else in this project: the
            # pilot iterates on `dev` only. `test` is reserved for the final scaled
            # run, so that tuning decisions made off the pilot can't contaminate the
            # numbers we eventually report. This caps the pilot's clean arm at
            # however many dev entries survive filtering, which is small — accepted
            # deliberately rather than quietly padding it from test.
            group = [e for e in group if e["split"] == "dev"]
            ordered = sorted(group, key=lambda e: rank_key(e["entry_id"]))
            chosen.extend(ordered[:per_set])
        elif set_name == "contaminated_positive":
            graded = sorted(group, key=lambda e: float(e["kd_molar"]))
            n = min(per_set, len(graded))
            # even stride across the sorted-by-Kd list
            idxs = [round(i * (len(graded) - 1) / max(n - 1, 1)) for i in range(n)]
            chosen.extend(graded[i] for i in sorted(set(idxs)))
        else:
            ordered = sorted(group, key=lambda e: rank_key(e["entry_id"]))
            chosen.extend(ordered[:per_set])
    return chosen


def write_manifest(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t",
                           extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for e in sorted(entries, key=lambda e: (e["set"], e["entry_id"])):
            w.writerow(e)


def summarize(entries: list[dict]) -> dict:
    by_set = Counter(e["set"] for e in entries)
    res = [int(e["n_res_total"]) for e in entries]
    return {
        "total_entries": len(entries),
        "by_set": dict(by_set),
        "by_split": dict(Counter(e["split"] for e in entries if e["split"])),
        "residues": {
            "min": min(res), "max": max(res),
            "mean": round(sum(res) / len(res), 1),
            "total": sum(res),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pilot", type=int, default=100,
                    help="entries per set in the pilot manifest (default 100)")
    args = ap.parse_args()

    splits = load_splits()
    clean, s1 = build_clean_positives(splits)
    cont, s2 = build_contaminated_positives(splits)
    muts, s3 = build_mutant_sets()
    entries = clean + cont + muts

    print("=== build stats (skips are expected, not failures — see module docstring) ===")
    for name, st in (("clean_positive", s1), ("contaminated_positive", s2), ("mutants", s3)):
        print(f"  {name}:")
        for k, v in sorted(st.items()):
            print(f"      {k:<34} {v}")

    write_manifest(OUT_DIR / "manifest.tsv", entries)
    full = summarize(entries)

    pilot = pick_pilot(entries, args.pilot)
    write_manifest(OUT_DIR / "pilot.tsv", pilot)
    pil = summarize(pilot)

    (OUT_DIR / "summary.json").write_text(
        json.dumps({"full": full, "pilot": pil,
                    "max_total_residues": MAX_TOTAL_RESIDUES,
                    "min_antigen_residues": MIN_ANTIGEN_RESIDUES}, indent=2) + "\n")

    print("\n=== full manifest ===")
    print(json.dumps(full, indent=2))
    print("\n=== pilot manifest ===")
    print(json.dumps(pil, indent=2))
    print(f"\nwrote {OUT_DIR/'manifest.tsv'}")
    print(f"wrote {OUT_DIR/'pilot.tsv'}")
    print(f"wrote {OUT_DIR/'summary.json'}")


if __name__ == "__main__":
    main()
