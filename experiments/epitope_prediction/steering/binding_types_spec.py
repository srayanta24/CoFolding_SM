#!/usr/bin/env python3
"""Predicted epitope -> BoltzGen design-spec `binding_types` string.

Uses **Model D** (a faithful fork of EpiFormer's antigen-branch encoder, retrained
antibody-agnostic on our data) — selected after a 4-way comparison on a genuinely
held-out test set (see experiments/epitope_prediction/README.md sec 2.6): Model D beats
Model A (AUC 0.876 vs 0.867, calibrated Brier 0.054 vs 0.060, precision@recall~0.3 0.937
vs 0.862) on every metric once given a training recipe suited to its larger size, and
Model A previously beat Models B/C on the same grounds Model A now loses to D. Superseded
Model A here for exactly the same reason Model A was originally chosen over B: it's the
model that actually wins the held-out comparison, not the one used historically.

Verified format (src/boltzgen/src/boltzgen/data/parse/schema.py:1066-1090): `binding_types`
is a single string, one character per residue, in the SAME order as the entity's declared
sequence (`U`nspecified / `B`inding / `N`ot-binding), padded with `U` if shorter than the
sequence. Positions are indexed by mmCIF `label_seq_id` (1-based, sequential in the
entity's declared sequence -- NOT auth_seq_id, which can have gaps/insertion codes) so
that unresolved residues (no coordinates, no prediction) correctly default to `U` at
their real position rather than silently shifting everything after them.

Adaptive selection (per the confidence -> residue-count design in PLAN.md sec 4): rank
antigen residues by ensemble-mean propensity, walk down the ranked list, stop adding to
the `B` set once either propensity or confidence drops below its floor. If nothing
clears both floors, return an all-`U` string -- skip epitope conditioning entirely for a
target the model has no real signal on, rather than forcing a low-confidence guess.

Usage:
    python3 experiments/epitope_prediction/steering/binding_types_spec.py <pdb_id>
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
from features import build_multirelational_features, group_residues  # noqa: E402
from gnn import ensemble_predict, load_ensemble  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from interface_labels import get_chain_atoms  # noqa: E402

PROPENSITY_FLOOR = 0.3
CONFIDENCE_FLOOR = 0.85


def predict_epitope(pdb_id: str, device: str | None = None) -> list[dict] | None:
    """Returns [{"label_seq_id": int, "auth_key": (chain, seq, comp), "propensity":
    float, "confidence": float}, ...] sorted by descending propensity, or None if the
    structure has no usable antigen."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    geo = build_multirelational_features(pdb_id)
    if geo is None:
        return None

    chains = get_chain_atoms(pdb_id)
    _, antigen_atoms = chains
    residues = group_residues(antigen_atoms)
    # geo["residue_keys"] and `residues` are built from the identical group_residues()
    # call chain (build_multirelational_features calls build_geometric_features
    # internally, which calls group_residues() too), so they're in the same order --
    # verified by construction, not just assumed.
    label_seq_ids = [res_atoms[0].label_seq_id for _, res_atoms in residues]

    models = load_ensemble("D", device)
    entry = {
        "node_scalars": geo["node_scalars"],
        "backbone_coords": geo["backbone_coords"],
        "edges_by_relation": geo["edges_by_relation"],
    }
    propensity, confidence = ensemble_predict(models, entry, device)

    predictions = [
        {
            "label_seq_id": int(lsid) if lsid.isdigit() else None,
            "auth_key": geo["residue_keys"][i],
            "propensity": propensity[i].item(),
            "confidence": confidence[i].item(),
        }
        for i, lsid in enumerate(label_seq_ids)
    ]
    predictions = [p for p in predictions if p["label_seq_id"] is not None]
    predictions.sort(key=lambda p: -p["propensity"])
    return predictions


def select_binding_residues(predictions: list[dict], propensity_floor: float = PROPENSITY_FLOOR,
                             confidence_floor: float = CONFIDENCE_FLOOR) -> set[int]:
    """Walks the propensity-ranked list, keeps adding label_seq_ids to the selected set
    until either floor is breached (predictions are pre-sorted by descending
    propensity, so this is a single pass, not a search)."""
    selected = set()
    for p in predictions:
        if p["propensity"] < propensity_floor or p["confidence"] < confidence_floor:
            break
        selected.add(p["label_seq_id"])
    return selected


def build_binding_types_string(predictions: list[dict], selected: set[int]) -> str:
    """`U`/`B` string, 1-indexed label_seq_id -> position. Length = max observed
    label_seq_id (a slight undercount is possible if the C-terminal-most residues are
    unresolved in the crystal structure and thus never appear in `predictions` at all --
    an acceptable, documented simplification: those positions would be `U` regardless,
    the same as everything else outside `selected`).

    Only valid for INLINE `protein:` entities with an explicit `sequence:` field
    (schema.py's string-parsing path, data/parse/schema.py:1066-1090). **Not** valid
    for `file:`-based entities (loading a real structure, which is what
    eval/downstream_eval.py actually uses) -- those need the structured range-list
    format instead, see build_binding_range_spec()."""
    if not predictions:
        return ""
    max_pos = max(p["label_seq_id"] for p in predictions)
    chars = ["U"] * max_pos
    for pos in selected:
        chars[pos - 1] = "B"
    return "".join(chars)


def build_binding_range_spec(selected: set[int]) -> str:
    """Structured range-list format for `file:`-based entities' `binding_types:` field
    -- verified against BoltzGen's own reference example
    (data/boltzgen_examples/repo/example/design_spec_showcasing_all_functionalities.yaml):
    `binding_types: [{chain: {id: ..., binding: "5..7,13"}}]`. Collapses consecutive
    label_seq_ids into ranges (`68..70`) and leaves singletons bare (`13`), matching the
    example's own mixed syntax -- not required (bare comma-separated numbers alone are
    valid too), but more compact and readable for a real epitope call."""
    if not selected:
        return ""
    positions = sorted(selected)
    ranges = []
    start = prev = positions[0]
    for pos in positions[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        ranges.append(f"{start}..{prev}" if start != prev else str(start))
        start = prev = pos
    ranges.append(f"{start}..{prev}" if start != prev else str(start))
    return ",".join(ranges)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdb_id")
    parser.add_argument("--propensity-floor", type=float, default=PROPENSITY_FLOOR)
    parser.add_argument("--confidence-floor", type=float, default=CONFIDENCE_FLOOR)
    args = parser.parse_args()

    predictions = predict_epitope(args.pdb_id)
    if predictions is None:
        print(f"{args.pdb_id}: no usable antigen")
        return

    selected = select_binding_residues(predictions, args.propensity_floor, args.confidence_floor)
    spec = build_binding_types_string(predictions, selected)
    print(f"{args.pdb_id}: {len(selected)}/{len(predictions)} residues selected as B "
          f"(propensity>={args.propensity_floor}, confidence>={args.confidence_floor})")
    print(f"binding_types: {spec}")
    top5 = sorted(predictions, key=lambda p: -p["propensity"])[:5]
    print("top 5 by propensity:", [(p["auth_key"], round(p["propensity"], 3), round(p["confidence"], 3)) for p in top5])


if __name__ == "__main__":
    main()
