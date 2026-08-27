#!/usr/bin/env python3
"""Downstream evaluation (PLAN.md sec 5's "the metric that actually matters"): does
conditioning BoltzGen's design stage on Model A's predicted epitope (via `binding_types`)
actually shift generated designs' real contact residues toward the true epitope,
compared to unconditioned generation on the same target?

Builds two design specs for a real dev.txt structure (using the actual experimental
SAbDab structure directly -- no need to re-predict via Boltz-2, we already have real
coordinates and real ground-truth epitope labels for these targets):
  - "conditioned": target entity has `binding_types` set from steering/binding_types_spec.py
  - "baseline": identical spec, no `binding_types` (BoltzGen picks its own contacts)

Chain-id gotcha under active verification: BoltzGen's own scaffold YAML comments say
`include: - chain: {id: ...}` matches **label** asym id (e.g.
"data/boltzgen_examples/repo/example/fab_scaffolds/adalimumab.6cr1.yaml": "Heavy chain
(label not auth): B") -- the OPPOSITE convention from summary.csv's antigen_chain /
databases/src/build_splits.py's entity_poly.pdbx_strand_id matching (verified auth
throughout this project's own SAbDab-derived CIFs). Reverse-engineering BoltzGen's exact
CIF-chain-resolution code wasn't conclusive from source alone -- this script builds the
spec with the label chain id and validates with `boltzgen check` (cheap, fast, and
exactly this project's own established "always validate before launching" convention,
SKILL.md) before trusting it, rather than assuming either convention.

Usage:
    python3 experiments/epitope_prediction/eval/downstream_eval.py <pdb_id>            # build + validate specs only
    python3 experiments/epitope_prediction/eval/downstream_eval.py <pdb_id> --launch    # + run both real campaigns
    python3 experiments/epitope_prediction/eval/downstream_eval.py <pdb_id> --compare   # after both campaigns finish
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from _common import WEIGHTS_DIR, venv_bin  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "steering"))
from binding_types_spec import build_binding_range_spec, predict_epitope, select_binding_residues  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from interface_labels import (  # noqa: E402
    SABDAB_DIR, _antigen_contacts, compute_interface_labels_by_label_seq, parse_atom_site_generic,
)

CONTACT_THRESHOLD = 5.0  # matches interface_labels.THRESHOLD; kept as its own constant
                          # here since this module compares against design outputs, not
                          # just SAbDab structures, and the two should stay independently
                          # adjustable even though they happen to agree today.

FAB_SCAFFOLDS = [
    "adalimumab.6cr1", "belimumab.5y9k", "dupilumab.6wgb", "golimumab.5yoy",
    "guselkumab.4m6m", "nirsevimab.5udc", "sarilumab.8iow", "secukinumab.6wio",
    "tezepelumab.5j13", "tralokinumab.5l6y", "ustekinumab.3hmw", "mab1.3h42",
    "necitumumab.6b3s", "crenezumab.5vzy",
]
BOLTZGEN_EXAMPLES_DIR = REPO_ROOT / "data" / "boltzgen_examples" / "repo" / "example"


def antigen_label_chain(pdb_id: str) -> str | None:
    """The label_asym_id for the antigen -- distinct from summary.csv's antigen_chain
    (auth-convention, verified). Derived from the same resolved atoms
    compute_interface_labels() already parses, taking the label_asym_id off any
    antigen atom (they're internally consistent within one physical chain)."""
    from interface_labels import get_chain_atoms

    chains = get_chain_atoms(pdb_id)
    if chains is None:
        return None
    _, antigen_atoms = chains
    if not antigen_atoms:
        return None
    return antigen_atoms[0].label_asym_id


def write_spec(pdb_id: str, out_dir: Path, binding_range: str | None) -> Path:
    """binding_range: BoltzGen's structured range syntax (e.g. "68..70,150"), from
    steering/binding_types_spec.py's build_binding_range_spec() -- NOT the raw U/B/N
    string, which only works for inline `protein:` entities, not `file:`-based ones
    like this (verified the hard way: a first version of this function dumped the raw
    string here and `boltzgen check` failed with a Python TypeError deep in schema.py,
    not a helpful validation message -- see this module's docstring)."""
    label_chain = antigen_label_chain(pdb_id)
    cif_name = f"{pdb_id}_sabdab.cif"
    out_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(SABDAB_DIR / "structures" / pdb_id / cif_name, out_dir / cif_name)

    # Computed relative to out_dir (not hardcoded): a first version hardcoded a fixed
    # "../../../../..." depth copied from scripts/design_binder.py's own out_root
    # (one level under data/designs/), which is the wrong depth for this script's
    # deeper eval/.downstream_runs/<pdb_id>/<variant>/ output layout -- boltzgen check
    # failed with FileNotFoundError before this fix.
    import os
    scaffold_rel_dir = os.path.relpath(BOLTZGEN_EXAMPLES_DIR / "fab_scaffolds", out_dir)
    scaffold_paths = [f"{scaffold_rel_dir}/{name}.yaml" for name in FAB_SCAFFOLDS]

    lines = ["entities:", "    - file:", f"        path: {cif_name}", "        include:",
              "            - chain:", f"                id: {label_chain}"]
    if binding_range:
        lines += ["        binding_types:", "            - chain:", f"                id: {label_chain}",
                   f"                binding: {binding_range}"]
    lines += ["    - file:", "        path:"]
    lines += [f"            - {p}" for p in scaffold_paths]

    spec_path = out_dir / f"{pdb_id}_spec.yaml"
    spec_path.write_text("\n".join(lines) + "\n")
    return spec_path


def validate_spec(spec_path: Path, out_dir: Path) -> bool:
    cmd = [venv_bin("boltzgen", "boltzgen"), "check", spec_path.name,
           "--output", str(out_dir / "check_output"), "--cache", str(WEIGHTS_DIR / "boltzgen")]
    result = subprocess.run(cmd, cwd=spec_path.parent)
    return result.returncode == 0


def run_campaign(spec_path: Path, num_designs: int = 50, budget: int = 5) -> None:
    # Skip-if-already-run guard: --launch always re-lists every variant in `specs`
    # (including "baseline", which never changes across which model produced the
    # conditioned spec) -- without this check, comparing a second model's conditioned
    # predictions on a pdb_id already evaluated once would silently re-run and burn
    # real GPU-hours re-doing an already-completed, already-reported baseline campaign.
    final_designs_dir = spec_path.parent / "run1" / "final_ranked_designs"
    if final_designs_dir.exists():
        print(f"[downstream_eval] {spec_path.parent.name}: run1 already complete "
              f"({final_designs_dir}) -- skipping, not re-running", file=sys.stderr)
        return
    cmd = [venv_bin("boltzgen", "boltzgen"), "run", spec_path.name,
           "--output", "run1", "--protocol", "antibody-anything",
           "--num_designs", str(num_designs), "--budget", str(budget),
           "--num_workers", "0", "--cache", str(WEIGHTS_DIR / "boltzgen")]
    print(f"[downstream_eval] launching: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=spec_path.parent)


def build_both_specs(pdb_id: str, conditioned_variant: str = "conditioned") -> dict[str, Path]:
    """conditioned_variant: which subdirectory name to write the conditioned spec under
    -- defaults to "conditioned" (the original Model A results), but pass e.g.
    "conditioned_D" to compare a different steering model's predictions without
    clobbering an earlier model's already-completed campaign in the same pdb_id
    directory. "baseline" (unconditioned) never varies by model, so it's always just
    "baseline" and is reused across every conditioned_variant for the same pdb_id."""
    out_root = Path(__file__).resolve().parent / ".downstream_runs" / pdb_id

    predictions = predict_epitope(pdb_id)
    if predictions is None:
        sys.exit(f"[downstream_eval] {pdb_id}: no usable antigen for epitope prediction")
    selected = select_binding_residues(predictions)
    if not selected:
        sys.exit(f"[downstream_eval] {pdb_id}: model selected zero residues above the "
                  f"confidence/propensity floor for this target -- pick a different dev.txt "
                  f"structure, this one has no strong enough epitope call to make a "
                  f"meaningful conditioned-vs-unconditioned comparison.")
    binding_range = build_binding_range_spec(selected)
    print(f"[downstream_eval] {pdb_id}: {len(selected)} residues selected -> "
          f"binding_types range: {binding_range}", file=sys.stderr)

    specs = {}
    for variant, bt in [(conditioned_variant, binding_range), ("baseline", None)]:
        variant_dir = out_root / variant
        spec_path = write_spec(pdb_id, variant_dir, bt)
        print(f"[downstream_eval] wrote {variant} spec: {spec_path}", file=sys.stderr)
        ok = validate_spec(spec_path, variant_dir)
        print(f"[downstream_eval] {variant} spec {'VALID' if ok else 'FAILED validation'}", file=sys.stderr)
        if not ok:
            sys.exit(f"[downstream_eval] {variant} spec failed boltzgen check -- see output above "
                      f"before trying --launch (likely the label-vs-auth chain-id gotcha in this "
                      f"file's own docstring)")
        specs[variant] = spec_path
    return specs


def design_contacts_by_label_seq(cif_path: Path, threshold: float = CONTACT_THRESHOLD) -> set[int]:
    """Antigen residues (keyed by label_seq_id, to match
    compute_interface_labels_by_label_seq()'s ground-truth keying) contacted by the
    designed antibody chains in one BoltzGen design-output CIF.

    Antigen chain identified as the largest chain by resolved-residue count, not a
    hardcoded letter: `boltzgen check`'s own output showed chain ids get renamed on
    conflict (observed "Renaming with {'A': 'C'}" when validating the specs earlier in
    this module), so a fixed "antigen = chain A" assumption isn't safe across variants
    or ranks. Length is reliable here because every FAB scaffold in FAB_SCAFFOLDS is
    ~100-140 residues while pdb_000010gh's antigen is 1006 -- an order of magnitude
    apart, not a close call."""
    atoms = [a for a in parse_atom_site_generic(cif_path) if a.type_symbol != "H"]
    by_chain: dict[str, list] = {}
    for a in atoms:
        by_chain.setdefault(a.label_asym_id, []).append(a)
    antigen_chain = max(by_chain, key=lambda c: len({a.label_seq_id for a in by_chain[c]}))
    antigen_atoms = by_chain[antigen_chain]
    antibody_atoms = [a for chain_id, chain_atoms in by_chain.items() if chain_id != antigen_chain
                       for a in chain_atoms]

    contacts = set()
    for atom, is_contact in _antigen_contacts(antibody_atoms, antigen_atoms, threshold):
        if is_contact and atom.label_seq_id.isdigit():
            contacts.add(int(atom.label_seq_id))
    return contacts


def _mean(values: list[float]) -> float:
    values = [v for v in values if v == v]  # drop NaN
    return sum(values) / len(values) if values else float("nan")


def compare_results(pdb_id: str, conditioned_variant: str = "conditioned") -> None:
    """Real ground-truth epitope (our own coordinate-based labeler) vs. each variant's
    top-5 ranked designs' actual contact residues (recomputed the same way, on each
    design's own refolded output CIF instead of the original antigen structure) --
    PLAN.md sec 5's "the metric that actually matters": does conditioning shift
    generated contacts toward the true epitope, compared to unconditioned generation?"""
    true_positive = {k for k, v in compute_interface_labels_by_label_seq(pdb_id).items() if v}
    print(f"[downstream_eval] {pdb_id}: {len(true_positive)} true epitope residues (ground truth)")

    out_root = Path(__file__).resolve().parent / ".downstream_runs" / pdb_id
    summaries = {}
    for variant in (conditioned_variant, "baseline"):
        run_dir = out_root / variant / "run1"
        design_dir = run_dir / "final_ranked_designs" / "final_5_designs"
        cifs = sorted(design_dir.glob("rank*.cif")) if design_dir.exists() else []
        if not cifs:
            print(f"[downstream_eval] {variant}: no ranked designs found at {design_dir} -- run with --launch first")
            continue

        print(f"[downstream_eval] {variant}: {len(cifs)} ranked designs")
        recalls, precisions, jaccards = [], [], []
        union_contacts: set[int] = set()
        for cif_path in cifs:
            contacts = design_contacts_by_label_seq(cif_path)
            union_contacts |= contacts
            inter = contacts & true_positive
            union = contacts | true_positive
            recall = len(inter) / len(true_positive) if true_positive else float("nan")
            precision = len(inter) / len(contacts) if contacts else float("nan")
            jaccard = len(inter) / len(union) if union else float("nan")
            recalls.append(recall)
            precisions.append(precision)
            jaccards.append(jaccard)
            print(f"    {cif_path.name}: {len(contacts)} contact residues, "
                  f"recall={recall:.3f} precision={precision:.3f} jaccard={jaccard:.3f}")

        union_recall = len(union_contacts & true_positive) / len(true_positive) if true_positive else float("nan")
        summaries[variant] = {
            "mean_recall": _mean(recalls), "mean_precision": _mean(precisions),
            "mean_jaccard": _mean(jaccards), "union_recall": union_recall,
        }
        print(f"[downstream_eval] {variant} summary: mean recall={summaries[variant]['mean_recall']:.3f}, "
              f"mean precision={summaries[variant]['mean_precision']:.3f}, "
              f"mean jaccard={summaries[variant]['mean_jaccard']:.3f}, "
              f"union-of-top-{len(cifs)} recall={union_recall:.3f} "
              f"({len(union_contacts & true_positive)}/{len(true_positive)} true epitope residues "
              f"covered by at least one of the top {len(cifs)} designs)")

    if conditioned_variant in summaries and "baseline" in summaries:
        c, b = summaries[conditioned_variant], summaries["baseline"]
        print(f"\n[downstream_eval] {pdb_id}: {conditioned_variant} vs baseline "
              f"(mean recall {c['mean_recall']:.3f} vs {b['mean_recall']:.3f}, "
              f"mean jaccard {c['mean_jaccard']:.3f} vs {b['mean_jaccard']:.3f}, "
              f"union recall {c['union_recall']:.3f} vs {b['union_recall']:.3f}) -- "
              f"conditioning {'DID' if c['mean_recall'] > b['mean_recall'] else 'did NOT'} "
              f"shift generated contacts toward the true epitope on this target.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdb_id")
    parser.add_argument("--launch", action="store_true", help="Actually run both real campaigns (slow, real GPU time)")
    parser.add_argument("--compare", action="store_true", help="Compare already-completed campaign results")
    parser.add_argument("--num_designs", type=int, default=50)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--conditioned-variant", default="conditioned",
                         help="Subdirectory name for the conditioned spec/campaign -- default 'conditioned' "
                              "(Model A's original results). Pass e.g. 'conditioned_D' to compare a different "
                              "steering model without clobbering an earlier model's completed campaign.")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.pdb_id, args.conditioned_variant)
        return

    specs = build_both_specs(args.pdb_id, args.conditioned_variant)

    if args.launch:
        for variant, spec_path in specs.items():
            print(f"\n{'=' * 60}\n[downstream_eval] {variant}\n{'=' * 60}", file=sys.stderr)
            run_campaign(spec_path, args.num_designs, args.budget)
        compare_results(args.pdb_id, args.conditioned_variant)
    else:
        print("\n[downstream_eval] Both specs validated, not launched (pass --launch to run "
              "both real campaigns). Launch commands:")
        for variant, spec_path in specs.items():
            print(f"  cd {spec_path.parent} && {venv_bin('boltzgen', 'boltzgen')} run {spec_path.name} "
                  f"--output run1 --protocol antibody-anything --num_designs {args.num_designs} "
                  f"--budget {args.budget} --num_workers 0 --cache {WEIGHTS_DIR / 'boltzgen'}")


if __name__ == "__main__":
    main()
