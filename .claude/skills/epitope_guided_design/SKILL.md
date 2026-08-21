---
name: epitope_guided_design
description: Predict an antibody epitope for a target antigen (Model A, experiments/epitope_prediction/) and use it to steer BoltzGen's diffusion design via binding_types conditioning, then evaluate whether conditioning actually shifted generated contacts toward the true epitope. Use for "predict the epitope for <target>", "run an epitope-guided/epitope-conditioned design", "steer the design toward the epitope", "run the conditioned-vs-unconditioned comparison". Not for retraining/re-comparing the epitope models themselves — see experiments/epitope_prediction/PLAN.md directly for that; not for a plain (non-epitope-guided) design/cofold job — see the "design" skill for that.
---

## What this is

Three pieces, chained together, all under `experiments/epitope_prediction/`:

1. **Model A** (geometric GNN, 5-member bagging-PU ensemble) predicts per-residue
   epitope propensity + confidence for a target antigen, from real structure
   coordinates in `databases/sabdab/`. Selected over Model B (geometric+ESM2) after a
   controlled comparison — Model B overfit even with proper early stopping; see
   `PLAN.md` sec 9 for the full story before assuming ESM2 embeddings would help.
2. **`steering/binding_types_spec.py`** turns the prediction into BoltzGen's
   `binding_types` conditioning format (a real, trained `ContactConditioning` module,
   not placeholder metadata — see `BOLTZGEN_PIPELINE.md`).
3. **`eval/downstream_eval.py`** builds a conditioned + baseline design spec pair,
   launches both as real BoltzGen campaigns, and measures whether the conditioned
   designs' actual contact residues shifted toward the true epitope.

Full methodology, architecture comparison, and validated downstream results:
`experiments/epitope_prediction/PLAN.md` (sec 9 = model selection, sec 10 = downstream
steering results). Read that before changing floors/thresholds here — the numbers in
this file summarize it, PLAN.md is the source of truth.

## Predicting an epitope for a target

```bash
source .venvs/epitope-prediction/bin/activate
python3 experiments/epitope_prediction/steering/binding_types_spec.py <pdb_id>
```

Requires the target to already be a real structure in `databases/sabdab/structures/`
— Model A's features are computed from real geometry (SASA, secondary structure,
k-NN graph), not sequence alone, so this doesn't work on a sequence-only or
hypothetical target without a structure first (e.g. from the "design" skill's cofold
capability, or fetched from the PDB directly).

Output: how many residues cleared both floors, the `binding_types` spec string, and
the top 5 by propensity. **Adaptive by design**: if nothing clears both the propensity
and confidence floor, the selection comes back empty — that's the model correctly
declining to guess on a target it isn't confident about, not a bug. Don't override
`PROPENSITY_FLOOR`/`CONFIDENCE_FLOOR` without a specific reason; they were chosen to
make exactly this abstention behavior possible (`PLAN.md` sec 4).

## Running an epitope-conditioned design campaign

`eval/downstream_eval.py` is the practical entry point — builds both a `conditioned`
and `baseline` spec (identical except for `binding_types`), validates both with
`boltzgen check` before launching anything, and optionally runs the real campaigns:

```bash
python3 experiments/epitope_prediction/eval/downstream_eval.py <pdb_id>            # build + validate specs only
python3 experiments/epitope_prediction/eval/downstream_eval.py <pdb_id> --launch    # + run both real campaigns
python3 experiments/epitope_prediction/eval/downstream_eval.py <pdb_id> --compare   # after campaigns finish: real contact-overlap comparison
```

`--launch` is slow, real GPU time — design step ranged ~50min-5.6h and folding
~1-9h across the three targets tested so far, depending on antigen size. Run it in the
background, no short timeout, same convention as every other long job in this project.
Output lands in `experiments/epitope_prediction/eval/.downstream_runs/<pdb_id>/`
(gitignored — large and fully reproducible by rerunning; results belong in `PLAN.md`,
not in a kept copy of the raw campaign output).

## What the evidence says so far (PLAN.md sec 10, 3 real dev.txt targets)

- When Model A makes a **confident** epitope call, `binding_types` conditioning
  measurably improves the generated designs' true-epitope recall/precision/jaccard
  (+11% to +40% relative per design, on both confident targets tested).
- When Model A's confidence floor correctly **declines** to make a strong call
  (near-empty selection), conditioning has no effect either way on the one such target
  tested — not a conditioning-mechanism failure, just nothing meaningful to condition
  on.
- **Conclusion**: v2 (true gradient-guided steered diffusion, modifying
  `AtomDiffusion.sample`'s core denoising loop) is a **no-go for now** — the existing
  `binding_types` conditioning already works when given a good prediction. The real
  bottleneck exposed by this data is epitope-model accuracy/coverage, not the steering
  mechanism. Revisit v2 only if a future confident, correct prediction still fails to
  shift generated contacts.

## Gotchas (verified the hard way — don't reintroduce)

- **`binding_types` has two incompatible YAML formats.** A raw `U`/`B`/`N` string
  keyed by `label_seq_id` only works for inline `protein:` entities with an explicit
  `sequence:` field. A `file:`-based entity (loading a real structure — what
  `downstream_eval.py` always uses) needs the structured range-list format instead:
  `binding_types: [{chain: {id: ..., binding: "68..70,150"}}]`. Use
  `build_binding_range_spec()`, not `build_binding_types_string()`, for any
  `file:`-based spec — get this wrong and you get a `TypeError: string indices must be
  integers` deep in `schema.py`, not a helpful validation message.
- **`include:`/`binding_types:` chain ids are `label_asym_id`, not `auth_asym_id`.**
  `summary.csv`'s `antigen_chain` column is auth-convention — don't hand-copy it into a
  spec; use `antigen_label_chain()` in `downstream_eval.py`, which derives the correct
  label id from the parsed structure.
- **Comparing a design's own output against ground truth needs `label_seq_id`, not
  `auth_seq_id`.** BoltzGen renumbers `auth_seq_id` sequentially in its output CIFs, so
  it no longer matches the original structure's auth numbering — but `label_seq_id`
  (entity-sequence position) is preserved exactly, verified empirically position-by-
  position. Use `compute_interface_labels_by_label_seq()`, not
  `compute_interface_labels()`, when the comparison target is a design output rather
  than a SAbDab structure.
- **BoltzGen's own output CIFs use a different, shorter `_atom_site` column layout**
  than SAbDab's regenerated CIFs (no separate `auth_atom_id`/`auth_comp_id` columns).
  Use `parse_atom_site_generic()` (header-driven, builds its own column map), not
  `parse_atom_site()` (hardcoded SAbDab column positions), on any BoltzGen design
  output.
- **The antigen chain's letter in a design output isn't stable across variants or
  ranks** — `boltzgen check` renames chain ids on conflict (observed
  `Renaming with {'A': 'C'}` on every campaign so far). Identify the antigen chain by
  residue count (always far longer than any FAB scaffold chain, ~1000 vs ~100-140
  residues), not a hardcoded letter — see `design_contacts_by_label_seq()`.
