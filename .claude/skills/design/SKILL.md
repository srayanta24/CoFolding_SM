---
name: design
description: Two related but distinct capabilities in ~/co_folding: (1) cofold a target against a GIVEN partner (small molecule, protein, antibody, peptide, RNA, DNA) to predict/evaluate a complex, or (2) generate a NOVEL binder/antibody against a target from scratch via BoltzGen. Use for "run a design", "cofold X against Y", "predict a complex for <target>" (capability 1), or "design a binder/antibody against <target>", "generate a binder for <target>" (capability 2). Always clarify which one is meant if ambiguous — they use different tools and very different runtimes.
---

**First, disambiguate which capability the user wants** — "design" is
genuinely ambiguous between these two, and they use different tools:

- **Evaluate a given pair** ("cofold X against Y", partner already known/specified)
  → §A below (Boltz-2 / OpenFold3, ~1-3 min).
- **Generate something new** ("design a binder/antibody against X", no
  partner sequence given) → §B below (BoltzGen, tens of minutes to hours,
  needs the target's 3D structure not just sequence). If the user's request
  doesn't make clear which they mean, ask — don't assume generation is
  wanted just because they said "design" (see conversation precedent: a
  prior "design an antibody against lysozyme" request was clarified before
  starting, since defaulting to the wrong one wastes a lot of GPU time).

---

## §A. Cofold a given target + partner

Runs one cofolding prediction end to end: fetch a target sequence, build
per-backend configs, invoke the backend(s), report structure + confidence.
Backed by `scripts/fetch_target.py` and `scripts/run_design.py` — see those
for implementation, and DESIGN.md for why these specific flags/backends.

## Before running

- Confirm `scripts/setup_env.py` has already been run (`.venvs/boltz2` and
  `.venvs/openfold3` should exist). If not, run it first — it's idempotent.
- **Protenix is not usable for this** (DESIGN.md §3: GB10's sm_121 compute
  capability has no compatible kernel in the installed torch build). Only
  `boltz2` and `openfold3` are viable backends. Don't offer Protenix as an
  option.
- `antibody` and `peptide` modalities have no CDR-specific or other
  specialized handling in either backend — they're modeled as a plain
  protein chain. Say this to the user rather than implying antibody design
  gets special treatment it doesn't have yet.

## Gathering inputs

Ask the user (or infer from their message) for:

1. **Target**: a UniProt accession (e.g. `P69905`) or PDB ID (e.g. `1CRN`),
   OR a raw sequence if they already have one. `fetch_target.py` auto-detects
   UniProt vs PDB from the identifier's shape — don't ask the user which
   source it's from.
2. **Partner modality**: one of `small_molecule`, `protein`, `antibody`,
   `peptide`, `rna`, `dna`.
3. **Partner value**: a SMILES string for `small_molecule`, or a sequence
   for everything else.
4. **Backend**: default to `both` unless the user wants a quick check
   (Boltz-2 alone is faster, ~1-2 min including MSA vs OpenFold3's ~2-3 min)
   or a more thoroughly-validated single result (OpenFold3 is the primary
   backend per DESIGN.md).
5. **Job name**: derive something short and descriptive from the target +
   partner if the user doesn't give one (e.g. `p69905_aspirin`).

Don't ask about MSA settings, sampling params, or output paths — the script
has sensible verified defaults for a first pass.

## Running

```
python3 scripts/run_design.py \
  --target-id <UniProt-or-PDB-ID>   `# or --target-seq <raw sequence>` \
  --partner-modality <modality> \
  --partner-value <SMILES-or-sequence> \
  --name <job-name> \
  --backend <boltz2|openfold3|both>
```

This can take a few minutes (MSA server round-trip + inference per
backend) — run it in the background and report back when done, same as any
other long-running job in this project. Don't set a short timeout.

## Reporting results

The script prints, per backend: the structure file path(s) (`.cif`) and a
confidence summary (`confidence_score`, `ptm`, `iptm` for Boltz-2;
`avg_plddt`, `ptm`, `iptm`, `sample_ranking_score` for OpenFold3 — these are
not directly comparable across backends, report them separately, don't
average them). Summarize plainly:

- Where the structure file(s) landed (`data/designs/<name>/<backend>/...`).
- The confidence numbers, with a one-line sense of what they mean (e.g. ipTM
  above ~0.8 suggests a confident interface prediction; well below that is
  low-confidence and shouldn't be over-interpreted from a single seed).
- If a backend FAILed, say so plainly — don't paper over it. Boltz-2 and
  OpenFold3 failing is unexpected (flag as a real bug to investigate);
  Protenix isn't run by this skill at all, so it shouldn't come up.

If the user wants to actually look at the structure, note that no
visualization is wired up yet (see UI_DESIGN.md's Results page, not yet
built) — they can open the `.cif` in any structure viewer (PyMOL, ChimeraX,
or a quick `py3Dmol` snippet) in the meantime.

---

## §B. Generate a novel binder/antibody (BoltzGen)

This is generative design, not cofolding — a different tool
(source at `src/boltzgen`, installed editable into `.venvs/boltzgen`,
`boltzgen` CLI) with a much heavier workflow. Chosen over RFantibody (the
more established Baker Lab pipeline) specifically because RFantibody
assumes CUDA 11.8 + DGL, which has open aarch64 wheel bugs — same class of
problem that blocked Protenix (DESIGN.md §3). BoltzGen is pip-installable
and, as of the HEL/antibody-anything test run on 2026-07-18, actually works
on this GB10 box. If BoltzGen ever turns out to be inadequate, RFantibody
is the documented fallback, not a from-scratch search.

### What it needs (different from §A)

- The target's **real 3D structure** (a PDB/CIF file with actual
  coordinates), not just a sequence. Two ways to get one:
  - **A known target with a solved structure**: fetch it directly, e.g.
    `curl -s https://files.rcsb.org/download/<PDB_ID>.pdb
    -o data/designs/<name>/<pdb_id>.pdb`. Prefer a high-resolution apo
    (unbound) structure unless the user wants to condition on a specific
    known epitope.
  - **Sequence only, no experimental structure available**: use
    `scripts/design_binder.py`, which predicts the structure first (via
    `scripts/predict_structure.py`, Boltz-2 single-chain fold) and wires
    the result straight into a generated design spec. See "Sequence-only
    entry point" below — this is the common case for a novel/uncharacterized
    target, so default to this path unless the user specifically has (or
    asks for) an experimental structure.
- **For antibody design**: reuse the Fab framework scaffolds already pulled
  into this project at `data/boltzgen_examples/repo/example/fab_scaffolds/`
  (14 real antibody frameworks — adalimumab, belimumab, etc. — each a
  `.yaml`+`.cif` pair with CDR loops marked for redesign). Don't hand-build
  a scaffold from scratch; reference these via the `file: path: [...]` list
  form, same pattern as `data/designs/hel_antibody/hel.yaml`. For nanobody
  design there's an equivalent `nanobody_scaffolds/` directory in the same
  sparse checkout — use `--protocol nanobody-anything` with those instead.
- **Design spec YAML** (`entities:` list): one `file:` entity for the
  target (with `include: - chain: {id: ...}` to pick which chain), one
  `file:` entity whose `path:` is the list of scaffold YAMLs. Optionally add
  `binding_types:` under the target entity to restrict which target
  residues the binder should contact (an epitope) — omit it to let the
  model bind anywhere on the exposed surface, which is fine for a first
  pass. **Gotcha the README calls out explicitly**: residue indices in
  these specs are the mmCIF `label_seq_id`, not the PDB author numbering —
  don't hand-translate author residue numbers into a spec without checking
  in a viewer (molstar.org/viewer works without install).

### Sequence-only entry point (no experimental structure needed)

`scripts/design_binder.py` chains `predict_structure.py` (Boltz-2
single-chain fold) into the design-spec generation above, so the whole
thing can start from nothing but a sequence:

```
python3 scripts/design_binder.py \
  --target-id <UniProt-or-PDB-ID>   `# or --target-seq <raw sequence>` \
  --name <job-name> \
  --protocol antibody-anything      `# or nanobody-anything`
```

By default this stops after `boltzgen check` (fast — folds the target,
writes the spec, validates it) and prints the exact command to launch the
real campaign, rather than silently committing to a multi-hour run — same
"always validate first, confirm before scaling up" pattern as the rest of
this skill. Pass `--launch` (plus `--num_designs`/`--budget` if you want
something other than the 50/5 smoke-test default) to run the campaign
directly.

**Read the printed pLDDT before trusting anything downstream.** This stacks
two layers of prediction: the target's fold is itself predicted, not
experimental, and the binder is then designed against that prediction. A
high pLDDT (>~85, e.g. Crambin folded at 95.1 in this project's own test)
means the target structure is trustworthy to design against. Below ~70,
the script prints an explicit warning — that usually means part of the
input sequence doesn't fold into an ordered domain on its own (a signal
peptide, a transmembrane segment, an intrinsically disordered region), and
if you know the folded domain's boundaries, re-run with just that
subsequence rather than the full-length sequence. Prefer an experimental
structure (the non-sequence-only path above) whenever one exists — this
path is for filling the gap when one doesn't, not a strictly-better default.

### Running

1. **Always validate first**: `boltzgen check <spec>.yaml --output
   check_output` — cheap, catches spec errors before burning GPU time, and
   writes an mmCIF you can point the user to for visual sanity-checking
   (designed region should render as a distinct color/chain from the rest
   of the target).
2. **Always pass `--num_workers 0`.** Verified 2026-07-18: the folding step
   hit the exact same CUDA-fork/DataLoader deadlock as Boltz-2 (DESIGN.md
   §3) — 38 minutes elapsed, 30 seconds of actual CPU time, all threads
   parked in `futex_do_wait`. If a run ever looks stalled (high elapsed
   time, near-zero CPU, steady but unmoving GPU memory), it's almost
   certainly this, not a slow step — kill it and restart with `--reuse`,
   which picks back up from the last completed step, no progress lost.
3. **Smoke test before a real campaign**: `boltzgen run <spec>.yaml
   --output <run_dir> --protocol <protein-anything|antibody-anything|
   nanobody-anything|peptide-anything|protein-small_molecule|
   protein-redesign> --num_designs 50 --budget 5 --num_workers 0`. This is
   BoltzGen's own recommended first step, not an arbitrary shortcut —
   confirms the pipeline behaves before committing to a large run.
4. **Real campaigns need far more designs**: the README's own guidance is
   10,000–60,000 `--num_designs` for a real hit-finding campaign (most
   individual designs don't pan out) — this is hours of GPU time, not
   minutes. Confirm with the user before launching one; don't silently
   scale up from a 50-design smoke test.
5. **Always pass `--cache weights/boltzgen`** — unlike Boltz-2/OpenFold3,
   BoltzGen does not default to a path inside this repo, so omitting this
   flag downloads ~6GB to `~/.cache` instead of the permanent weights
   folder (see DESIGN.md §4). First run with a cold cache means a
   multi-minute pause with no GPU activity yet — this is normal, not a
   hang (verified on the HEL run: weights finished downloading a few
   minutes in).
6. Run in the background with a generous timeout, same as any other
   long-running job in this project — don't poll with short sleeps, use
   Monitor/background-task notifications.

### Reporting results

Per the README's documented output layout under `<run_dir>/`:
- `final_ranked_designs/final_<budget>_designs/` — the actual answer:
  quality + diversity optimized top designs, with `.cif` structures.
- `final_ranked_designs/all_designs_metrics.csv` and
  `final_designs_metrics_<budget>.csv` — the metrics behind the ranking.
- `final_ranked_designs/results_overview.pdf` — plots, worth pointing the
  user to directly rather than re-deriving a summary from the CSV by hand.

Report: how many designs were generated vs. how many survived filtering,
where the top-ranked structures landed, and a plain-language note that
these are *computational predictions of novel candidates*, not validated
binders — real confidence requires either external cofolding validation
(chain into §A's `run_design.py` with the designed sequence as the
`--partner-value`) or wet-lab testing. Don't imply a design "works" just
because it ranked highly in silico.
