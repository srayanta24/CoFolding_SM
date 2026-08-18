# Improving the design step — options and plan

Status: draft v1 · 2026-08-16

Companion to [BOLTZGEN_PIPELINE.md](BOLTZGEN_PIPELINE.md) (how the pipeline works) — this
is *where to intervene* to actually improve it, working through the design (generation)
stage first per the phased approach: improve each of the 5 pipeline stages
incrementally, starting with stage 1.

Three candidate approaches were researched and compared. **Execution order agreed:
(1) epitope prediction → (2) Germinal comparison → (3) fine-tuning — one at a time, not
in parallel**, informed by whatever the earlier steps teach us.

## Option 1: Epitope prediction + existing conditioning (do first)

**Detailed plan**: [experiments/epitope_prediction/PLAN.md](experiments/epitope_prediction/PLAN.md)
— sequence-based vs. structure-based approach comparison, training/eval data plan (and
a real gap found + fixed: AACDB's labels alone only cover 4 of our 851 clean test/dev
structures), architecture, and the steering integration plan.

**Finding that reframes this option**: BoltzGen's design-spec format already has a
`binding_types` field, and it is not just metadata — it feeds the pairformer through a
real trained embedding and a `ContactConditioning` module baked into the trunk
(`src/boltzgen/src/boltzgen/model/modules/trunk.py:28-72,175`). Epitope conditioning is
already a first-class, trained-for input to the diffusion model.

**What's actually missing**: an epitope *predictor* to generate a `binding_types` spec
automatically when the user only has an antigen and no known target epitope (the common
case). We already have real training signal for this: `databases/aacdb/`'s
`interacting_res_distance/` and `interacting_res_SASA/` — paratope/epitope residue
annotations by two independent methods, ~3,674 complexes, cross-referenced to
`databases/sabdab/structures/` by PDB id.

**Why first**: cheapest of the three — reuses data already built, reuses an existing
trained conditioning pathway rather than requiring new sampling/guidance code. Lowest
risk to the existing pipeline (an epitope predictor is a new, separate small model; it
doesn't touch BoltzGen's checkpoints at all).

## Option 2: Germinal — compare as a parallel pipeline, not a merge

**Correction to initial framing**: Germinal is *not* diffusion-based. It's a
gradient-optimization ("hallucination") approach — backpropagates through a frozen
structure predictor (AlphaFold-Multimer recommended; Chai-1 or Protenix supported) plus
an antibody language model (IgLM or AbLang2), through a 3-phase schedule (logits →
softmax annealing → semi-greedy). Its pipeline shape nonetheless parallels BoltzGen's:
**Hallucination → AbMPNN redesign → cofolding validation** maps onto BoltzGen's
**Design → Inverse-folding → Folding**.

Verified facts (GitHub `SantiagoMille/germinal`, Apache 2.0):
- **Epitope targeting**: explicit hotspot-residue config
  (`target_hotspots: "25,26,39,41"`) + a loss term weighting interface metrics
  (iPAE, iPTM, pLDDT) + a contact-based penalty (`binder_near_hotspot` filter). This is
  the closest real precedent for what "steered generation toward an epitope" looks like
  in practice — informs Option 1's epitope-conditioning design even though the
  generative mechanism differs.
- **Sample efficiency**: published wet-lab hit rates of 4–22% from only 43–101 designs
  per target (nanomolar-to-low-micromolar affinities, four diverse antigens). BoltzGen's
  own documentation says real campaigns need 10,000–60,000 designs — a large gap worth
  taking seriously.
- **Compute cost**: ~2–8 minutes *per design* on an H100, 200–400 GPU-hours for ~200
  successful designs — far more expensive per-design than BoltzGen's one-shot diffusion
  sample. The efficiency gain is in designs-needed-to-find-a-hit, not GPU-seconds.
- **Hardware/license reality on this machine**: Germinal's recommended oracle
  (AlphaFold3) is license-barred for us (DESIGN.md's own backend table: CC-BY-NC-SA,
  bars commercial use and training similar models). Protenix (open alternative) is
  currently broken on this GPU (sm_121 kernel gap, DESIGN.md §3). **Chai-1** (Apache
  2.0, inference-only, already surveyed but never run) is the only viable oracle here.
  IgLM (if used instead of AbLang) is non-commercial-academic licensed — a
  redistribution flag to track if we ever go that route (DESIGN.md already anticipated
  this class of concern generally).

**Why second, and why "compare" not "merge"**: Germinal is a mature, separately
packaged pipeline with a fundamentally different generative mechanism — not something
to graft piecemeal into BoltzGen's codebase. The plan is to stand it up as an
independent tool, run it on the same targets/database as BoltzGen, and cross-validate:
same pattern this project already uses (chaining a BoltzGen design into an independent
Boltz-2/OpenFold3 cofold as a sanity check, README.md's TROP2 worked example). Real
compute cost, but zero risk to the existing pipeline.

## Option 3: Fine-tuning (do last)

BoltzGen ships real training code (`src/boltzgen/src/boltzgen/task/train/train.py`),
supporting continuation from a pretrained checkpoint (`pretrained:` param) — fine-tuning
is architecturally supported, not something to bolt on from scratch. But:

- Every training config in the repo (`design`, `inverse_folding`, and the
  no-distillation variant) uses **8 GPUs even for the "small" 12-block variant** (vs.
  64 blocks full-scale) — confirmed by grep across all three `resources/config/train/*.yaml`
  files. Full-scale continued training is out of reach on our single GB10, matching
  DESIGN.md's original conclusion (which had scoped *Protenix*, not BoltzGen, as the
  cheap fine-tuning target — a plan now stale since Protenix is currently broken here).
- **No LoRA/PEFT support exists in the vendored code** (verified: no matches for
  lora/peft anywhere in `src/boltzgen/`). Would need to add parameter-efficient
  fine-tuning ourselves — a well-trodden technique, but real engineering work.
- The training data pipeline expects a specific structured format (`target_dir`/
  `msa_dir`/`moldir`, cluster-based sampling via pre-computed `cluster_id`s, MSA
  generation, cropping) — feeding `databases/sabdab/structures/` in means building a
  real data adapter, not pointing at raw mmCIFs.

**Why last**: highest effort (new PEFT code, new data adapter, unverified single-GPU
memory/throughput feasibility for backprop through a 64-block trunk), so sequence it
last, informed by what Options 1 and 2 teach us about where the pipeline actually
underperforms.

## Shared prerequisite: leak-free train/test splits

Before training or evaluating *anything* (epitope predictor now, fine-tuned checkpoints
later), we need splits that don't leak against BoltzGen's own training data — otherwise
"improvement" numbers on a contaminated test set are meaningless (the base model could
already have memorized the answer). Verified directly: **every training config in
`src/boltzgen/` filters training data to structures released on or before
`2023-06-01`** (`DateFilter(date="2023-06-01", ref="released")`, consistent across
`design`, `inverse_folding`, and `no_distillation` configs).

Built and verified: `databases/splits/` (temporal cutoff + MMseqs2 antigen-sequence
clustering, so a later crystal form of an already-known antigen doesn't leak into the
"clean" pool via the date check alone). Of 11,458 unique antigen-bound structures in
`databases/sabdab/`: **8,072 `train_era`** (safe for our own training use),
**751 `test`** + **100 `dev`** (genuinely novel post-cutoff antigens, clean holdout),
**2,244 `excluded_ambiguous`** (post-cutoff but clusters with a train-era antigen) and
**291 `excluded_no_sequence`** (post-cutoff, redundancy unverifiable — excluded rather
than assumed safe). Full methodology, verified failure-mode breakdown, and
per-source cross-reference in `databases/splits/README.md`.
