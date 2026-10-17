# Improving antibody design end to end — findings and plan

Status: **draft v2 · 2026-10-17** (v1 was 2026-08-16, scoped to the design stage only)

Companion to [BOLTZGEN_PIPELINE.md](BOLTZGEN_PIPELINE.md) (how the pipeline works
mechanically) — this is *where to intervene* to actually improve it.

**What changed in v2.** v1 framed the problem as three approaches (epitope prediction →
Germinal comparison → fine-tuning) applied to pipeline stage 1, to be walked through one
at a time. Option 1 is done (twice: a v1 build and a v2 model pass — see §6.1); Option 2
is paused at milestone 1 of 7; Option 3 never started. In the meantime stages 2-5 were
never touched at all, and a literature pass plus a direct read of BoltzGen's own paper
and training configs surfaced a reframe that makes the original ordering the wrong one.
This revision records that reframe, lists the near-term work it implies, and explicitly
defers the larger question rather than pretending the near-term work answers it.

---

## 1. The central finding: BoltzGen has no antibody specialization anywhere

BoltzGen is a **universal** binder design model. It was trained on all complex types and
was never fine-tuned for antibodies. Its "antibody mode" is an inference-time
construction, not a trained capability. Verified directly against the vendored source,
not inferred from the paper:

- **`antibody-anything` and `nanobody-anything` are byte-identical config dicts**
  (`src/boltzgen/src/boltzgen/cli/boltzgen.py:95-106`). Both set exactly two things:
  `analysis: largest_hydrophobic=false, largest_hydrophobic_refolded=false` and
  `filtering: filter_cysteine=true`. Nothing else. The *only* thing that distinguishes a
  Fab campaign from a nanobody campaign is which scaffold YAML the user's own design spec
  points at (`fab_scaffolds/` vs `nanobody_scaffolds/`).
- **No protocol selects a different checkpoint.** `ARTIFACTS`
  (`cli/boltzgen.py:120-139`) is fixed: `design-diverse`, `design-adherence`,
  `inverse-fold`, `folding`, `affinity`, `moldir` — the same weights serve every
  modality.
- **Zero occurrences of `antibody`, `nanobody`, `cdr`, `immunoglobulin`, `sabdab`, `fab`
  or `vhh` across all four training configs**
  (`src/boltzgen/src/boltzgen/resources/config/train/{boltzgen,boltzgen_small,boltzgen.no_distillation,inverse_folding}.yaml`).
  Training data is filtered only by `SizeFilter` / `DateFilter` / `ResolutionFilter` /
  `ConfidenceFilter` and sampled by `ClusterSampler` — sequence-cluster sampling over a
  general PDB corpus.
- **The absence is meaningful, not an oversight of inspection.** The same configs *do*
  carry modality-specific loss upweighting — `nucleotide_loss_weight: 5.0`,
  `ligand_loss_weight: 10.0` (`boltzgen.yaml:362-363`). The authors upweight the
  modalities they wanted to improve. Antibodies are not among them; they are ordinary
  proteins in the corpus.
- **The inverse folding model is generic and soluble-only.** Per the paper's own §3.5, it
  is "largely similar to SolubleMPNN" with 6 encoder layers instead of 3, trained
  *excluding transmembrane and fibrillar proteins*, and verified by the authors to perform
  "similar to ProteinMPNN and SolubleMPNN." There is no antibody term and no CDR-aware
  behaviour; the antibody protocol's only influence on it is the cysteine ban, which it
  shares with the peptide protocol.

**Consequence.** Every lever currently available to us — epitope conditioning, sampling
schedules, inverse-folding temperature, filter weights — tunes a general-purpose binder
model into behaving more like an antibody designer at inference time. These are real and
worth doing, but they are **incremental by construction**: none of them changes the fact
that nothing in the stack was ever trained to know what an antibody is. Treat the
near-term work in §4 as exactly that, and do not mistake gains there for having addressed
this.

**Corroborating gap in the paper's own evidence.** Across 26 targets and 8 wet-lab
campaigns, the experimentally validated binder classes are **nanobodies, proteins,
peptides and small molecules**. There is no Fab campaign anywhere in the validation.
Our default protocol (`antibody-anything`, i.e. Fab scaffolds) is therefore the *least*
externally evidenced path in a model that has no antibody training to begin with.

---

## 2. Scale and measurement: we have been reading smoke tests as campaigns

Independent of §1, a second finding invalidates most of this project's own quality
numbers.

**BoltzGen's method is extreme oversampling plus hard filtering.** From the paper's
wet-lab methods: for each of 10 novel targets they generated **60,000 designs** and
advanced the **15 highest-ranking** to the lab. The same shape holds throughout — 9,999
designs → 5 tested (brilacidin), 20,000 → 3 (heme), 10,000 → 6 (rucaparib), 1,000 → 6
(peptide targets). Figure 1a states it plainly: "samples tens of thousands of designs
that are subsequently ranked and filtered... resulting in a small set on the order of
tens of candidates."

This project's campaigns are 50-3,000 designs keeping 5-300. **We are 20-1,000×
under-sampled**, and have been reading the unfiltered top of a funnel as if it were the
pipeline's output quality. A mean `design_to_target_iptm` of 0.31-0.34 and mean
`filter_rmsd` of ~11.6Å across our campaigns is approximately what BoltzGen's own raw
output looks like before its filter does the work.

**Our pass/fail thresholds are borrowed from the wrong task.**
`experiments/thresholds.py` classifies `design_to_target_iptm > 0.8` as pass, sourced
from README.md's *cofolding* guidance. The antibody-design literature's empirically
validated operating point is **ipTM > 0.6** (Bennett et al., Nature 2025 — reported AUC
0.86 for predicting actual binding success, with only 9% of their ordered VHH designs
clearing it). BoltzGen's own self-consistency cutoff is **2.5Å** RMSD (paper §4.2), not
our 3.0Å. At 0.8 our campaigns contain zero candidates; at 0.6 they contain real ones.
Until this is fixed, every quality number this project has produced is measured against a
miscalibrated scale.

**Full-scale campaigns are out of reach on this hardware, and that is a planning
constraint, not a detail.** Our own 3,000-design run took ~76.5h wall-clock, of which the
refolding stage was 61.5h (~80%) — consistent with the paper's own per-stage timing
(Figure 14), where refolding dominates. Linear extrapolation to 60,000 designs is ~64
days of continuous GPU time. We cannot run BoltzGen's own protocol at its own scale.
**This makes the cheap stages more important, not less**: if we cannot buy quality with
sample count, we have to buy it with better filtering and better per-design cost.

---

## 3. Stage-by-stage state of the pipeline

For `antibody-anything`/`nanobody-anything` the pipeline is 5 stages (BOLTZGEN_PIPELINE.md
§1). Status of our own intervention effort:

| Stage | What it does | Touched? |
|---|---|---|
| 1. design | CDR-loop backbone diffusion against a fixed Fab scaffold | **Partly** — epitope `binding_types` conditioning only (§6.1). Sampling schedule, step/noise scale, checkpoint blend, `mask_backbone` all untouched. |
| 2. inverse_folding | sequence assignment, SolubleMPNN-like | **Never** |
| 3. folding | Boltz-2 refold for self-consistency | **Never** |
| 4. analysis | metrics, PLIP contacts, developability liabilities | **Never** |
| 5. filtering | quality-diversity ranking to budget | **Never** |

Verified: every reference in this repo to a stage 2-5 knob (`sampling_temperature`,
`inverse_fold_num_sequences`, `diffusion_samples`, `alpha`, `metrics_override`,
`refolding_rmsd_threshold`) is a *generated default* inside a completed run's `config/`
directory. None was ever deliberately set.

**Specific untouched facts worth knowing before planning:**

- `inverse_fold_num_sequences` defaults to **1** — one sequence per generated backbone.
  `sampling_temperature` is hardcoded at 0.1 (`model/modules/inverse_fold.py:536,558`)
  and is not exposed as a CLI flag.
- The default ranking rule is now known exactly (paper Algorithm 2): **worst weighted
  metric rank** over six metrics with weights `design_iptm:1, design_ptm:2,
  neg_min_design_to_target_pae:1, plip_hbonds_refolded:2, plip_saltbridge_refolded:2,
  delta_sasa_refolded:2`, scored as `max_j(rank_ij / w_j)`. Fully overridable via
  `--metrics_override` with no model change.
- Paper §3.4 documents the design-stage knob semantics: **increase step scale or decrease
  noise scale → higher designability, lower diversity** (and the reverse). They use
  N=300 function evaluations; our resolved `design.yaml` uses 500 sampling steps — an
  unexplained divergence worth checking.
- Known upstream failure mode: **ubiquitin memorization** at binder lengths 73-76, where
  generation diversity collapses (paper §5).

---

## 4. Near-term work — explicitly incremental, do first anyway

Three items, ordered. All are cheap, all are prerequisites for being able to measure
anything else honestly. **None of them addresses §1.**

### Step 1 — recalibrate the yardstick (free)

Fix `experiments/thresholds.py` to the antibody-design operating points (ipTM 0.6,
refold RMSD 2.5Å), mark their provenance the way that module already requires
(`readme` vs `heuristic` → now also `literature`), and re-score every campaign already on
disk. Everything downstream depends on this; doing it after the filtering work would mean
tuning filters against a broken target.

### Step 2 — filtering and ranking experiments (zero GPU)

The full design pools from completed campaigns are on disk with all 197 metric columns,
and `steps.yaml` + `config/` are retained, so `boltzgen execute --reuse` can re-run the
filtering stage alone. Nothing needs regenerating. Work:

- Sweep `--metrics_override` weights and `alpha` against the known defaults above.
- Evaluate the "RMSD-Tuned Triad" training-free filter rule from the autoresearch filter
  paper (reported ROC-AUC 0.637 → 0.806 on 7 antibody-antigen systems) as an alternative
  to BoltzGen's worst-weighted-rank.
- **Caveat to carry**: we have no wet-lab labels, so "better filter" can only be scored
  against proxies (epitope-contact recovery, held-out reference complexes via
  `experiments/score_reference.py`). State that limit in any result rather than implying
  a validated improvement.

### Step 4 — interpose a cheap pre-refold filter (engineering)

Refolding is ~80% of campaign cost and currently runs on **every** generated design.
Filtering on design-stage metrics *before* the refold step would let a fixed GPU-hour
budget cover substantially more samples — the single highest-leverage way to partially
close the 20-1,000× sampling gap in §2 on hardware that cannot close it by brute force.
Needs: a cheap discriminative signal available pre-refold, measured against the
post-refold ranking on an existing pool (which step 2's work produces as a by-product).

*(Numbering follows the working list these were agreed from; there is no separate step 3.)*

---

## 5. The comprehensive question — deferred until after §4

Once steps 1, 2 and 4 are done we will have, for the first time, trustworthy measurements
and a sense of how much headroom exists purely in inference-time tuning. **The larger
question — what to do about §1, that nothing in this stack was ever trained on antibodies
— is deliberately left open here rather than answered prematurely.**

Candidate directions exist (antibody-specialized inverse folding; antibody-aware
fine-tuning of the design model; a different base model; reward-guided sampling such as
Feynman-Kac steering, which is inference-time and gradient-free), and the v1 options
below remain live. But choosing among them before knowing how much of the current gap is
*tuning* versus *training* would be guessing. Revisit this section with real numbers from
§4 in hand.

---

## 6. Status of the original three options (v1)

### 6.1 Option 1: Epitope prediction + `binding_types` conditioning — **done**

Two full passes. v1: coordinate-based interface labeler, AACDB training labels, GBT
baseline, Models A/B as 5-member bagging-PU ensembles, steering integration, 8-target
conditioned-vs-baseline validation. v2: training data 3,520 → 6,149 structures, a real
`STANDARD_RESIDUES` labeling bug fixed (sanity Jaccard 0.587 → 0.748), SASA scale fix,
features 4 → 7, class-imbalance handling, isotonic calibration, Model A retrained
0.625 → 0.867 AUC, plus two EpiFormer-derived architectures (C: 0.838, D: 0.876) and
EpiFormer itself stood up as an independent downstream evaluator. Full detail:
[experiments/epitope_prediction/PLAN.md](experiments/epitope_prediction/PLAN.md) and
[README.md](experiments/epitope_prediction/README.md).

**Two go/no-go calls both landed on no-go for gradient-guided steered diffusion**, with
the bottleneck diagnosed as prediction accuracy rather than the conditioning mechanism.

**Honest reframe, new in v2**: BoltzGen's own flagship novel-target results were obtained
**without specifying a binding site**. Epitope conditioning buys *control* over where a
binder lands, and plausibly costs hit rate to get it. The two predictors belong in the
"steerability" column, not the "performance" column — they are not an answer to §1.

**Open and unresolved**: whether Model A or Model D is the production steering model.
`steering/binding_types_spec.py` loads D; the root README and the
`epitope_guided_design` skill both say A; the epitope README's own synthesis leaves it
explicitly undecided after D's offline superiority failed to survive the downstream
re-run (3W/3L/1T vs A's 5W/1L/1T). Needs a decision recorded in one place.

### 6.2 Option 2: Germinal comparison — **paused at milestone 1 of 7**

PyRosetta resolved on aarch64 (uv-managed Python 3.10, direct wheel install bypassing
`pyrosetta-installer`'s own aarch64 guard bug). JAX installed and GPU-smoke-tested, but
**ColabDesign — the AF2 autodiff engine Stage A actually depends on — was never
attempted**, which is the undersrisked part. See
[experiments/germinal_comparison/PLAN.md](experiments/germinal_comparison/PLAN.md).

Side finding worth promoting: that work diagnosed the **Protenix sm_121 failure** as a
plain `F.linear` bf16 GEMM against a torch build shipping sm_120 SASS with no PTX, and
verified a fix in isolation on a nightly build exposing `compute_120`. DESIGN.md §8 and
README.md's backend table still describe this as unfixed and should be updated.

**v2 note on priority**: Germinal answers "is a different pipeline better," not "how do
we improve this one." It stays deprioritized relative to §4, but becomes more interesting
if §5 concludes the answer to §1 is "change the base model."

### 6.3 Option 3: Fine-tuning — **not started**

`experiments/finetuning/` exists and is empty. Blockers from v1 still stand and were
re-verified: every training config needs **8 GPUs** even for the 12-block "small" variant
(vs 64 blocks full-scale), there is **no LoRA/PEFT anywhere** in the vendored code, and
feeding `databases/sabdab/structures/` in requires a real data adapter
(`target_dir`/`msa_dir`/`moldir`, precomputed `cluster_id`s, MSA generation, cropping).

**v2 note**: §1 makes this the option most directly aimed at the real gap — it is the
only one of the three that would put antibody knowledge into the weights. Its cost
profile has not improved, but its *relevance* has. Reassess in §5.

---

## 7. Shared prerequisite: leak-free train/test splits

Built and verified at `databases/splits/`: temporal cutoff mirroring BoltzGen's own
training filter (`DateFilter(date="2023-06-01", ref="released")`, consistent across all
training configs) plus MMseqs2 antigen-sequence clustering, so a later crystal form of an
already-known antigen cannot leak into the clean pool via the date check alone.

Of 11,458 unique antigen-bound structures in `databases/sabdab/`: **8,072 `train_era`**,
**751 `test`** + **100 `dev`**, **2,244 `excluded_ambiguous`**, **291
`excluded_no_sequence`** (post-cutoff, redundancy unverifiable — excluded rather than
assumed safe).

**Known documentation gap**: `databases/splits/README.md` is referenced from several
places in this repo but does not exist and was never committed. The verified counts and
per-source cross-reference currently live only in `databases/splits/splits_summary.json`
and `.claude/skills/splits/SKILL.md`. Worth writing.

---

## Sources for the v2 literature pass (2026-10-17)

- BoltzGen, *Toward Universal Binder Design* — https://hannes-stark.com/assets/boltzgen.pdf
  (read directly; closes BOLTZGEN_PIPELINE.md §7's "paper not fetched" gap)
- Bennett et al., *Atomically accurate de novo design of antibodies with RFdiffusion*,
  Nature 2025 — https://www.nature.com/articles/s41586-025-09721-5 (ipTM 0.6 operating
  point, AUC 0.86)
- *AntiFold: improved structure-based antibody design using inverse folding* —
  https://academic.oup.com/bioinformaticsadvances/article/5/1/vbae202/8090019
- *Benchmarking antigen-aware inverse folding methods for antibody design* —
  https://www.biorxiv.org/content/10.1101/2025.08.05.668698v1.full
- *Autoresearch Discovery of Interpretable Filter Rules for Antibody* —
  https://www.biorxiv.org/content/10.64898/2026.05.05.723069v1.full.pdf
- *Controllable protein design through Feynman-Kac steering* —
  https://arxiv.org/html/2511.09216v1 (inference-time, gradient-free; code at
  https://github.com/ErikHartman/FK-RFdiffusion)
- *Scaling Atomistic Protein Binder Design with Generative Pretraining and Test-Time
  Compute* — https://arxiv.org/html/2603.27950v1
- Boltz release history — https://boltz.com/news (no BoltzGen-2 as of this date; BoltzGen
  Oct 2025 remains current)
