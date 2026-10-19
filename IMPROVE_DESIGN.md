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

### Step 2 — build a labelled scoring benchmark → [experiments/scoring_calibration/PLAN.md](experiments/scoring_calibration/PLAN.md)

**✅ Pilot complete 2026-10-18** — 332 cofolds, 16.9 GPU-hours, zero failures. Full
record: [EXPERIMENT_01](experiments/scoring_calibration/EXPERIMENT_01_iptm_calibration.md).
Headline: **ipTM does not measure binding affinity.** Spearman vs log10(Kd) = +0.134
across 100 real complexes spanning nine orders of magnitude, with the trend mildly
*backwards* (pM median 0.710, weak-binder median 0.824). A real binder scores median
0.699, so step 1's 0.6 threshold is validated locally (0.8 would reject 78% of genuine
antibodies). Separation on binding-destroying mutants is weak, ROC-AUC 0.634.

The consistent reading is that ipTM measures *whether a credible interface can be
modelled at all* — roughly binary — not interface quality on a continuum. A de novo
funnel has to rank thousands of candidates against each other, which is exactly the
continuum it cannot see. This is why the filtration subsection below exists, and it
adds a third branch to §5's open question.

**Rescoped 2026-10-17, after step 1.** This was originally "sweep filter weights on the
existing design pools at zero GPU cost." Step 1's re-scoring showed that is not viable:
across all 1,100 designs on disk only **2** clear both bars, and more fundamentally
**we have no binding labels for designs at all** — a filter rule has nothing to be tuned
or validated against.

The labels exist, but for *real* complexes, in `databases/`, and nothing in this project
has ever scored them through the design pipeline's own metric path. Step 2 is now
building that benchmark: 851 leak-free confirmed binders from `dev`/`test`, 493
Kd-labelled pairs for dose-response, and AB-Bind/SKEMPI destabilising mutants as hard
negatives, all cofolded through `scripts/run_design.py` so the numbers land in the same
space as a campaign's `all_designs_metrics.csv`.

Two reasons this is worth more than the original version:

- It fixes what step 1 exposed. `thresholds.py`'s ipTM > 0.6 is imported from Bennett
  et al.'s **AlphaFold3** results on their hardware. Measuring what a known-real binder
  scores *here* replaces a borrowed constant with a measured one.
- It surfaces the memorisation gap. 99.6% of the Kd-labelled pairs fall inside
  BoltzGen's own training window, so scoring the clean and contaminated sets side by
  side quantifies how inflated any pre-2023-derived threshold is.

**Carry the transfer caveat**: all of it calibrates on real antibodies, not designed
ones. The calibration is high-confidence; a filter rule transferred to designs is not,
and the plan says so explicitly rather than averaging the two together.

### Step 4 — interpose a cheap pre-refold filter (engineering)

Refolding is ~80% of campaign cost and currently runs on **every** generated design.
Filtering on design-stage metrics *before* the refold step would let a fixed GPU-hour
budget cover substantially more samples — the single highest-leverage way to partially
close the 20-1,000× sampling gap in §2 on hardware that cannot close it by brute force.
Needs: a cheap discriminative signal available pre-refold, measured against the
post-refold ranking on an existing pool (which step 2's work produces as a by-product).

*(Numbering follows the working list these were agreed from; there is no separate step 3.)*

### Filtration — what to score with instead of ipTM

Literature review prompted by EXPERIMENT_01's finding. The question: if ipTM is blind to
the affinity continuum, what metric isn't? Candidates below are ordered by
value-per-unit-cost on *our* setup, not by how impressive they are in the abstract.

**The single most useful external result** is a meta-analysis of **3,766 experimentally
characterised binders** across 15 targets (436 confirmed binders, 11.6% hit rate) —
[Overath et al. 2025](https://www.biorxiv.org/content/10.1101/2025.08.14.670059v1.full).
It ranks in-silico metrics by how well they actually predict wet-lab success:

| rank | feature | note |
|---|---|---|
| 1 | **ipSAE_min** | best single feature; 1.4× the average precision of ipAE |
| 2 | **ipSAE_min × interface ΔG/ΔSASA** | best *interaction* feature — ML confidence × Rosetta energy |
| 3 | LIS × interface shape complementarity | |
| 4 | ipTM | what we currently use |
| 5 | AF2 initial-guess ipAE | |

Two things to take from that ordering. First, **ipTM is mid-table** among metrics
computable from the same prediction. Second, **physics complements rather than replaces**
— the best combinations pair an ML confidence score with an orthogonal physical interface
descriptor. But the gain from combining is modest (median AP 0.58 vs 0.57) and their
greedy feature selection found that adding 2–5 features *rarely improved generalisation*
to novel targets. So: pick one good confidence metric and one good physics term, and
resist building a stack.

**Tier 0 — ipSAE. Free, nothing new to compute.**
[Rēs ipSAE loquunt](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11844409/) diagnoses
what is wrong with ipTM and fixes it: count only residue pairs with good interchain PAE,
set `d0` from the number of such residues rather than total chain length, and use the PAE
value directly instead of a probability distribution. Reported 1.4× higher precision than
ipTM/ipAE for binder ranking, and it explicitly supports **Boltz** outputs. **We already
have every PAE matrix** — `pae_*.npz` is written per entry and all 332 are on disk. This
is a pure re-analysis of data we already paid for.

**Tier 1 — Rosetta interface energy. Cheap, CPU-only, already installed.**
`InterfaceAnalyzer` gives ΔG, **ΔG/ΔSASA** (the meta-analysis's best physics term),
buried SASA and shape complementarity (the recommended pre-filter uses `sc > 0.62`).
Bennett et al.'s deep-learning binder paper found Rosetta ddG gave roughly **10× higher
experimental success rate** than the unfiltered population — though the same paper found
AF2 confidence metrics outperformed Rosetta energies for binary binder prediction, which
is consistent with the meta-analysis putting physics in a *complementary* role.

Two practical notes. **FastRelax before scoring is mandatory, not optional** — the
Rosetta scorefunction is sensitive to sub-ångström atom placement, so scoring a
non-Rosetta structure raw produces meaningless energies. And **PyRosetta 2023.11 is
already installed and verified on this aarch64 box** (`.venvs/germinal`, from the paused
Germinal work — `FastRelax` and `InterfaceAnalyzerMover` both confirmed loading). That
work's main deliverable turns out to be useful here rather than there.

**Tier 2 — inverse-folding log-likelihood. The most interesting option, and nearly free.**
[AbBiBench](https://arxiv.org/abs/2506.04235) (155,853 mutated antibodies, 9 antigens)
finds that **structure-conditioned inverse-folding models are the best available
affinity predictors** — ProteinMPNN avg Spearman 0.30, ESM-IF 0.28, with ρ≈0.35–0.50
reported on some sets. That is weak in absolute terms but it is *the right sign and the
right quantity*, which ipTM is not.

The opportunity specific to us: **BoltzGen already runs an inverse-folding model as
pipeline stage 2**, and its per-sequence likelihood is computed and then discarded. We
would be extracting a signal the pipeline already produces. Worth testing on the 332
structures before anything more exotic.

**Tier 3 — ΔΔG predictors (FoldX / Flex ddG). Benchmarkable, weak, known numbers.**
On AB-Bind specifically — one of our own sources — FoldX reaches Pearson **0.34**; on
SKEMPI2, FoldX 0.319 and Rosetta 0.311, with Flex ddG above FoldX. "MD + FoldX" has been
reported best among eight methods for antibody-antigen specifically. These are weak, but
they are *affinity-correlated*, which sets a meaningful floor: anything we adopt should
beat Pearson ~0.3 on our own data.

**Tier 4 — MM-GBSA / MM-PBSA. Probably not worth it here.** Full protocols with MD
equilibration run **~1 hour per complex on a single GPU**, which is unusable at
funnel scale (BoltzGen's own protocol generates tens of thousands). Nwat-MM-GBSA does
report +20–30% ROC-AUC over standard MM-GBSA for protein-protein screening, so it is not
without merit — but on this single-GPU box it could only ever rescore a final top-N, and
Tiers 0–2 should be exhausted first. **Recommend deferring**, not pursuing.

### ✅ EXPERIMENT_02 — results (2026-10-19)

Run: [EXPERIMENT_02](experiments/scoring_calibration/EXPERIMENT_02_alternative_metrics.md).
All three tiers computed over the same 332 structures. **The answer changed the
question.**

A null control (ESM-IF1 scoring the *antigen* chain, which is identical between a
binding-retained and a binding-destroyed mutant and therefore cannot carry signal)
invalidated two of three analysis columns. It scored +0.341 on cross-complex
dose-response — stronger than any real metric — and 0.396 on unpaired AUC. Two direct
checks confirmed why: `matched_positive` and `hard_negative` share only 17 of 34 parent
complexes, so the unpaired comparison partly measures composition; and log10(Kd) spread
*within* a repeated antigen is 1.73 against 7.97 across the corpus, so four-fifths of
the affinity variance is between antigens and any antigen-only feature correlates
spuriously.

**Cross-complex affinity ranking is therefore untested on this benchmark, not
disproven** — including for ipTM. EXPERIMENT_01's "monotonically backwards"
dose-response is better explained as a property of the corpus than of the metric, and
that document now carries the correction.

The valid test is **within-complex** — which is also the only comparison a design
campaign actually poses, since BoltzGen ranks thousands of candidates against *one*
target. On the 17 parent complexes carrying both mutant types:

| metric | paired /17 | p |
|---|---|---|
| **packstat** | **14/17** | **0.006** |
| **shape complementarity** | **13/17** | **0.025** |
| −dG/dSASA, −dG_separated | 12/17 | 0.072 |
| ipTM | 9/17 | 0.500 — chance |
| ipSAE_min | 8/17 | 0.685 — chance |
| ESM-IF1 ll_antibody | 8/17 | 0.685 — chance |
| *ll_antigen (control)* | *9/17* | *0.500 — correct null* |

**The funnel is ranked by a metric that performs at chance on the comparison it is used
for, while three geometric terms do better.**

**Correction to an earlier draft of this section**: it claimed packstat and shape
complementarity are already computed in BoltzGen's analysis stage. They are not —
verified against a campaign's 197-column metrics CSV, where `packstat` and `sc_value`
appear nowhere. Both are Rosetta-only, so adopting them means paying Rosetta's
22.7 min/structure, which the cost finding rules out funnel-wide.

What *is* free: **buried SASA also reaches 13/17 (p=0.025)** and is available as
`delta_sasa_refolded` — which BoltzGen already weights at 2 in its default ranking,
against `design_iptm` at 1. So the zero-cost change is **re-weighting the six metrics
BoltzGen already has**, pushing weight toward `delta_sasa_refolded` and away from
`design_iptm`, not adding new columns.

Why each tier failed to beat ipTM at the headline task, since the reasons differ:

- **Tier 0 (ipSAE)** reweights the same PAE matrix and inherits its blindness by
  construction. 43% of entries collapse to exactly 0.0 at Dunbrack's PAE<10 cutoff,
  calibrated on AlphaFold-with-MSA against our higher no-MSA Boltz-2 PAEs. *No
  PAE-derived metric can rescue a PAE-derived metric.*
- **Tier 1 (Rosetta)** does win the valid test — but through its *geometric* terms, not
  its energy terms (p=0.072, marginal).
- **Tier 2 (ESM-IF1)** is the only metric with the correct cross-complex sign (−0.037),
  but negligible against AbBiBench's ≈0.28. Likely because AbBiBench scores crystal
  structures and these are no-MSA predictions — an inverse-folding likelihood is only as
  good as the backbone it conditions on. Cheap to test by re-running on deposited
  structures; not done.

**Cost, measured.** Rosetta ran ~15 min/structure against ~3 min of GPU to *generate* a
design: rescoring is **5× more expensive than generation**, which rules it out as a
funnel-wide filter at 60,000-design scale regardless of accuracy. ESM-IF1 (7 s) and
ipSAE (~0) are funnel-viable but do not work. packstat/sc are the fortunate case —
effective *and* already free.

**Correction to this section's earlier Tier 2 claim.** It said BoltzGen "already runs an
inverse-folding model and discards the likelihood," implying a one-line extraction. The
decoder's `forward()` does expose a teacher-forced logits path, but it randomises edge
visibility (`inverse_fold.py:622`) so a single pass is stochastic, and driving it
requires building BoltzGen's full feature dict. ESM-IF1 was both less work and the model
AbBiBench actually measured.

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

**Third branch, added 2026-10-18 after EXPERIMENT_01.** The question was framed above as
*tuning vs. training*. The calibration experiment adds a third possibility that is not a
property of the generator at all: **the selection signal may not measure the property we
want.** ipTM is blind to the affinity continuum, and it drives both BoltzGen's internal
ranking and every judgement this project has made about design quality.

This matters for sequencing, because the three branches are not independent:

- If the scoring branch dominates, then improving the generator is partly wasted — a
  better generator still gets selected by a metric that cannot see affinity, and the
  campaign's reported quality would not move.
- It also casts doubt on past conclusions that *relied* on ipTM-family metrics to
  compare pipelines or models, including parts of the epitope downstream work.
- And it is by far the cheapest branch to investigate: EXPERIMENT_02 (§4's filtration
  subsection) runs on structures already on disk.

So the revised recommendation is to resolve the scoring branch *before* reopening
tuning vs. training, not because it is more likely to be the dominant cause, but because
it is cheap, it is a prerequisite for measuring the other two honestly, and a wrong
selection signal would contaminate any conclusion drawn about them.

**Update 2026-10-19, after EXPERIMENT_02.** The scoring branch is now partly resolved,
and the finding is actionable rather than merely cautionary. On the comparison a design
campaign actually makes — ranking candidates against one target — ipTM performs at
chance (9/17), while packstat (14/17, p=0.006) and shape complementarity (13/17,
p=0.025) discriminate. Both are already computed in BoltzGen's analysis stage, so the
fix is a `--metrics_override`, not new machinery.

Two cautions carry forward. The result rests on 17 parent complexes and would not
survive Bonferroni correction over the ten metrics tested, so it is a strong lead rather
than a settled fact. And it was measured on *point mutants*, which are a much subtler
perturbation than two independently generated designs — whether the advantage transfers
to real design diversity is exactly milestone 7's transfer question, now with a concrete
hypothesis to test rather than an open-ended one.

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

Filtration review (§4), added 2026-10-18:

- Overath et al., *Predicting Experimental Success in De Novo Binder Design: A
  Meta-Analysis of 3,766 Experimentally Characterised Binders* —
  https://www.biorxiv.org/content/10.1101/2025.08.14.670059v1.full
- *Rēs ipSAE loquunt: What's wrong with AlphaFold's ipTM score and how to fix it* —
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11844409/
- *AbBiBench: A Benchmark for Antibody Binding Affinity Maturation and Design* —
  https://arxiv.org/abs/2506.04235
- Bennett et al., *Improving de novo protein binder design with deep learning* —
  https://www.nature.com/articles/s41467-023-38328-5
- *Flex ddG: Rosetta ensemble-based estimation of changes in protein-protein binding
  affinity upon mutation* — https://www.biorxiv.org/content/10.1101/221689.full.pdf
- Rosetta Relax application docs (FastRelax before scoring non-Rosetta structures) —
  https://docs.rosettacommons.org/docs/latest/application_documentation/structure_prediction/relax

Original v2 pass:

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
