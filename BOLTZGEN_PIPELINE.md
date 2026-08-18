# BoltzGen pipeline internals — a reference for improving it

Status: draft v1 · 2026-08-16

This documents exactly how BoltzGen's antibody/binder design pipeline works
mechanically, grounded directly in the vendored source at `src/boltzgen/` (file:line
citations throughout) rather than its own README's operational summary. Written as the
first step before trying to actually improve generation quality — everything here is
either a fact about the running code or an explicitly flagged unverified claim, not
inference from the paper (which wasn't fetched — see §7).

README.md's "How antibody design works" section covers this pipeline at the level a
user needs to run and interpret a campaign. This doc goes one level deeper: which
network architecture, which checkpoint, which config knob — the level needed to decide
*where* an improvement effort should intervene.

## 1. Orchestration

CLI entry point: `pyproject.toml:56` → `boltzgen.cli.boltzgen:main`, all in one file,
`src/boltzgen/src/boltzgen/cli/boltzgen.py`.

- `run_command` (line 558) = `configure_command` (615) + `execute_command` (732).
- `configure_command` builds a `BinderDesignPipeline` (858) of `PipelineStep`s (836),
  each pointing at a Hydra template under `resources/config/` plus dot-list overrides.
  Writes the fully-resolved per-stage config to `OUTPUT/config/<step>.yaml` and an
  ordered manifest `OUTPUT/steps.yaml` — what's on disk under
  `data/designs/<name>/run1/`.
- `execute_command` reads `steps.yaml`, and for each step either subprocess-launches
  `resources/main.py <config.yaml>` or in-process Hydra-instantiates the step's
  `_target_` class and calls `.run(config)` (802-823).
- Full possible step list (69-77): `design, inverse_folding, design_folding, folding,
  affinity, analysis, filtering`. `design_folding`/`affinity` are protocol-conditional
  (1129-1172, protein/small-molecule protocols only). **For `antibody-anything` and
  `nanobody-anything` — what this project uses — it's exactly 5 stages: design →
  inverse_folding → folding → analysis → filtering.**

**The load-bearing surprise**: three of the five stages are the *same* Task class,
`boltzgen.task.predict.predict.Predict`, just given a different Hydra template and a
different checkpoint. Only analysis and filtering are genuinely separate code.

| Stage | `_target_` | Base config |
|---|---|---|
| design | `boltzgen.task.predict.predict.Predict` | `resources/config/design.yaml` |
| inverse_folding | `boltzgen.task.predict.predict.Predict` | `resources/config/inverse_fold.yaml` |
| folding | `boltzgen.task.predict.predict.Predict` | `resources/config/fold.yaml` |
| analysis | `boltzgen.task.analyze.analyze.Analyze` | `resources/config/analysis.yaml` |
| filtering | `boltzgen.task.filter.filter.Filter` | `resources/config/filtering.yaml` |

`Predict.run()` (`task/predict/predict.py:97-146`) is the shared GPU driver: it calls
`Boltz.load_from_checkpoint(self.checkpoint, **self.override)` (136) — one model class,
`boltzgen.model.models.boltz.Boltz`, loaded from a stage-specific checkpoint file with
stage-specific override kwargs — then `pytorch_lightning.Trainer.predict(...)`.

## 2. Design stage — novel CDR-loop backbone generation

**Model**: `boltzgen.model.models.boltz.Boltz` (`model/models/boltz.py:56`) — its own
docstring says it "does either: 1. Design 2. Folding with confidence prediction
3. Inverse folding 4. Affinity prediction" (124-130), mode selected by constructor
flags baked into each checkpoint's saved hyperparameters.

**Architecture**: an AlphaFold3/Boltz-2-style pairformer + atom-level diffusion trunk —
**not** an RFdiffusion-style SE(3)-equivariant/frame-based model.
- `InputEmbedder`, `MSAModule`, `PairformerModule`/`MiniformerModule`,
  `RelativePositionEncoder`, `ContactConditioning`, `TemplateModule`, `DistogramModule`
  — all from `model/modules/trunk.py`, wired in `boltz.py:220-333` (standard AF3-lineage
  trunk, also used by Boltz-1/Boltz-2, hence the class name).
- Structure generation: `DiffusionConditioning`
  (`model/modules/diffusion_conditioning.py`) feeds `AtomDiffusion`
  (`model/modules/diffusion.py:266`), an **Elucidated Diffusion Model** (Karras et al.)
  with AF3-style preconditioning (`c_in`/`c_noise`/`c_skip`/`c_out`, 404-415; there's
  even a code comment at line 612 noting a suspected AF3-paper typo they intentionally
  corrected). EDM hyperparameters `sigma_min=0.0004, sigma_max=160.0, sigma_data=16.0,
  rho=7, P_mean=-1.2, P_std=1.5` (`design.yaml:78-84`) match standard AF3/Boltz.
- Scale: `pairformer_args.num_blocks: 64`, `token_s: 384`, `token_z: 128`, `atom_s: 128`,
  `atom_z: 16` (`resources/config/train/boltzgen.yaml:207-281`) — Boltz-2/AF3-scale, not
  a lightweight network.

**What it conditions on**: the design-spec YAML's `include`/`design`/`exclude`/
`structure_groups`/`design_insertions` blocks (parsed by
`boltzgen.data.parse.schema.YamlDesignParser`) produce a per-token `design_mask`.
`BoltzMasker.forward` (`model/modules/masker.py:29-254`) strips sequence identity
(→UNK), MSA, profile, and side-chain reference-atom features for every masked token,
while **framework backbone coordinates stay fully visible**
(`masker.py:65-66`, `override.masker_args.mask_backbone: false` in `design.yaml:48-50`
— currently never turned on). Concretely, for the antibody protocol (e.g.
`example/fab_scaffolds/adalimumab.6cr1.yaml:12-63`): the whole Fab framework is
`include`d (rigid, resolved), CDR loop ranges are `design:`+`exclude:`d (native
coords/identity hidden, anti-leakage), and `design_insertions:` lets loop *length* vary
(e.g. `num_residues: 7..9`). So the model generates variable-length CDR-loop backbones
in 3D space around a fixed, real, solved Fab scaffold — not a whole antibody from
scratch. Matches README.md's "Fab scaffold grafting" section (351-365).

**Sampling**: `AtomDiffusion.sample` (`diffusion.py:501-629`) — Karras-style denoising
from Gaussian noise (`init_sigma * randn(shape)`, 555), **500 steps** (`design.yaml:43`,
vs. 200 elsewhere). Per step: center → optional random rigid augmentation → correlated
noise scaled by `noise_scale` → denoise → optional weighted-rigid-alignment → Euler
update scaled by `step_scale`. Two noise schedules: `"af3"` (standard EDM/AF3, line 417)
and BoltzGen's own `"dilated"` (436, non-uniformly stretches time spent in a sigma
range — `time_dilation=2.667`, `_start=0.6`, `_end=0.8`, used by default,
`design.yaml:85-88`). `step_scale`/`noise_scale` follow a **per-design-batch schedule**
across 4 quartiles of the requested pool (`design.yaml:52-69`; keyed on
`inference_counter` in `boltz.py:1220-1274`) — different subsets of the pool get
systematically different exploration/exploitation settings.

**Two checkpoints, blended 50/50**: `boltzgen1_diverse.ckpt` and
`boltzgen1_adherence.ckpt` (`cli/boltzgen.py:120-125`, `design.yaml:90-95`).
`Boltz.setup()` (438-455) precomputes `switch_points`; `predict_step` (1220-1229)
literally hot-swaps the model's `state_dict` (`load_checkpoint_weights`, 485-490)
partway through the run — half the pool comes from a "diverse"-tuned checkpoint, half
from a "target-adherence"-tuned one.

## 3. Inverse folding stage — sequence assignment

**Model**: same `Boltz` class, `inverse_fold: true`, swapping the AF3 trunk for a
compact structure-conditioned graph network:
- `InverseFoldingEncoder` (`model/modules/inverse_fold.py:290-513`): k-NN graph over
  backbone-4-atom coordinates (`init_knn_graph`, 357, `topk=30`), Gaussian-smeared
  pairwise distances, 6 `MLPAttnGNN` message-passing layers (340).
- `InverseFoldingDecoder` (517-783): 3 `MLPAttnGNNDecoder` layers, decoding
  **autoregressively in random node order** (`torch.randperm`, 673) — each position's
  logits depend on already-decoded neighbors fed back into the graph (619-644,
  700-761). Architecturally ProteinMPNN/ESM-IF-like, but a bespoke BoltzGen
  implementation inside the shared `Boltz` codebase, not borrowed weights.
- **Target-aware**: the k-NN graph spans the whole complex (binder + target), so
  residue choice is conditioned on proximity to target atoms, not just the binder's
  own backbone.
- Non-designed positions held fixed via `design_mask`/`inverse_fold_design_mask`
  (651-659) — only CDR-loop positions get new logits.
- **Diversity control**: `sampling_temperature: float = 0.1` default (536, 558) — *not
  currently exposed as a top-level CLI flag*. At sample time (646-783): `None` →
  deterministic argmax (746); otherwise `multinomial(softmax(logits/temperature))`
  (748-751). A hard restriction mask (`inverse_fold_restriction`) zeroes disallowed
  amino acids — this is how the antibody protocol's cysteine ban is enforced
  (`inverse_folding.yaml:88-89`: `[CYS]`). `tie_symmetric_sequences` (537, 582-617)
  forces identical residues across symmetry-related positions (homomers).
- **Multiplicity**: `--inverse_fold_num_sequences` (default 1, or 10 for
  `--only_inverse_fold`).

**Checkpoint**: `boltzgen1_ifold.ckpt` — a third, separate checkpoint.

## 4. Folding stage — self-consistency refold

**Checkpoint**: `boltz2_conf_final.ckpt` (`cli/boltzgen.py:133-136`) — confirms
README.md's claim directly ("Re-fold the designed binders with their targets using
Boltz-2 model", `src/boltzgen/README.md:371`).

**Independence from design**: genuinely separate checkpoint, loaded fresh in a separate
`Predict.run()` (full trunk, `inverse_fold=False`, `confidence_prediction` enabled —
evidenced by the confidence/PAE/ipTM keys in `fold.yaml:17`). Weight-independent, but
**not code-independent**: same `Boltz` class/trunk/`ConfidenceModule`
(`model/modules/confidence.py`), just different weights and flags. **5 independent
refold samples per design**, 200 diffusion steps (`fold.yaml:57-58`).

## 5. Analysis and filtering

Already covered in depth by `experiments/`'s own research (see
`experiments/aggregate_metrics.py`, `experiments/thresholds.py`) — brief pointer here:
- `task/analyze/analyze.py`: `class Analyze(Task)` (56); `compute_metrics` (521),
  `compute_diversity` (1167), `compute_novelty` (1287).
- `task/filter/filter.py`: `class Filter(Task)` (33); greedy quality/diversity
  selection, `(1-alpha)*quality + alpha*(1-seq_identity)` (79-142), default
  `alpha=0.1` in-code (165) though CLI default differs by protocol (see §6).

## 6. Model weights — one architecture, four checkpoints

From `ARTIFACTS` (`cli/boltzgen.py:120-139`), confirmed in resolved
`data/designs/hel_antibody/run1/config/*.yaml`:

| Artifact | HF file | Used by |
|---|---|---|
| `design-diverse` | `boltzgen1_diverse.ckpt` | design, 50% of pool |
| `design-adherence` | `boltzgen1_adherence.ckpt` | design, 50% of pool |
| `inverse-fold` | `boltzgen1_ifold.ckpt` | inverse_folding |
| `folding` | `boltz2_conf_final.ckpt` | folding (+ design_folding, non-antibody) |
| `affinity` | `boltz2_aff.ckpt` | affinity (small-molecule protocols only) |
| `moldir` | `mols.zip` | all stages (ligand/mol dictionary) |

Four distinct checkpoint files loaded into the same `Boltz` architecture class, hot-swapped via `load_state_dict(strict=False)` — not one shared multi-task checkpoint.

## 7. What wasn't verified

BoltzGen's own paper is linked from its README as a PDF
(`hannes-stark.com/assets/boltzgen.pdf`) but was not fetched while building this doc —
everything above is grounded in the running code, not the paper's claims. The repo's
own README/PYPI_DESCRIPTION contain no explicit benchmark numbers or written comparison
to RFdiffusion/RFantibody; this project's choice of BoltzGen over RFantibody
(README.md:614-616) was a licensing/engineering decision (RFantibody's training code is
exclusively licensed to Xaira Therapeutics, plus CUDA 11.8/DGL/aarch64 issues), not a
documented performance comparison. Worth fetching the paper before deciding where to
intervene, to check claimed results against what's actually implemented here.

## 8. Configuration knobs relevant to improving generation quality

**Design**
- Which checkpoint(s) to blend and in what ratio (`--design_checkpoints`,
  `first_checkpoint_num_samples`).
- `sampling_steps` (500), noise schedule (`"af3"` vs `"dilated"` + dilation params).
- `step_scale`/`noise_scale` — fixed override vs. the default 4-quartile schedule.
- `override.masker_args.mask_backbone` — currently always `false`; hiding framework
  backbone too is an untried lever.
- The design-spec YAML itself: which residues are redesigned, loop-length ranges,
  which of the 14 Fab/nanobody scaffolds.

**Inverse folding**
- `sampling_temperature` (0.1 default) — real diversity/confidence trade-off sitting
  unexposed at the CLI level; would need a `--config inverse_folding
  override.inverse_fold_args.sampling_temperature=...` override.
- `--inverse_fold_num_sequences` — sequences sampled per backbone.

**Folding**
- `diffusion_samples` (5 replicates per design) — more replicates = more expensive but
  more reliable self-consistency signal.
- `--folding_checkpoint` override.

**Filtering**
- `--budget`, `--alpha`, `--metrics_override`, `--additional_filters`,
  `--refolding_rmsd_threshold` — tunable without touching the generative model at all.
