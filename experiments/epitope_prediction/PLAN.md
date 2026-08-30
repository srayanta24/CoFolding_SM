# Epitope prediction + diffusion steering — detailed plan

Status: draft v1 · 2026-08-16

Implements Option 1 from [IMPROVE_DESIGN.md](../../IMPROVE_DESIGN.md): predict a likely
epitope on an antigen, feed it into BoltzGen's existing (already trained)
`binding_types` conditioning to steer the design stage toward it. Two phases: build the
predictor (this doc's main focus), then wire it into generation.

## 1. Problem framing — which epitope-prediction problem is this

There are two distinct problems in the literature, easy to conflate:
- **Antibody-aware interface prediction**: given a *specific* antibody and antigen,
  predict where they contact (e.g. Epi4Ab — uses ESM2 for the antigen + AntiBERTy for
  the antibody). Not our problem — we don't have a candidate antibody yet; picking the
  epitope is what happens *before* generation.
- **Antibody-agnostic epitope/antigenicity prediction** (Discotope-3.0, SEMA-2.0,
  RoBep): given only the antigen, predict which surface residues are generally
  antigenic — likely to be *someone's* epitope. **This is our problem.**

Target output format, verified directly in BoltzGen's own schema parser
(`src/boltzgen/src/boltzgen/data/parse/schema.py:1066-1090`): the design-spec YAML's
`binding_types` field is a per-residue string, one character per residue —
`U`nspecified / `B`inding / `N`ot-binding — same length as the sequence (padded with
`U` if shorter). This is the exact contract the predictor needs to produce.

## 2. Literature grounding (sequence-based vs. structure-based)

- **Structure-based** (Discotope-3.0, SEMA-2.0, RoBep): the field's leading approach,
  because most real antibody epitopes are *conformational* (spatially clustered on the
  folded surface, not contiguous in sequence) — a sequence-only model can't see this.
  SEMA-2.0 reports ROC AUC 0.76 on an independent test set — a concrete number to
  benchmark against, not just "make it work." **DiscoTope-3.0 specifically trains on
  both solved and AlphaFold-*predicted* structures** — direct precedent for our
  situation (we can always produce a structure, real or Boltz-2/OpenFold3-predicted,
  before predicting an epitope).
- **Sequence-only** (SeRenDIP-CE, BepiPred-family): weaker for conformational epitopes,
  no 3D context. Real use case: extremely fast triage, or a fallback when structure
  prediction is impractical — neither applies here, since this project already has a
  working structure-prediction pipeline (`scripts/run_design.py`,
  `scripts/predict_structure.py`).

**Recommendation: structure-based, v1.** "Sequence-only" input isn't a separate
modeling problem to solve — it's "fold first via the existing Boltz-2 pipeline, then
run the same structure-based predictor," exactly DiscoTope-3.0's own design choice. A
pure sequence-only model is not in v1 scope; revisit only if structure prediction proves
to be a bottleneck in practice (it hasn't been so far in this project).

## 3. Training/eval data — two label sources, used for different things

**Training labels: AACDB's precomputed interface annotations**
(`databases/aacdb/interacting_res_distance/*.txt`) — free, already on disk, no new
computation needed. Verified format: three columns (`antibody`, `antigen`, `distance`),
already self-labeled which residue is on which side — no chain cross-referencing
needed. Verified cutoff convention: max distance present is 5.99Å (checked directly),
i.e. AACDB itself only lists pairs ≤6Å — use that as the positive-label threshold, for
label consistency between train and eval. **Restricted to
`databases/splits/train_era.txt` structures only** (3,628 of AACDB's 3,674 unique PDBs
— verified in `databases/splits/README.md`'s cross-reference table) — this is exactly
what the leak-free splits work exists for.

Framing, matching the field's own standard (DiscoTope-3.0 explicitly calls this
**positive-unlabeled learning**, not naive binary classification): a residue observed in
contact with *any* crystallized antibody is a positive (`BINDING`); everything else is
*unlabeled*, not confidently `NOT_BINDING` — a real epitope simply may not have been
crystallized yet. Worth carrying this framing into the loss function (e.g. PU-learning
adjustment or conservative negative sampling), not just labeling everything else `0`.

**Real gap found and how we're fixing it — eval labels need a different source.**
AACDB's structures are weighted toward pre-2023 PDBs (matches its own May-2024 v1.0
snapshot): only **4** of our 851 `test`+`dev` structures have an AACDB annotation — far
too few for a meaningful held-out evaluation. But **1,437** `test`+`dev` structures have
a real protein antigen and full atomic coordinates already sitting in
`databases/sabdab/structures/`. Per your decision: **build our own coordinate-based
interface-contact labeler** rather than accepting the 4-structure eval set.

- Verified feasible with the same lightweight approach already used for
  `databases/src/build_splits.py`: mmCIF `_atom_site` rows are plain
  whitespace-delimited text (columns confirmed:
  `type_symbol, label_asym_id, label_seq_id, Cartn_x/y/z, auth_seq_id, auth_asym_id`,
  etc.) — no `gemmi`/`biotite` needed, a stdlib parser is enough.
  - `extract_interface_labels(pdb_id) -> dict[residue_id, bool]`: parse antibody chains
    (`Hchain`/`Lchain` from `summary.csv`) and antigen chain(s) (`antigen_chain`)
    coordinates, compute heavy-atom (`type_symbol != "H"`) pairwise distances, label an
    antigen residue `BINDING` if any of its heavy atoms is within 5Å of any antibody
    heavy atom (matching AACDB's own ≤6Å convention, slightly tighter — document the
    exact number used, treat as a tunable parameter).
  - Used **only for evaluation** (`dev.txt`/`test.txt`), not training — AACDB's
    precomputed labels remain the training source (already correct, no reason to
    recompute what AACDB already did well for `train_era`).
  - This also gives a natural **sanity check**: computing our own labels for the small
    overlap where AACDB *does* have `train_era` annotations lets us verify our
    from-scratch labeler agrees with AACDB's before trusting it on the eval set.

## 4. Architecture — build both, compare, use the winner for steering

Per decision: build **two** architectures in parallel rather than deferring the
language-model variant — compare them properly on `test.txt` and take the better one
into §6's steering integration, rather than assuming geometric-only is "good enough."

**Shared**: graph structure is a k-NN graph over antigen residues (backbone atoms),
mirroring BoltzGen's own `InverseFoldingEncoder`
(`src/boltzgen/src/boltzgen/model/modules/inverse_fold.py:290-513`, `init_knn_graph`,
`topk=30`) — not reusing its weights (wrong task, wrong output head), but deliberately
mirroring its architecture pattern for consistency and because it's a proven design for
this exact class of problem (structure-conditioned per-residue prediction), already
validated on this codebase's own hardware. Output head for both: per-residue 3-way
classifier (`BINDING`/`NOT_BINDING`/`UNSPECIFIED` — `UNSPECIFIED` for low-confidence
residues, so the design-spec doesn't force a wrong constraint on borderline cases).

**Model A — geometric-only**: node features are SASA, secondary structure, backbone
geometry (curvature, local packing density) — no sequence-embedding dependency. Also
worth a fast gradient-boosted-trees version of just the hand-engineered features (no
graph at all) as a cheap sanity floor before either GNN.

**Model B — geometric + ESM2**: same graph/head as Model A, with a learned projection of
per-residue ESM2 embeddings (`esm2_t30_150M_UR50D`, 640-dim, verified with a real GPU
forward pass on this machine) concatenated into the node features before the GNN —
matching SEMA-2.0's own hybrid approach. `fair-esm`, `torch-geometric`, and `freesasa`
all verified installing and running cleanly on this aarch64 machine (real GPU forward
passes tested for both torch-geometric's `SAGEConv`/`GATConv` and ESM2 during planning),
isolated in `.venvs/epitope-prediction/`, bootstrapped by `setup_env.py`. One real gotcha
hit and fixed: `torch_geometric.nn.knn_graph` needs the optional compiled extension
`pyg-lib`, not installed — implemented k-NN manually with `torch.cdist`+`topk` instead
of chasing another possibly-aarch64-fragile compiled dependency (see
`model/features.py`'s `build_knn_edge_index`).

**Ensemble + confidence** (per your addition — see the dedicated subsection below,
not an afterthought): each architecture trains as a **5-member bagging-PU ensemble**,
not a single model, producing a `(propensity, confidence)` pair per residue rather than
a point estimate.

**Comparison**: both trained on the identical `train_era`-derived AACDB labels, both
evaluated on the identical from-scratch `test.txt` labels (§3) — a fair, apples-to-apples
comparison, not just two papers' reported numbers. Report AUC, precision/recall,
calibration, *and* the downstream conditioning metric (§5) for both — the winner on the
downstream metric is what actually matters for steering, not classifier AUC in isolation.

### Ensemble training and confidence (bagging-PU)

Each architecture (A and B) trains as a 5-member ensemble
(`model/gnn.py`'s `ENSEMBLE_SIZE`), where each member sees the same confirmed positives
but a different bootstrap resample of the *unlabeled* residues as weak negatives —
standard bagging-PU. This does double duty: it's both the confidence-estimation
mechanism and a principled treatment of positive-unlabeled label uncertainty (a single
model trained once on "unlabeled = negative" can't express "I'm not sure this residue
is really a negative, it just hasn't been crystallized with an antibody yet" — an
ensemble that disagrees on borderline unlabeled residues does express that).

Per-residue output is a **pair**: `propensity` (mean predicted probability across the 5
members) and `confidence` (`1 - normalized_std` — agreement across members). High
propensity + high confidence = strong epitope call; high propensity + low confidence =
plausible but uncertain; low propensity regardless of confidence = not an epitope
residue. `eval/classifier_metrics.py` includes a **confidence-sanity check**: residues
where the ensemble disagrees more (lower confidence) should show measurably higher
prediction error than high-confidence residues — if not, the confidence signal isn't
doing its job, and the eval script says so explicitly rather than hiding it.

**Why this feeds "how many residues to select" (your stated goal)**: this pair is
exactly the contract `steering/binding_types_spec.py` needs: rank antigen residues by
`propensity`, walk down the ranked list, and stop adding residues to the `B` set once
`confidence` drops below a floor *or* `propensity` drops below a floor — producing a
variable-length epitope selection per antigen (a well-supported epitope for an antigen
the model is confident about; a short list, or none at all — everything left `U` — for
a genuinely novel antigen the ensemble disagrees on). Also enables an antigen-level
gate: if *no* residue clears both floors, skip epitope conditioning entirely for that
target rather than forcing a low-confidence guess.

## 5. Evaluation plan

- **Classifier metrics**: per-residue ROC AUC on `test.txt` (compare directly against
  SEMA-2.0's reported 0.76 as an external sanity benchmark — a number from a different
  dataset/method, so not perfectly comparable, but the right order of magnitude to aim
  for), precision/recall at a few fixed operating thresholds, calibration (reliability
  curve, Brier score — matters because `propensity` needs to be a genuinely
  interpretable probability for the residue-count-selection logic in
  `binding_types_spec.py`, not just a monotonic ranking score), and the
  confidence-sanity check described above.
- **Downstream metric — the one that actually matters**: does conditioning BoltzGen's
  design stage on the predicted epitope (via `binding_types`) actually shift generated
  designs' real contact residues toward the predicted/true epitope, compared to
  unconditioned generation on the same target? Measured via the existing analysis
  pipeline's own interface-residue reporting (`task/analyze/analyze.py`) — a classifier
  can have a good AUC and still fail to move the needle on actual generation if the
  conditioning pathway doesn't respond the way we expect. This is the real go/no-go
  signal for whether to proceed to v2 (true gradient-guided steering) or stop at v1.
- Discipline: iterate against `dev.txt` only; touch `test.txt` for final reported
  numbers, same rule established in `databases/splits/README.md`.

## 6. Steering integration — reuse existing conditioning first, new sampling code only as fallback

**v1**: predicted epitope → `binding_types` `U`/`B`/`N` string → design-spec YAML's
target entity → BoltzGen's existing, already-trained `ContactConditioning` module
(`model/modules/trunk.py:28-72,175`, verified real and trained-for, not placeholder
metadata — see `BOLTZGEN_PIPELINE.md` §2). **No new sampling code needed for v1** — this
is the cheapest, lowest-risk integration point, reusing a pathway the model was already
trained to respond to.

**v2 (only if the downstream metric in §5 shows `binding_types` conditioning is too
weak)**: true steered diffusion — gradient/classifier guidance injected directly into
`AtomDiffusion.sample`'s denoising loop
(`src/boltzgen/src/boltzgen/model/modules/diffusion.py:501-629`), adding a loss term at
each step that pulls designed-chain atoms toward proximity with predicted-epitope
residues. Conceptually similar to Germinal's hotspot-loss (`IMPROVE_DESIGN.md` §2) but
mechanistically different — gradient guidance during BoltzGen's own diffusion sampling,
not backprop through a separate frozen oracle. Real new engineering (modifying a
core sampling loop in vendored code) — correctly sequenced as a fallback, not the
starting plan.

## 7. Folder layout (`experiments/epitope_prediction/`)

```
experiments/epitope_prediction/
  PLAN.md                    # this file
  setup_env.py                # one-time .venvs/epitope-prediction/ setup (torch, torch-geometric, fair-esm, freesasa, scikit-learn)
  README.md                  # (once built) results summary, how to reproduce
  data/
    interface_labels.py       # our own coordinate-based labeler (auth_asym_id-keyed, see gotcha below), for dev/test eval + a sanity check against AACDB
    train_labels.py            # AACDB-derived positive labels, restricted to train_era, cached to .train_labels_cache.json
  model/
    features.py                 # shared geometric features (SASA, secondary structure, local density) + manual k-NN graph construction
    esm2_features.py              # per-residue ESM2 embeddings + learned projection (Model B only)
    dataset.py                     # assembles+caches (features, graph, labels) per structure per split -- .dataset_cache/{train,dev,test}.pt
    baseline.py                     # gradient-boosted-trees sanity floor, saved to checkpoints/baseline.pkl
    gnn.py                           # shared GNN architecture (SAGEConv-based) + 5-member bagging-PU ensemble training, saved to checkpoints/model_{A,B}_seed{0-4}.pt
  eval/
    classifier_metrics.py         # AUC/precision/recall/calibration/confidence-sanity-check on dev.txt or test.txt, all three models
    downstream_eval.py             # builds+launches real conditioned/baseline BoltzGen campaigns, compares generated contacts vs true epitope (see sec 10)
  steering/
    binding_types_spec.py          # Model A propensity+confidence -> binding_types (U/B/N string or structured range)
```

**Real gotcha hit and fixed while building `interface_labels.py`** (worth keeping
visible, not just in a commit message): `summary.csv`'s `Hchain`/`Lchain`/`antigen_chain`
columns are **`auth_asym_id`**, not `label_asym_id` — verified systematically across a
30-structure random sample (`_entity_poly.pdbx_strand_id`, which
`databases/src/build_splits.py` matches `antigen_chain` against, is consistent with
`auth_asym_id` in every case, never with `label_asym_id` alone). `build_splits.py` was
unaffected (it happened to match the right field already), but the first draft of
`interface_labels.py` filtered atoms by `label_asym_id` and silently produced wrong
results (caught immediately by testing against a known AACDB-annotated structure before
trusting it further — see `data/interface_labels.py`'s module docstring for the full
story and `--sanity-check` for the ongoing verification). Also fixed during the same
pass: a PDB entry can have multiple AACDB annotation files (multiple antibody copies in
the asymmetric unit, confirmed for 1,990 of 3,628 overlapping structures) — both the
labeler and its sanity check now union across all of them rather than comparing against
an arbitrary single file.

## 8. Milestones

1. ✅ Build `interface_labels.py` (our own coordinate-based labeler); sanity-checked
   against AACDB's own labels on the `train_era` overlap (mean Jaccard 0.587 after
   fixing the two bugs above; hand-verified individual cases as high as 0.92, with
   remaining low-agreement cases traced to genuine SAbDab/AACDB source disagreements,
   not labeler bugs).
2. ✅ Build `train_labels.py` (AACDB-derived, `train_era`-restricted: 3,520 structures,
   2.9M residues, 6.4% positive) + `features.py` (shared geometric features) +
   `esm2_features.py` (Model B embeddings, alignment-safe by construction) +
   `dataset.py` (caching layer).
3. ✅ Build `baseline.py` — GBT AUC 0.574 on `test.txt`.
4. ✅ Train **both** Model A and Model B as 5-member bagging-PU ensembles.
5. ✅ Evaluated on `test.txt` — see §9, **Model A wins**.
6. ✅ Built `binding_types_spec.py` + `downstream_eval.py`; ran three real
   conditioned-vs-unconditioned campaign pairs across `dev.txt` targets — see §10.
   Result: conditioning measurably helps when Model A is confident.
7. ✅ Go/no-go on v2 (true gradient-guided steered diffusion): **no-go for now** — see
   §10's recommendation.

## 9. Final results and model selection (test.txt, 751 held-out structures)

| | AUC | Brier | Confidence-sanity check |
|---|---|---|---|
| GBT baseline | 0.574 | 0.087 | n/a |
| **Model A (geometric)** | **0.625** | 0.086 | ✅ passes (error 0.276→0.087 monotonically) |
| Model B (geometric+ESM2) | 0.582 | 0.088 | ❌ fails (non-monotonic, jumps to 0.103 in the top confidence bin) |

**Model A selected for §6's steering integration.**

Two real bugs were caught and fixed before these numbers were trustworthy — both are
worth remembering, not just fixing silently:

- **`freesasa`'s `hasRelativeAreas` flag doesn't guarantee `relativeTotal` is a real
  number.** 40% of training structures had NaN in the SASA feature column (non-standard
  residues, chain termini with no reference max-area), which propagated to NaN training
  loss across an entire first ensemble run. Fixed with an explicit NaN/Inf check and a
  raw-area fallback (`model/features.py`'s `compute_sasa`).
- **Alternate conformations (`label_alt_id` A/B) were not filtered**, producing
  duplicate atoms per residue (two `CA`, two `CB`, etc.) that also fed NaN into
  `freesasa`. Fixed by keeping only the primary altloc (`data/interface_labels.py`'s
  `parse_atom_site`).

A third, non-bug finding mattered just as much: **the first training pass had no early
stopping** (fixed 30 epochs), and Model B — with much more input capacity via ESM2 —
overfit hard: lower training loss than Model A (0.23 vs 0.34) but *worse* held-out AUC,
and a confidence-sanity check that correctly flagged it as unreliable. Adding proper
early stopping (`gnn.py`'s `split_train_val` + best-checkpoint pattern, held out from
`train_era` only, distinct from `dev.txt`/`test.txt`) raised Model B's *internal*
validation AUC substantially (0.86 vs Model A's 0.80) — but on the true, temporally- and
sequence-cluster-held-out `test.txt`, Model B still underperforms Model A, and its
confidence signal still fails the sanity check. This is a real generalization gap, not
a training artifact: ESM2 embeddings likely let the model pick up antigen-family-specific
patterns well-represented in the older training corpus that don't transfer to genuinely
novel antigen families, while the geometric features (SASA, secondary structure, local
packing) are universal biophysical properties that do transfer. The confidence-sanity
check did exactly the job it was built for — catching this before it silently fed a
less-trustworthy model into the steering integration.

## 10. Downstream conditioned-vs-unconditioned comparison (milestone 6, eight targets)

**Method** (same for all eight targets below): build `conditioned` (design spec's
target entity has `binding_types` set from Model A's prediction) and `baseline`
(identical spec, no `binding_types`) specs, run both as real full BoltzGen campaigns
(50 designs, budget 5, `antibody-anything` protocol, ranked to a final top 5 each;
design step ~50min-5.6h and folding/refolding ~1-9h depending on antigen size, real GPU
time on the single DGX Spark). For each variant's top-5 ranked refolded designs,
recompute antigen contact residues directly from the design's own output CIF
(`downstream_eval.py`'s `design_contacts_by_label_seq` — same 5Å distance logic as
`interface_labels.py`'s labeler, reused via a shared helper, applied to the designed
antibody chains vs. the antigen chain in the refolded complex) and compare against the
true epitope. Antigen-chain identity in the output CIF verified empirically before
trusting this (`pdb_000010gh`: design's antigen chain has the exact same 1006-residue
sequence, same order, same `label_seq_id` numbering as the original structure —
BoltzGen preserves this because the antigen entity is `include`d directly from the
input structure, not regenerated).

Targets 1-3 came from `dev.txt` (used to build and validate the pipeline itself).
Targets 4-8 came from `test.txt` — this project's own held-out convention reserves
`test.txt` for exactly this kind of final reported comparison — selected purely by
Model A's own confidence output (never by checking the true epitope first, which would
bias the evaluation).

**Real bug found and fixed while selecting targets 4-8, worth keeping visible**: an
initial scan of `test.txt` for confident predictions produced nonsensical "confidence"
values as high as 18.5 (confidence should be bounded ~0-1). Root cause: 416 of 751
`test.txt` structures have a **multi-copy antigen** — the same author chain letter
(`auth_asym_id`) shared by more than one physical copy in the asymmetric unit, which
makes `label_seq_id` non-unique within what `summary.csv`'s single `antigen_chain`
field names as one chain (verified directly: `pdb_00009jo9`'s raw propensity/confidence
values were completely sane, 0.008-0.346 / 0.858-0.997, but 206 of its 2060 "residues"
shared a `label_seq_id` with another entry). This breaks any per-position aggregation
keyed by `label_seq_id` alone, and — more importantly for correctness, not just this
scan — means conditioning `binding_types` against a single named chain is ambiguous
about which physical copy a selected position refers to. Targets 4-8 were filtered to
single-copy antigens only (no `label_seq_id` collisions) before ranking by confidence;
multi-copy antigens are out of scope for this pipeline until that ambiguity is
resolved.

**Target 1 — `pdb_000010gh`** (27 true epitope residues; Model A made a **weak, sparse
call** here — only 2 residues selected, positions 123-124, propensity/confidence floors
correctly declining to say more): both variants scored **zero overlap** with the true
epitope (mean recall 0.000, union-of-top-5 recall 0.000/27, both conditioned and
baseline). Diagnosed, not just accepted at face value: the conditioned run's designs
*did* cluster near the predicted region (contacts at 103-167, right around 123-124) —
`binding_types` conditioning is mechanically working, BoltzGen respects it — but the
prediction itself was simply **wrong**: the true epitope is at residues 292-523, a
different region entirely. A correctly-conditioned design built on a wrong prediction
still produces a wrong design.

**Target 2 — `pdb_00008tzu`** (77 true epitope residues; Model A made a **strong,
confident call** — 22 residues selected, mean propensity 0.86, confidence 0.99):
conditioned mean recall **0.301** vs baseline **0.221** (+36% relative), mean precision
0.799 vs 0.635 (+26%), mean jaccard 0.281 vs 0.201 (+40%) — conditioning measurably
improved every individual design. The one metric that went the other way:
union-of-top-5 recall (the fraction of the true epitope covered by *at least one* of
the 5 designs) was actually higher for baseline (0.831 vs 0.740, 64/77 vs 57/77) —
baseline's 5 designs, less constrained, spread out more and collectively covered more
ground, even though each individual baseline design was less accurate on average.

**Target 3 — `pdb_00009cb5`** (31 true epitope residues; Model A made a **moderate,
confident call** — 24 residues selected, mean propensity 0.53, confidence 1.0):
conditioned mean recall **0.355** vs baseline **0.303** (+17%), mean precision 0.455 vs
0.391 (+16%), mean jaccard 0.253 vs 0.228 (+11%); union-of-top-5 recall tied at
0.774/31 for both.

**Target 4 — `pdb_00009cct`** (24 true epitope residues; confident call — 42 residues
selected, confidence 0.94): conditioned mean recall **0.067** vs baseline **0.050**
(+34% relative, both low absolute), mean jaccard 0.032 vs 0.025; union-of-top-5 recall
0.250 vs 0.083 (6/24 vs 2/24) — conditioning helped, though the absolute signal is weak
for both variants on this target.

**Target 5 — `pdb_00009me7`** (39 true epitope residues; confident call — 37 residues
selected, confidence 0.94): conditioned mean recall **0.010** vs baseline **0.026** —
**conditioning underperformed baseline** here, the first such case. Both variants'
absolute recall is very low (near-zero real signal either way) — a high self-reported
confidence (0.938) did not translate into an accurate call on this target, more likely
a genuine miss by Model A than a conditioning-mechanism failure, but it's a real
counterexample to "confident implies conditioning helps."

**Target 6 — `pdb_00008pmy`** (23 true epitope residues; confident call — 19 residues
selected, confidence 0.94): the largest effect observed — conditioned mean recall
**0.243** vs baseline **0.000** (baseline's entire top-5 scored *zero* overlap with the
true epitope on this target), mean jaccard 0.192 vs 0.000, union-of-top-5 recall 0.783
vs 0.000 (18/23 vs 0/23). Unconditioned generation missed this epitope completely;
conditioning recovered most of it.

**Target 7 — `pdb_00009me5`** (49 true epitope residues; confident call — 25 residues
selected, confidence 0.93): conditioned mean recall **0.376** vs baseline **0.335**
(+12%), mean precision 0.725 vs 0.666, mean jaccard 0.329 vs 0.285 (+15%);
union-of-top-5 recall tied at 0.490/49 for both.

**Target 8 — `pdb_00009uvi`** (18 true epitope residues; confident call — 27 residues
selected, confidence 0.93): conditioned and baseline mean recall **tied at 0.289**;
mean jaccard actually favored baseline slightly (0.109 vs 0.117). Union-of-top-5 recall
favored conditioned (0.500 vs 0.389, 9/18 vs 7/18). A genuine wash on the per-design
metrics, mild win on population coverage.

**Synthesis across all eight**: excluding target 1 (Model A's own confidence gate
correctly declined to make a real call there — not part of the "confident" cohort),
the seven confident-call targets split **5 wins / 1 loss / 1 tie** on mean recall
(targets 2,3,4,6,7 favor conditioning; target 5 favors baseline; target 8 ties), and
**5 wins / 2 losses** on mean jaccard (targets 5 and 8 favor baseline there instead).
Averaged across all seven, conditioning improves mean recall by roughly +0.06
(6 percentage points) — but that average is pulled up substantially by target 6's
outlier win (baseline's total miss); excluding it, the average improvement across the
other six confident targets is closer to +0.03 (3 points). **The pattern from the
first three targets holds directionally but is noisier than it first looked**:
conditioning helps more often than not and can produce large wins when Model A's call
lands on the real epitope (target 6), but it is not a guaranteed improvement — target 5
shows a confident call that didn't pay off, a reminder that "confident" (per the
ensemble's own agreement) is not the same as "correct," consistent with Model A's
still-modest 0.625 test AUC. The union-of-top-5 (population diversity) metric remains
the least consistent of the three — 3 wins, 2 losses, 3 ties across all eight — meaning
conditioning's main, reliable effect is tightening individual designs toward the
predicted region, not necessarily broadening the campaign's overall coverage.

**Go/no-go on §7 (v2, true gradient-guided steered diffusion): no-go, reaffirmed with
more evidence.** v1 (`binding_types` conditioning) delivers a real, positive effect on
average across eight real targets, including one dramatic recovery of an otherwise
total miss (target 6) — there's still no evidence the conditioning signal itself is too
weak to be worth v2's much larger engineering investment (modifying
`AtomDiffusion.sample`'s core denoising loop). What the larger sample sharpens is
*where* the bottleneck actually is: not the steering mechanism, but **Model A's
prediction accuracy** — target 1's total miss (low confidence, correctly abstained) and
target 5's miss (high confidence, still wrong) are both prediction-quality failures,
not conditioning failures. The better next investment remains improving or
better-calibrating the epitope model (more training data, sharper confidence
calibration so target-5-like false-positive-confidence cases become rarer) rather than
building v2. Revisit v2 only if a future confident, *verifiably correct* prediction
still fails to shift generated contacts — that hasn't happened in any of the eight
targets tested so far.

## 11. Root-cause analysis: why Model D's offline superiority didn't hold downstream (2026-08-30)

Follow-up to README.md §2.7's synthesis, which left "Model D wins offline on every
metric but loses the downstream 8-target comparison (3W/3L/1T vs Model A's original
5W/1L/1T)" as an open, unresolved question. Investigated by comparing what the offline
metrics measure against what the deployed steering path (`binding_types_spec.py`)
actually does, using the real, already-generated campaign spec YAMLs
(`.downstream_runs/<pdb_id>/conditioned{,_D}/<pdb_id>_spec.yaml`) as ground truth for
what was actually fed to BoltzGen — not a recomputation with today's checkpoints (Model
A's checkpoint has since been overwritten by v2 retraining, so it can no longer
regenerate the exact predictions behind the original `conditioned` campaigns).

**Finding 1 — a real selection-size collapse, not a quality reversal, on most targets.**
`select_binding_residues()` doesn't threshold independently; it walks the
propensity-sorted list and stops at the *first* residue whose confidence dips below
`CONFIDENCE_FLOOR`. Reading the real deployed `binding_types` ranges off both variants'
spec files: on 5 of 8 targets Model D's real selection was **10-38% the size** of Model
A's (`pdb_00009cct`: 4 vs 42; `pdb_00009me7`: 5 vs 37; `pdb_00008pmy`: 7 vs 19;
`pdb_00009cb5`: 9 vs 24; `pdb_00008tzu`: 0 vs 22 — an outright abstention). In every one
of these cases the stop reason was confidence, never the propensity floor — so
`CONFIDENCE_FLOOR` alone governs selection size in practice, and it was tuned
(README §2.7) against an aggregate error-vs-confidence-bin curve, never checked against
whether it reproduces selection *sizes* comparable to Model A's real per-target
behavior. A narrow B-region gives BoltzGen's conditioning much less to steer toward
regardless of how individually correct those few residues are.

**Verified with an offline proxy** (no GPU cost — reranks Model D's own already-cached
propensity output, current on-disk checkpoint, confirmed to reproduce the real deployed
selections exactly): dropping the confidence gate and taking Model D's own top-K by
propensity alone, K set to Model A's real deployed count for that target —
- `pdb_00009cct`: selection-recall **0.000 → 0.500** (0/24 → 12/24) — the clearest case;
  the gate was discarding real, present signal.
- `pdb_00009me7`: **0.077 → 0.231** (3/39 → 9/39) — same direction, smaller effect.
- `pdb_00008tzu`: only **0.039** (3/77) reachable, because only 12 residues on the whole
  antigen ever clear `PROPENSITY_FLOOR=0.3` — this target's gap is *not* fixable by
  touching the confidence floor; Model D's raw propensity signal itself is too thin here,
  a genuine target-specific model-quality gap distinct from Finding 1.

**Finding 2 — `pdb_00009me5` is a genuine per-target quality miss, not a threshold
artifact.** Both models selected exactly 25 residues here (no size gap), but only 7
overlapped. Model A's 18 unique picks are 72% close-or-correct (10 exact hits + 3 within
3.8Å) vs Model D's 18 unique picks at 61% (5 exact + 6 within ~8Å) — Model D is simply
less accurate on this specific antigen despite winning in aggregate over all of
`test.txt`. Both models' wrong calls cluster in the *same* wrong sequence region
(A: residues 233-240, D: residues 225-226, sequence-adjacent) — a shared blind spot
(plausibly a decoy convex/high-SASA patch), not independent architecture-specific noise.

**Net picture**: three separable causes, not one — (1) an under-tuned confidence floor
actively costing Model D real signal on some targets (`cct`, partly `me7`), fixable by
re-tuning or replacing the floor mechanism; (2) targets where Model D's raw propensity
signal is genuinely weaker regardless of thresholding (`tzu`); (3) targets that are a
real, individual-target regression for Model D despite its better aggregate metrics
(`me5`), including a cross-architecture shared blind spot. Only (1) looks fixable
without new model training.

**Next step (not yet run)**: launch a real BoltzGen campaign for `pdb_00009cct` with a
budget-matched Model D selection (top-42 by propensity, `PROPENSITY_FLOOR=0.3` only, no
confidence gate — matching Model A's real deployed count on this target) as a new
`conditioned_D_budgetmatched` variant, reusing the existing completed `baseline` run, to
test whether the offline selection-recall recovery (0.000 → 0.500) survives all the way
through real design + refolding, or whether BoltzGen's actual generation behaves
differently from what the selection-quality proxy predicts. If it does survive,
`CONFIDENCE_FLOOR` for Model D should be replaced with a per-target adaptive budget
(e.g. sized relative to antigen length or Model A's own historical selection sizes)
rather than a fixed global threshold, before trusting Model D over Model A in
production.
