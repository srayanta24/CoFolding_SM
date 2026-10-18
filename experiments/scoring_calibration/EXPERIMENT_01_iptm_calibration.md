# Experiment 01 — Does ipTM measure what we are using it to measure?

Status: **complete** · run 2026-10-17/18 · 332 cofolds, 16.9 GPU-hours, 0 failures

Formal record of the first scoring-calibration experiment. Method rationale lives in
[PLAN.md](PLAN.md); the generated numbers live in
[results/calibration_report.md](results/calibration_report.md) and its JSON sidecar.
This document is the self-contained account: what was asked, how it was asked, and what
came back.

---

## 1. What was the experiment

Every antibody design this project generates is ranked, filtered and reported on by a
small set of model-confidence metrics, of which **`design_to_target_iptm` is the
primary one**. It is weight-1 in BoltzGen's own default ranking (Algorithm 2), it is the
headline column in every benchmark report this project has produced, and it is the
number that led to the conclusion that our campaigns "underperform."

That metric had never been calibrated. Nobody had established what value a *known,
real, experimentally confirmed* antibody-antigen binder scores when pushed through this
exact pipeline on this exact hardware. Without that, a design scoring 0.34 is
uninterpretable — it could be hopeless, or it could be normal.

The experiment: assemble a labelled benchmark of real antibody-antigen complexes with
known binding outcomes, push every one of them through the *same* scoring path a design
goes through, and ask what the metric actually does.

## 2. What were we testing

Four distinct hypotheses, deliberately separated because they have different
consequences and different confounds:

**H1 — Calibration.** *A real binder scores high enough that our pass thresholds admit
it.* If real binders routinely fail our thresholds, the thresholds are wrong, not the
designs. Before this experiment the project used ipTM > 0.8 (inherited from README's
*cofolding* guidance) and, after step 1, ipTM > 0.6 (imported from Bennett et al.'s
AlphaFold3 results — a different model, different hardware, never checked here).

**H2 — Memorisation.** *Scores on complexes inside BoltzGen's training window are
inflated relative to genuinely novel ones.* If so, every literature threshold derived
from pre-2023 complexes is optimistic, and so is any calibration built on them.

**H3 — Dose-response.** *ipTM increases as binding gets tighter.* This is the
load-bearing assumption behind using ipTM to **rank** candidates. Ranking only makes
sense if the metric is monotonic in the quantity you care about.

**H4 — Separation.** *ipTM can distinguish a complex that binds from one that does
not.* Tested with point mutants that measurably destroy binding on an otherwise
identical complex — the cleanest available negative.

H3 is the one that had never been questioned anywhere in this project, and it turned
out to be the one that failed.

## 3. What was the result

| set | n | median ipTM | mean | p25 | p75 | ≥0.6 | ≥0.8 |
|---|---|---|---|---|---|---|---|
| `clean_positive` (leak-free real binders) | 32 | **0.699** | 0.688 | 0.578 | 0.798 | 72% | 22% |
| `contaminated_positive` (train-era, Kd-labelled) | 100 | **0.787** | 0.757 | 0.651 | 0.877 | 80% | 46% |
| `hard_negative` (binding destroyed) | 100 | **0.695** | 0.713 | 0.623 | 0.822 | 87% | 29% |
| `matched_positive` (binding retained) | 100 | **0.790** | 0.775 | 0.676 | 0.860 | 89% | 48% |

**H1 — supported, with a correction.** A real binder scores median **0.699**. The 0.8
threshold the project used before step 1 rejects **78%** of genuine antibodies. The 0.6
threshold rejects **28%**. Bennett et al.'s imported 0.6 therefore transfers to this
Boltz-2 pipeline usably — an external constant validated locally, which is what this
experiment existed to do — but it is not conservative, and a hard ipTM gate discards
roughly a quarter of real binders.

**H2 — supported, modest.** Clean 0.699 vs contaminated 0.787, gap **+0.088**,
P(contaminated > clean) = **0.628**. Real, smaller than expected.

**H3 — REJECTED, and the trend runs backwards.**

| Kd band | n | median ipTM |
|---|---|---|
| pM (<100 pM) | 17 | 0.710 |
| 100 pM – 10 nM | 52 | 0.775 |
| 10 nM – 1 µM | 27 | 0.818 |
| weak (>1 µM) | 4 | 0.824 |

Spearman ipTM vs log10(Kd) = **+0.134** across 100 complexes spanning nine orders of
magnitude. The expected sign is *negative*. Not merely uninformative — mildly inverted.

**H4 — weakly supported.** matched_positive 0.790 vs hard_negative 0.695,
**ROC-AUC 0.634**. On the 17 parent complexes holding both a retained and a destroyed
mutant — which controls for complex identity — positives score higher in only
**9/17**.

## 4. What dataset did we use

Built by [`build_sets.py`](build_sets.py) into `sets/manifest.tsv` (1,366 entries);
332 scored in this pilot. All sequences are the **resolved** chains from `_atom_site`
coordinates, not the declared `_entity_poly` construct — see §5's bug note.

| set | n scored | source | label meaning |
|---|---|---|---|
| `clean_positive` | 32 | `databases/sabdab/` ∩ `splits/dev.txt` | Crystallised Ab-Ag complex → confirmed binder by construction. Post-cutoff and antigen-cluster-clean, so outside BoltzGen's training data. |
| `contaminated_positive` | 100 | `databases/sabdab/affinity/` | Measured Kd, 4.0×10⁻¹³ – 2.0×10⁻⁴ M. 99.6% fall in `train_era`. |
| `hard_negative` | 100 | `databases/abdesign_db/` | Point mutant with ΔΔG ≥ +2.0 kcal/mol or ELISA mut/wt ratio ≤ 0.1 — binding measurably destroyed. |
| `matched_positive` | 100 | `databases/abdesign_db/` | Point mutant on the same parent complexes with \|ΔΔG\| < 0.5 or ELISA ratio ≥ 0.8 — binding retained. The control that makes the negatives interpretable. |

**Scope limits, chosen deliberately**: Fabs only (both chains present), because the
pipeline being calibrated runs `antibody-anything` against Fab scaffolds; protein
antigens only (`antigen_type == PROTEIN` exactly), so 111 short-peptide-antigen affinity
rows were dropped; total complex ≤ 1,200 residues as a cost guard.

**AB-Bind was not read directly.** Only 20 of its 32 complexes cross-reference into
SAbDab. Of the other 12, eight are not antibody-antigen at all (cyclophilin/HIV capsid,
CheA/CheY, β-lactamase/inhibitor, TGF-β receptor/ligand ×2, a *monobody*, and two
Fc/Fc-receptor pairs that bind through Fc rather than a paratope) and four are homology
models. AbDesign-DB already reprocesses the usable AB-Bind rows with sequences attached,
so it was used instead.

## 5. How did we test the hypothesis

**Scoring path** ([`score_sets.py`](score_sets.py)). Each entry was cofolded with
**Boltz-2**, heavy + light + antigen as three chains, and the confidence metrics
recorded. Two choices were made for comparability rather than convenience:

- **Boltz-2, not OpenFold3**, because BoltzGen's own folding/refold stage loads a
  Boltz-2-derived checkpoint (`boltz2_conf_final.ckpt`). Its ipTM is therefore the one
  actually comparable to a campaign's `design_to_target_iptm`.
- **No MSA** (required adding `--no-msa` to `scripts/run_design.py`, which previously
  hardcoded `--use_msa_server`). BoltzGen refolds designs with the target template and
  no MSAs. Scoring real complexes *with* an MSA would hand them an information source
  designs never get, inflating the reference and defeating the calibration.

**Statistics** ([`analyze.py`](analyze.py)) are distribution-free throughout — medians,
Spearman rank correlation, and Mann-Whitney U expressed as an AUC — because ipTM is
bounded on [0,1] and visibly skewed.

**Provenance rule, enforced in code rather than left to the reader.** Structural sets
(clean, contaminated) share one extraction path; the mutant sets necessarily use
AbDesign-DB's curated sequences, because no crystal structure of a point mutant exists.
Separation is therefore computed as `hard_negative` vs `matched_positive` — both
curated — and **never** against `clean_positive`. Crossing that boundary would confound
sequence provenance with the effect and would have produced a flatteringly high AUC for
the wrong reason.

**Confound handling.** The contaminated set skews high-affinity (29% pM, 50% nM) while
the clean set is "whatever got crystallised." This was flagged before the numbers landed
rather than after. Finding H3 resolves its direction: high-affinity complexes score
*lower*, so affinity differences would **shrink** the memorisation gap, not inflate it —
making +0.088 a floor rather than a ceiling.

**A real bug, caught before it cost GPU time.** The first version of `build_sets.py`
read sequences from `_entity_poly`, which is the declared *construct* and silently
includes signal peptides — `pdb_000011zw`'s heavy chain came out as
`MPMGSLQPLATLYLLGMLVASVLA|EVQLVESGGG…`. Cofolding those would have produced disordered
termini on every entry, depressed every metric, and inflated the GPU bill, while
calibrating against something the design pipeline never sees (it conditions on a
*structure*). Rebuilt from `_atom_site` resolved coordinates: that entry became
214/212/113 residues, starting cleanly at `EVQLVESGGG`. `esm2_features.py` had reached
the same conclusion independently, for the same reason.

**Cost, measured.** 3.1 min/complex mean (2.5 median, 9.3 max); 16.9 GPU-hours for 332
complexes; **zero failures**. PLAN.md had estimated 3–5 min from the BoltzGen paper's
timing figure — the low end was right.

## 6. What were the findings

**1. ipTM does not measure binding affinity.** Across 100 real complexes spanning nine
orders of magnitude of Kd, the rank correlation is +0.134 with the wrong sign. This is
the experiment's central result and it was the cheapest question it asked.

The coherent interpretation, consistent with H4 also being weakly positive: ipTM
measures **whether a credible interface can be modelled at all** — a roughly binary
property — not interface quality on a continuum. That explains how it can weakly detect
a mutation that *abolishes* binding (AUC 0.634) while being blind to the difference
between a picomolar and a micromolar binder.

A de novo design funnel needs to rank thousands of candidates against each other. That
is precisely the continuum this metric appears unable to see.

**2. The project's thresholds were measuring against the wrong scale, and step 1's fix
is now empirically justified rather than borrowed.** 0.8 rejects 78% of real
antibodies; 0.6 rejects 28%. The imported Bennett et al. constant survives local
validation.

**3. The design-vs-binder gap is real, not a threshold artefact.** Real binders sit at
median 0.699; our campaign designs sit at ~0.33. Our single best design ever recorded
(`ctrop2` at 0.692) lands almost exactly at the *median* of real binders — one design in
1,100 reaching typical-binder quality. The distributions barely overlap.

**4. A third explanation for the pipeline's performance joins the two in
IMPROVE_DESIGN.md §5.** That section framed the open question as tuning vs. training.
This experiment adds: **the scoring function may not measure the property we care
about.** Improving the generator cannot help if selection is being driven by a metric
blind to affinity.

**5. Filter-rule work (PLAN.md milestone 6) cannot proceed on ipTM alone**, which tops
out near AUC 0.634 — roughly the *untuned baseline* of the autoresearch filter paper
(0.637), whose tuned rule reached 0.806. Any real gain must come from metrics not yet
computed. See IMPROVE_DESIGN.md §4's filtration section for the literature review this
finding prompted; the two immediately actionable candidates (ipSAE from the saved PAE
matrices, and Rosetta interface energy via the already-installed PyRosetta) both run on
the 332 structures already on disk with **no new GPU time**.

### Limitations

- `clean_positive` is n=32 (`dev` only, per split discipline). The median is usable; its
  confidence interval is wide. The full 283-entry clean arm would tighten it.
- The weakest Kd band is n=4. The monotonic trend across the other three bands is the
  stronger evidence, not that single bin.
- Complex size is uncontrolled. Larger, harder-to-model complexes may be both
  lower-affinity in this corpus and lower-scoring, which a size-stratified reanalysis
  should separate before H3 is treated as fully settled.
- Everything here is measured on *real* antibodies. That a metric fails to rank real
  complexes by affinity strongly suggests it cannot rank designs by affinity either, but
  that step is an inference, not a measurement.
