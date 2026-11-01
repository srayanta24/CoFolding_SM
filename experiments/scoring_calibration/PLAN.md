# Scoring calibration — what does a real binder score on this pipeline?

Status: **arc complete · 2026-11-01** (plan text below is draft v1, 2026-10-17)

> **Outcome, after six experiments and three benchmarks.** The benchmark this plan set
> out to build exists three times over: 28 parent complexes with paired
> retained/destroyed point mutants; six antigens with 1,530 affinity-labelled
> antibodies; and 1,096 real BoltzGen designs across two antigens with a
> structural-self-consistency label. Four metric families were scored through them:
> BoltzGen confidences, ipSAE, Rosetta (FastRelax + InterfaceAnalyzer, ~1,300
> core-hours total), and inverse-folding likelihoods including AntiFold.
>
> **The answer is that no cheap scalar ranks antibody designs by binding.** ipTM is at
> chance (14/28) on the within-complex comparison a campaign makes. `packstat` beats it
> there (23/28, p=0.0005) but [does not generalise](EXPERIMENT_04_packstat_transfer.md):
> ρ=+0.058 on graded affinity (below its null), and
> [0.508 on real designs](EXPERIMENT_05_design_transfer.md) (chance, below its null,
> worse than chance on a second target). The 23/28 was specific to point mutants of
> near-native crystal backbones.
>
> **The one usable design-level signal is ipTM** (AUC 0.708) — but on a *structural
> self-consistency* label, not binding, and it is already computed for free. It is the
> property EXPERIMENT_01 said ipTM measures: whether a credible structure can be
> modelled, not whether it binds.
>
> **Full conclusions: [§11](#111-the-headline).** The decisive fact is not in any single
> experiment: across the three benchmarks every metric's rank changes with the question
> (two-benchmark ordering agreement ρ = +0.087; the two metrics beating the null on both
> are exactly what chance predicts, expected 1.88, observed 2, p=0.71). So the claim is
> not "the correlations were low" but **no metric survives a change of question**.
>
> Write-ups: [EXPERIMENT_01](EXPERIMENT_01_iptm_calibration.md) ·
> [_02](EXPERIMENT_02_alternative_metrics.md) ·
> [_03](EXPERIMENT_03_within_antigen_ranking.md) ·
> [_04](EXPERIMENT_04_packstat_transfer.md) ·
> [_05](EXPERIMENT_05_design_transfer.md) ·
> [_06](EXPERIMENT_06_proteinmpnn_transfer.md). Generated numbers in `results/`,
> regenerable with `analyze.py`, `analyze_metrics.py`, `analyze_paired.py`,
> `analyze_asd.py`, `analyze_designs.py`, `analyze_crossbench.py`.
>
> What the null controls contributed is worth recording separately: they changed the
> conclusion in EXPERIMENT_02, _03 and _04, and stayed cleanly at chance in _05. Every number in this arc was run against a
> quantity that provably cannot carry the signal, and that discipline caught three
> results that would otherwise have been reported.

Implements **step 2** of [IMPROVE_DESIGN.md](../../IMPROVE_DESIGN.md) §4, after step 1
(threshold recalibration) changed what step 2 could usefully be. This plan replaces the
original "sweep filter weights on an existing design pool" framing, which step 1 showed
is not viable, with the thing that has to happen first: **build a labelled benchmark of
real antibody-antigen complexes scored through our own pipeline.**

## 1. Why the original step 2 doesn't work

Step 2 was going to re-rank existing campaign pools with different metric weights and
filter rules, at zero GPU cost. Step 1's re-scoring killed that:

- Across all 1,100 designs on disk, **2 clear both bars** (ipTM > 0.6 and
  `filter_rmsd` ≤ 2.5). Two positives is not enough to tune or validate anything.
- More fundamentally, **we have no binding labels at all.** The autoresearch filter-rule
  paper that motivated this step had 7 antibody-antigen systems with binder/non-binder
  labels. We have campaign metrics and no ground truth about whether any design binds.

A filter rule needs labels. We do not have them for designs, and we are not going to —
that requires a wet lab. But we *do* have them for real complexes, sitting in
`databases/`, and nothing in this project has ever scored them through the design
pipeline's own metric path.

## 2. A second problem step 1 exposed

`experiments/thresholds.py` now uses ipTM > 0.6, imported from Bennett et al.
(Nature 2025). That number was measured with **AlphaFold3, on their hardware, on their
designs**. We adopted it because it was better-grounded than the 0.8 we had, not because
it was measured here. The `filter_rmsd` ≤ 2.5 threshold is on firmer ground (it is
BoltzGen's own default, `task/filter/filter.py:162`), but the interface-confidence
threshold — the one that decides whether a design is worth anything — is borrowed.

The only way to fix that is to measure what a known-real binder scores on *this*
pipeline, on *this* hardware. That is this plan's primary deliverable, and it is worth
doing even if no filter rule ever comes out of it.

## 3. Data inventory — verified directly, not quoted from the source READMEs

| Source | Content | Structures on disk |
|---|---|---|
| `databases/sabdab/affinity/` | **493 real Ab-Ag pairs with measured Kd**, 4.00e-13 to 2.00e-04 M, median 6.20e-09 | **491 / 493** |
| `databases/sabdab/structures/` + `splits/` | **851 confirmed binders** in `dev`(100) + `test`(751) — post-cutoff, sequence-cluster-clean | yes |
| `databases/ab_bind/` | 1,101 mutations / 32 complexes, ddG −3.37 to +8.00, median 0.63 | yes (incl. homology models) |
| `databases/abdesign_db/` | 1,303 mutants / 47 parent complexes (658 AbDesign ELISA ratios, 645 ddG from SKEMPIv2 + AB-Bind) | **pre-built mutant structures** (3.5GB) |

Kd distribution of the 493 affinity pairs: 143 pM (<1e-9), 246 nM (1e-9..1e-7), 73
sub-uM, 29 uM, 2 weak (>=1e-4). Nearly nine orders of magnitude, well spread — a real
dose-response axis, not a binary label.

AB-Bind ddG distribution: **447 mutations at ddG >= +1.0**, 252 at >= +2.0, 129 at
>= +3.0, and 360 near-neutral (|ddG| < 0.5). The destabilizing tail is the hard-negative
source; the near-neutral set is a matched positive control on the *same* complex.

## 4. The contamination finding — this is what shapes the design

**99.6% of the affinity-labelled pairs (491 of 493) fall in `train_era`** — released on
or before 2023-06-01, which is exactly BoltzGen's own training cutoff
(`DateFilter(date="2023-06-01", ref="released")`, consistent across all its training
configs). The same holds for 20 of AB-Bind's 32 complexes.

This is not a fixable defect in the data; affinity measurements accumulate over decades,
so any Kd-labelled corpus is intrinsically old. Calibrating "what does a true binder
score here" against complexes the folding model has memorized would produce an
**inflated** reference and make genuinely novel designs look worse than they are. That is
precisely the memorisation concern [DESIGN.md](../../DESIGN.md) already raises about this
model class.

**The way around it**: every structure in SAbDab *is* a crystallised antibody-antigen
complex — a confirmed binder. The Kd label is not needed to know it binds. So
`dev.txt` + `test.txt` give **851 confirmed binders that BoltzGen and Boltz-2
demonstrably did not train on**, already temporally filtered and antigen-sequence-cluster
deduplicated by the splits work. That is a leak-free positive set an order of magnitude
larger than the affinity file.

Scoring both the clean and the contaminated sets does double duty: it gives an honest
calibration *and* **measures the memorisation gap directly**, which is a result worth
having on its own.

## 5. The four sets

All scored through the **same path the design pipeline uses** — cofold via
`scripts/run_design.py` (ipTM / pTM / PAE), plus PLIP contact counts and delta-SASA — so
the numbers land in the same space as a campaign's `all_designs_metrics.csv` and are
directly comparable to it. This is the point; metrics computed a different way would not
be.

| Set | Source | Role |
|---|---|---|
| **A. Clean positives** | stratified sample of the 851 `dev`/`test` complexes | The honest answer to "what does a real binder score here." Primary calibration. |
| **B. Contaminated positives** | `train_era` affinity pairs | Scored only to measure the memorisation gap against A. Not used to set thresholds. |
| **C. Graded positives** | the 493 Kd-labelled pairs | Does ipTM track log(Kd)? If it does not, that is a more important finding than any filter rule. |
| **D. Hard negatives** | AB-Bind ddG >= +2.0 (252) and AbDesign/SKEMPI mutants | Binding measurably destroyed on an otherwise identical complex — far cleaner than a random decoy. Ship with pre-built mutant structures, so no modelling step. |

An **easy-negative** set (antibody paired with an antigen it does not bind, drawn from
the same pool) is deliberately *not* in v1. It is trivial to generate and likely trivial
to separate, so it would inflate any reported AUC without telling us anything. Add it
only as a sanity floor if set D turns out to be too hard to separate at all.

### Real caveat on AB-Bind, found while checking and worth keeping visible

**AB-Bind is not a pure antibody-antigen set.** Of its 32 complexes, only 20
cross-reference into SAbDab. Of the remaining 12, checked individually:

- **8 are not antibody-antigen at all** — `1AK4` (cyclophilin/HIV capsid), `1FFW`
  (CheA/CheY), `1JTG` (beta-lactamase/inhibitor), `1KTZ` + `HM_1KTZ` (TGF-beta
  receptor/ligand), `3K2M` (a *monobody*, not an antibody), and `1T83` / `3WJJ` (Fc /
  Fc-gamma-receptor interactions, which are Fc-mediated, not paratope-mediated).
- **4 are genuine antibody complexes but homology models, not crystal structures** —
  `HM_1YY9` (EGFR/hu225), `HM_2NYY` and `HM_2NZ9` (BoNT/A2 mAbs), `HM_3BN9` (Epithin/Fab
  E2).

Filter to the 20 SAbDab-backed complexes for v1. The 4 homology models are usable but
should be tagged as such and reported separately, never pooled silently with crystal
structures.

## 6. Method

For each selected complex:

1. Take the antibody heavy + light sequences and the antigen sequence (already in the
   affinity CSV for set C; parsed from the mmCIF via the same `summary.csv`
   `Hchain`/`Lchain`/`antigen_chain` convention used elsewhere in this project for the
   others — noting that gotcha, those columns are **`auth_asym_id`**, documented in
   `experiments/epitope_prediction/PLAN.md` §7).
2. Cofold via `scripts/run_design.py --partner-modality antibody` with the heavy and
   light chains comma-separated, which is the real two-chain Fab path rather than the
   single-chain approximation.
3. Record ipTM / pTM / PAE from the backend, plus PLIP H-bonds, salt bridges and
   delta-SASA computed from the predicted complex, into one tidy table keyed by PDB id
   with the label (`clean_positive` / `contaminated_positive` / Kd / ddG) attached.

**Do not use the crystal structure's own coordinates to compute the contact metrics.**
The design pipeline scores a *predicted* refold, so the calibration has to score a
predicted structure too, or the distributions are not comparable. This is the same
reasoning that made `score_reference.py` rerun a reference complex through Skill A rather
than measuring its deposited structure.

## 7. Evaluation

- **Calibration curves** — distribution of each metric over set A. Report the percentile
  a given ipTM corresponds to among real binders, which is directly actionable: "ipTM
  0.6 is the Nth percentile of genuine binders on this pipeline" replaces an imported
  constant with a measured one.
- **Memorisation gap** — A vs B on the same metrics. If B scores systematically higher,
  quantify by how much, and treat every literature threshold derived from pre-2023
  complexes as inflated by roughly that margin.
- **Dose-response** — Spearman correlation of ipTM (and each other metric) against
  log(Kd) on set C. A weak correlation would be a significant negative result worth
  reporting on its own.
- **Separation** — ROC-AUC of each single metric, and of BoltzGen's own
  worst-weighted-rank score (`BOLTZGEN_RANKING_WEIGHTS` in `experiments/thresholds.py`),
  for **hard_negative vs matched_positive**. This is the first point at which a filter
  rule can honestly be compared against the default.
- Split discipline, same rule as everywhere else in this project: iterate on `dev`,
  touch `test` only for final reported numbers. The pilot's clean arm is therefore
  `dev`-only, which caps it at 32 entries — thin, accepted deliberately rather than
  padded from `test`.

### Provenance constraint — which comparisons are legitimate

Established while building the manifest, and it changes the separation analysis above.
Structural sets (clean and contaminated positives) share one extraction path
(`_atom_site` resolved coordinates), so the memorisation-gap comparison between them
is provenance-clean. The mutant sets necessarily use AbDesign-DB's own curated
sequences — **there is no crystal structure of a point mutant** — so they carry a
different sequence provenance.

So: compare **within** a provenance, never across it.

| Comparison | Sets | Provenance |
|---|---|---|
| Calibration percentiles | clean_positive alone | structural |
| Memorisation gap | clean vs contaminated | structural, both sides |
| Dose-response vs log(Kd) | contaminated_positive | structural |
| Separation / ROC | hard_negative vs **matched_positive** | curated, both sides |

Scoring hard negatives against clean positives would confound sequence provenance with
the thing being measured, and would likely produce a flatteringly high AUC for the
wrong reason. `matched_positive` exists precisely so that comparison has a same-source,
same-parent-complex control.

## 8. Milestones

1. ✅ Data inventory and contamination audit — §3-§5 above, all numbers verified directly
   against the files on disk.
2. ✅ Set construction (`build_sets.py`) — one tidy manifest, `sets/manifest.tsv`,
   **1,366 entries**: 283 `clean_positive` (dev 32 / test 251), 207
   `contaminated_positive` (all `train_era`, Kd-labelled), 256 `hard_negative`,
   620 `matched_positive`. Plus `sets/pilot.tsv` (332 entries) and
   `sets/summary.json`. AB-Bind is not read directly — AbDesign-DB already
   reprocesses its usable rows with sequences attached, which avoids both the
   non-antibody entries and any mutation-application numbering risk.

   **Real bug caught here, before it cost GPU time**: the first version read
   sequences from `_entity_poly`, which is the *declared construct* and silently
   included signal peptides — `pdb_000011zw`'s heavy chain came out as
   `MPMGSLQPLATLYLLGMLVASVLA|EVQLVESGGG…`, its antigen as
   `MGILPSPGMPALLSLVSLLSVLLMGCVA|ETGTPELEVVQ…`. Cofolding those would have produced
   disordered termini, depressed every confidence metric, and inflated GPU cost —
   while calibrating against something the design pipeline never sees, since that
   pipeline conditions on a structure. Now built from `_atom_site` (resolved
   coordinates only): the same entry is 214/212/113 residues instead of 261/·/175
   and starts cleanly at `EVQLVESGGG`. `esm2_features.py` reached the same
   conclusion independently for the same reason.
3. ✅ **Pilot**: 332 cofolds (32 clean / 100 contaminated / 100 hard negative / 100
   matched positive) via `score_sets.py`, Boltz-2, no MSA. **16.9 GPU-hours, zero
   failures.** Measured cost **3.1 min/complex** mean (2.5 median, 9.3 max) — the low
   end of the 3-5 min estimate.
4. ✅ Analysis (`analyze.py` → `results/calibration_report.{md,json}`). Headline
   results, full detail in the report:
   - **Calibration**: a real binder scores median ipTM **0.699**. The old 0.8 mark
     rejects **78%** of genuine binders; 0.6 rejects **28%**. Bennett et al.'s 0.6
     transfers to this Boltz-2 pipeline, but is not conservative.
   - **ipTM does not track affinity.** Spearman vs log10(Kd) = **+0.134** across 100
     complexes spanning nine orders of magnitude, and the trend is mildly *backwards*
     (pM median 0.710, weak-binder median 0.824). The pilot's most consequential
     finding, and the cheapest question it asked.
   - **Memorisation gap**: +0.088 median, P(contaminated > clean) 0.628 — real but
     modest, and the affinity confound runs against it being an overestimate.
   - **Separation**: ROC-AUC **0.634** (matched_positive vs hard_negative), 9/17 on
     the parent-paired check. ipTM weakly detects binding being *abolished* while
     being blind to the affinity continuum — consistent with it measuring "can a
     credible interface be modelled at all."
5. ✅ Recalibrated `experiments/thresholds.py` from measured values. Both
   `design_to_target_iptm` and `complex_plddt` are now anchored to the real-binder
   distribution under one scheme — **pass ≥ the median real binder, borderline ≥ their
   p25, fail below** — with a new `measured` provenance kind and a `MEASURED_REFERENCE`
   table of percentiles other code can cite. ipTM's pass mark moved *up* (0.6 → 0.699),
   since Bennett et al.'s 0.6 is where they stopped testing, not where something
   resembles a real binder; it now sits inside the borderline band. `complex_plddt`
   graduated from heuristic to measured, and the old guess (0.85/0.70, rescaled from
   README's single-chain bands) turned out close — 0.85 admits 50% of real binders,
   0.70 admits 88%. The module now also carries the affinity caveat prominently:
   clearing the ipTM bar means a credible interface can be modelled, not that it binds.
6. ✅ *(reframed)* Alternative-metric comparison rather than filter-rule fitting —
   see [EXPERIMENT_02](EXPERIMENT_02_alternative_metrics.md). Three independent metric
   families (ipSAE / Rosetta / ESM-IF1) over the same 332 structures. A null control
   invalidated the unpaired-AUC and cross-complex columns, leaving the parent-paired
   test as the only valid one. There: **packstat 14/17 (p=0.006)** and **shape
   complementarity 13/17 (p=0.025)** beat **ipTM at 9/17 (chance)**; ipSAE and ESM-IF1
   are also at chance. Fitting a multi-feature rule was *not* attempted — the
   meta-analysis this plan cites found adding 2-5 features rarely improves
   generalisation, and with 17 parents there is nothing to fit against safely.
7. ✅ **Transfer validation** — [EXPERIMENT_05](EXPERIMENT_05_design_transfer.md),
   2026-11-01. The question was *do packstat and shape complementarity separate good
   from bad designs in a real campaign pool, as they separate retained from destroyed
   point mutants here?* **Answer: no, and emphatically for packstat.** Scored on 1,096
   real BoltzGen designs across two antigens against a structural-self-consistency
   label (`filter_rmsd` ≤ 2.5): packstat is at chance (AUC 0.508 on the committed
   target, below its own null) and worse than chance on the other; shape complementarity
   likewise fails. The point mutants were a far subtler perturbation than generated
   designs, and the transfer did not follow. The one usable signal is ipTM (0.708) — but
   on a *structural-consistency* label, not binding, and it is already free. This closed
   the arc and **cancelled finetuning milestone 2**.

## 8b. Deferred: re-weighting BoltzGen's ranking on real campaign pools

**Status: parked 2026-10-19, blocked on data retention — not on effort.**

EXPERIMENT_02 showed `delta_sasa_refolded` (13/17, p=0.025) discriminates where
`design_iptm` (9/17) does not, and both are already in BoltzGen's default ranking at
weights 2 and 1. The obvious follow-up is to re-weight toward the former, re-run only
the filtering stage on pools already on disk, and check whether the selected top-N
improves. Zero GPU.

Two blockers, both worth recording so the next attempt does not rediscover them:

1. **BoltzGen's `final_rank` could not be reproduced.** Implementing Algorithm 2's
   worst-weighted-rank gives 11.0 where the CSV stores `quality_score` 1.0.
   `quality_score` turns out to be a normalised rank, and `final_rank` additionally
   reflects Algorithm 3's quality-diversity greedy selection. The `rank_*` columns are
   also only ~88% monotone with their own metrics, which remains unexplained. Any
   re-weighting is therefore *our* ranking, not a faithful variant of theirs — fine for
   a comparison, but it must be labelled as such.
2. **The evaluation signal is range-restricted.** Of 1,150 designs across the 23
   epitope-downstream campaigns, only ~10 per campaign-variant have a retained CIF, and
   those are the ones BoltzGen *already* ranked top-10. Epitope-contact recovery can
   therefore only be computed for designs that already won. Re-ranking can be tested for
   reordering *within* a top-10, never for surfacing a better design from the unranked
   remainder — which is the question that actually matters.

**What unblocks it**: campaigns that retain structures for a stratified sample across
the whole rank range, not only the top `budget`. That is a campaign-launch decision
(`--budget`, or keeping `intermediate_*` outputs more broadly), so the cost is disk
rather than compute. Revisit when the next real design campaign is run — note the
retention requirement *before* launching, since it cannot be recovered afterwards
without re-folding every design.

## 9. The risk, stated plainly

**Everything here calibrates on real antibodies, not designed ones.** The assumption is
that a metric threshold separating real binders from real non-binders also separates good
designs from bad ones. That assumption is plausible — it is what Bennett et al. and the
autoresearch filter paper both implicitly rely on — but it is unproven, and this plan
cannot prove it without wet-lab data.

Confidence should therefore be reported differently per deliverable, not averaged:

- **The calibration (§7's first three bullets) is high-confidence.** "A real binder
  scores X here" is a direct measurement with no transfer assumption, and it is useful
  immediately.
- **A filter rule transferred to designs is medium-confidence at best**, and must be
  labelled as such wherever it is used. The only proxy check available is whether the
  rule's ranking correlates with epitope-contact recovery on the epitope-prediction
  downstream targets (`experiments/epitope_prediction/eval/`), which is a weak and
  indirect signal — it measures *where* a design binds, not *whether* it binds.

Do not let a good AUC on set A vs D become a claim about design quality. That is exactly
the kind of slippage this project has caught itself in before (see
`experiments/epitope_prediction/README.md` §2.7, where a model that won every offline
metric lost the real downstream comparison).

## 10. Folder layout

As built (the plan above anticipated three files; the arc needed thirteen):

```
experiments/scoring_calibration/
  PLAN.md                  # this file
  EXPERIMENT_01..04_*.md   # one write-up per experiment, 6-section format

  build_sets.py            # assemble A/B/C/D into a labelled manifest
  build_asd_sets.py        # the six-target within-antigen benchmark (EXPERIMENT_03)
  score_sets.py            # drive scripts/run_design.py, collect BoltzGen metrics
  compute_ipsae.py         # ipSAE from the predicted-aligned-error matrices
  compute_rosetta.py       # FastRelax + InterfaceAnalyzer   (the 1,201 core-hours)
  compute_esmif.py         # ESM-IF1 log-likelihoods, incl. the ll_antigen null control
  compute_antifold.py      # AntiFold -- the controlled A/B on antibody fine-tuning
  packstat_relax_test.py   # is FastRelax skippable? (no: 23/28 -> 15/28)

  analyze.py               # EXPERIMENT_01: calibration, memorisation gap, ROC
  analyze_metrics.py       # EXPERIMENT_02: cross-metric comparison
  analyze_paired.py        # the 28-parent paired sign test (the trustworthy column)
  analyze_asd.py           # EXPERIMENT_03 + _04: within-antigen ranking, partials
  analyze_crossbench.py    # sec 11: do the two benchmarks agree? (no)
  compute_proteinmpnn.py   # EXPERIMENT_06: ProteinMPNN ll, per-complex backbone
  compute_proteinmpnn_shared.py # EXPERIMENT_06 §3.1: ll on one shared predicted backbone
  compute_proteinmpnn_native.py # EXPERIMENT_06 §3.2: ll on the real crystal backbone
  build_design_sets.py     # EXPERIMENT_05: stratified design manifests
  analyze_designs.py       # EXPERIMENT_05: AUC per target, leak-free verdicts
  analyze_crossbench.py    # sec 11: do the two benchmarks agree? (no)
  compute_proteinmpnn.py   # EXPERIMENT_06: ProteinMPNN ll, per-complex backbone
  compute_proteinmpnn_shared.py # EXPERIMENT_06 §3.1: ll on one shared predicted backbone
  compute_proteinmpnn_native.py # EXPERIMENT_06 §3.2: ll on the real crystal backbone

  sets/                    # manifests (tracked - small, decision-bearing)
  results/                 # scored TSVs + generated .md/.json (tracked)
  .scores/                 # per-complex cofold outputs (gitignored, reproducible)
```

Every scorer appends to its own TSV keyed on `entry_id` and guards on a header/row
field-count mismatch. That guard is not decoration: extending the carried-column list
twice produced scorers that wrote wider rows than their headers, and **both times the
failure presented as success** (`[47/47] ok`, `0 to score`, exit 0) with silently
misaligned columns. All three affected tables were repaired and all three scorers now
refuse to write on mismatch.

Tracked/ignored split follows the convention already used in `databases/` and
`epitope_prediction/`: the manifests and analysis outputs are small and decision-bearing
so they stay in git; the raw cofold outputs are large and reproducible so they do not.

---

## 11. Conclusions — what five experiments established

Written 2026-10-31 after EXPERIMENT_04; **extended 2026-11-01 after EXPERIMENT_05, which
closed the arc.** Generated numbers:
[`results/cross_benchmark.md`](results/cross_benchmark.md) and
[`results/designs_comparison.md`](results/designs_comparison.md), regenerable with
`analyze_crossbench.py` and `analyze_designs.py`.

### 11.1 The headline

**No metric tested ranks antibody designs by *binding* in a way that survives a change
of question — and the one metric the arc had elevated fails outright on real designs.**

That is a stronger and more useful claim than "the correlations were low", because it
closes a remedy rather than just reporting a disappointment: the shortlist of cheap
scalars has been enumerated across four families, and none of it transfers.

The decisive evidence is not any single experiment but the comparison across three:

| metric | paired (retained vs destroyed) | ASD (graded affinity) | **designs (self-consistency)** |
|---|---|---|---|
| `packstat` | **23/28, p=0.0005** | +0.058 — *below null* | **0.508 — chance, below null** |
| −dG_separated | 18/28, p=0.093 | +0.125 | 0.666 — PASS |
| `ipTM` | 14/28 — chance | +0.133 | **0.708 — best** |
| shape complementarity | 16/28 | +0.112 | 0.495 — chance |
| *null control* | *17/28* | *+0.070* | *0.512* |

**Every metric's rank changes with the question.** packstat wins the first and loses
the other two. ipTM loses the first two and wins the third. The three columns are not
measuring one latent "design quality"; they are three different properties, and no
metric tracks more than one.

The critical caveat on the third column: `filter_rmsd` is **structural self-consistency,
not binding**. ipTM winning it is EXPERIMENT_01's reading confirmed from the other side —
ipTM measures whether a credible structure can be modelled, which is *not* the same as
whether it binds. So the third column does not rescue ipTM as a selection signal; it
explains what ipTM actually measures.

The two-benchmark detail below (ρ = +0.087 agreement, the chance calculation) predates
EXPERIMENT_05 and still stands for the binding question specifically.

The decisive two-benchmark evidence:

| metric | paired (retained vs destroyed) | ASD (graded affinity) |
|---|---|---|
| `packstat` | **23/28, p=0.0005** | +0.058 — *below null* |
| −dG_separated | 18/28, p=0.093 | +0.125 |
| −dG/dSASA | 18/28, p=0.093 | +0.100 |
| *null control* | *17/28* | *+0.070* |
| shape complementarity | 16/28 | +0.112 |
| buried SASA | 16/28 | +0.065 |
| `ipTM` | 14/28 — chance | +0.133 |
| ESM-IF1 ll_antibody | 13/28 | +0.082 |

**Rank agreement between the two orderings: ρ = +0.087.** Essentially zero. The best
metric on one benchmark is near-worst on the other, in both directions.

### 11.2 Why the two survivors are not survivors

Two metrics beat the null on both benchmarks (the Rosetta energies). Counting survivors
means nothing without knowing how many luck produces, so:

| | |
|---|---|
| beat the paired null | 3 of 8 |
| beat the ASD null | 5 of 8 |
| **expected overlap by chance** (hypergeometric) | **1.88** |
| **observed overlap** | **2** |
| P(overlap ≥ observed) | **0.714** |

The two energy terms surviving both benchmarks is **precisely what a no-signal world
produces.** Neither is individually significant either (p=0.093 on paired, uncorrected).

This is the null-control discipline applied one level up. Within a benchmark we compare
a metric against a quantity that cannot carry the signal; across benchmarks we compare a
*count of survivors* against the count chance produces. Both ask: is this more than
nothing?

### 11.3 The mechanism, which is the actually interesting result

Two metrics with nothing in common — a learned confidence head and a physics-based
packing measure — fail in the **identical pattern**:

| | detects abolished binding | ranks affinity |
|---|---|---|
| `ipTM` | AUC 0.634 | ρ +0.134, wrong sign |
| `packstat` | 23/28, p=0.0005 | ρ +0.058, below null |

Both detect **whether a credible interface exists** — a roughly binary property. Neither
ranks **how good one is**. That convergence across unrelated metric families is the
arc's real finding, and it is consistent with everything else observed: it explains how
ipTM can flag a binding-abolishing mutation while being blind to the difference between
a picomolar and a micromolar binder, and it explains the campaign gap EXPERIMENT_01
measured (real binders at median ipTM 0.699, our designs at ~0.33, one design in 1,100
ever reaching typical-binder quality).

**The pipeline has been selecting on interface existence while we believed it was
selecting on interface quality.** A de novo funnel ranks thousands of candidates that
*all* clear the existence bar, which is exactly the regime where none of these metrics
has been shown to work.

### 11.4 What survived as deliverable

Not nothing, and worth stating separately from the negative result:

1. **Recalibrated thresholds**, measured rather than borrowed — 0.8 rejects 78% of real
   antibodies, 0.6 rejects 28%. In `experiments/thresholds.py` with `measured`
   provenance and a percentile table. The one unambiguous positive.
2. **Three reusable benchmarks**: 28 parents with paired retained/destroyed mutants;
   6 antigens × 1,530 affinity-labelled antibodies; and 1,096 real designs across two
   antigens with a structural-self-consistency label — all scored through this
   pipeline's own metric path with four metric families attached.
3. **The null-control discipline itself.** It changed the conclusion in EXPERIMENT_02,
   _03 and _04 — in _04 it killed a rescue hypothesis that was otherwise about to be
   accepted, and in _05 it cleanly stayed at chance, showing the passing metrics were
   not riding a global-quality channel. Every future experiment here inherits it.
4. **A negative that saved work**: EXPERIMENT_05 cancelled finetuning milestone 2 before
   it was built. `packstat` was going to be the deliverable here — a cheap
   interface-integrity filter — until it was scored on real designs and came in at
   chance. Knowing that cost 97 core-hours; not knowing it would have cost a training
   pipeline built to approximate a metric that does not work.

**What is explicitly NOT a deliverable, having been retracted:** `packstat` as a
selection or filtering metric. Across the three benchmarks it beats chance on exactly
one (point mutants of near-native backbones) and is at or below chance on the two that
involve the real objects — graded affinity and generated designs. It is not wired into
anything.

### 11.5 The overturning check — run 2026-11-01, and it does not overturn

**AbBiBench publishes ProteinMPNN ρ ≈ 0.30 and ESM-IF1 ≈ 0.28 on the same affinity data
where our best inverse-folding column (ESM-IF1 ll_heavy) reached +0.111.** Someone else
got roughly double our best, so the "nothing ranks affinity" conclusion rested on a
benchmark whose resolving power was *borrowed rather than demonstrated*. The check: run
the stronger published scorer (ProteinMPNN) on our own structures and see which reading
holds.

`compute_proteinmpnn.py` (official ProteinMPNN v_48_020, vendored) scored all 1,530 ASD
complexes — same structures, same benchmark, same null as ESM-IF1, so a clean metric
swap. Pooled Spearman vs −log KD:

| | our result | published (AbBiBench) |
|---|---|---|
| ESM-IF1 ll_heavy | +0.111 | ≈0.28 |
| **ProteinMPNN ll_heavy** | **+0.097** | **≈0.30** |
| ProteinMPNN ll_antibody | +0.080 | — |
| *ProteinMPNN ll_antigen (null)* | +0.059 | — |

**The stronger scorer did not help — it landed slightly below ESM-IF1, ~3× under the
published number, and barely above its own null.** Per target it tracks ESM-IF1 almost
exactly: both reach ~0.33 on the two AlphaSeq targets and both go weak-to-negative on
the three AbBiBench targets where the 0.30 is claimed (1n8z −0.22, 4fqi −0.08, 3gbn
+0.22 but sitting *below* its own +0.43 null). No target reproduces 0.30.

**So the reading is the second one: a better inverse-folding model is not the lever on
our structures.** Metric choice is not what separates us from the published baseline.
Of the two things that could be — the scorer or the structures — the scorer is now
ruled out. §11.1 stands: swapping ESM-IF1 → ProteinMPNN buys nothing, and there is still
no validated binding ranker.

**What the check does NOT settle, and the clean follow-up.** This scored each antibody
against *its own per-complex predicted backbone* (what we have on disk, directly
comparable to ESM-IF1). AbBiBench's protocol instead threads every mutant onto *one
native crystal backbone per target*. So a low result here is consistent with two
non-exclusive causes, and this arm cannot separate them:

- **predicted-backbone quality** — our cofolded structures are too noisy to rank from;
- **per-complex-backbone washout** — scoring a sequence against the backbone folded
  *from that sequence* is partly circular and flattens mutation effects.

Either way it is **not the metric**. Separating the two causes needs a
**native-backbone arm**, run in two stages:

- **Stage 1 (done):** fix ONE shared reference backbone per target and thread the
  mutants — isolating the washout confound with zero alignment risk. Result:
  **ρ = +0.247** (up from +0.080 for the same score on per-complex backbones), ≈ the
  published 0.30 and the strongest affinity signal in the arc. So the per-complex-cofold
  benchmark construction was washing out a real signal. **But target-heterogeneous**:
  +0.56/+0.45/+0.40/+0.39 on four targets, −0.44 on 1n8z, −0.19 on 1mhp — a lead, not a
  usable ranker, and the shared reference is still a predicted cofold.
- **Stage 2 (done):** the *true crystal* arm — thread onto the SAbDab crystal backbone
  behind an alignment gate (passed 1mhp/1n8z/4fqi; excluded 3gbn for a light-chain gap).
  **4fqi reaches ρ=+0.591 on the crystal** — higher than the predicted backbone and the
  strongest affinity signal in the arc — so inverse folding genuinely ranks affinity on
  a real native backbone. And **1n8z inverts on the crystal too** (−0.16), proving the
  inversion is a real target property, not a predicted-reference artefact.

**Net:** the arc's "no binding ranker" is now precisely bounded. A fixed-native-backbone
inverse-folding score DOES rank affinity for some targets (4fqi +0.59) and inverts for
others (1n8z) — a real signal, not a universal ranker. The limit is no longer "no signal
exists" but "we cannot yet predict which targets it works on," which is a better-posed
open problem than the arc started with. See
[EXPERIMENT_06](EXPERIMENT_06_proteinmpnn_transfer.md) §3.1–3.2.

### 11.6 Where this leaves the programme

The arc was commissioned to replace an imported ipTM threshold with a measured one and
to give filter-rule work something to validate against. It did the first and found that
the second is not available: **filter-rule tuning has no trustworthy signal to tune.**

Selection cannot be fixed by choosing a better existing scalar, which pushes the work to
[`IMPROVE_DESIGN.md`](../../IMPROVE_DESIGN.md) §5's training branch — where
[`finetuning/MILESTONE_01`](../finetuning/MILESTONE_01_label_validation.md) then
established that the supervised binding data to train on does not exist in `databases/`
either (five antigens of graded affinity; `bool` negatives mostly mislabelled
neutralization; antigen diversity collapsing 832 → 6 clusters).

**Those two findings together are the real constraint on this project**, and they are
better stated plainly than worked around. Of the honest options that remained, the
cheapest — reproduce the published inverse-folding baseline, to find out whether the gap
is the *metric* or the *structure* — has now been run (§11.5 / EXPERIMENT_06). **It ruled
the metric out**: ProteinMPNN, the stronger published scorer, reproduced ESM-IF1's weak
result (+0.097 vs +0.111) and came nowhere near 0.30. So no scorer swap rescues
selection **on the per-complex-backbone benchmark** — but the native-backbone arm
(§11.5 stage 1) then showed that protocol was itself the problem: a shared backbone
lifts the same score to +0.247, ≈ the published number. The gap was the scoring
*protocol*, not the metric or (necessarily) the structures.

That leaves two forward directions worth the effort, and one that is not:

- **Worth it:** the fixed-reference-backbone inverse-folding score — the one real
  affinity lead the arc produced (+0.247, but target-heterogeneous with a −0.44
  inversion). Confirm it with the crystal arm (stage 2) and characterise *when* it works
  before trusting it; and generate our *own* binding labels rather than hunting for
  existing ones (the supervised branch is blocked on their absence).
- **Not worth it:** more scoring metrics on per-complex backbones. Five metric families
  across three benchmarks, plus a sixth scorer, all failed there — the shortlist is
  exhausted, and stage 1 shows the limitation was the benchmark protocol, not the lack
  of yet another metric.

EXPERIMENT_05 makes the fallback concrete and non-defeatist. The best design-level
signals it found — ipTM and `delta_sasa_refolded` — are the ones **BoltzGen's own
ranking already uses**, and they predict structural self-consistency, exactly what an
existence filter should do. So "accept the funnel as an existence filter" is not a
retreat: the funnel already is one, using close to the best cheap metrics for that job.
What it is *not*, on all six experiments' evidence, is a selector for binding — and no
choice of scorer makes it one. The next effort belongs on generator quality, real
binding labels, or the native-backbone structure-quality check — not on more scoring.
