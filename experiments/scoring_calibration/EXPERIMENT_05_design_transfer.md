# Experiment 05 — Does any of this work on real designs?

Status: **complete · 2026-11-01** · sections 1, 2, 4, 5 written 2026-10-31 BEFORE
results existed (pre-registration in git `4383a77`); sections 3 and 6 filled from the run

Generated numbers will be at [`results/designs_comparison.md`](results/designs_comparison.md),
regenerable with `analyze_designs.py`. Nothing here is hand-transcribed.

> **Why this document is half-written.** Sections 1, 2, 4 and 5 describe the design,
> which was fixed before any Rosetta job launched. They are committed now, unfilled
> results and all, so the framing cannot be retrofitted once the numbers land. This
> benchmark has twice produced a "finding" that evaporated under a criterion chosen
> after the fact — shape complementarity's 13/17 → 16/28, and EXPERIMENT_04's extremes
> rescue — and writing the hypothesis down first is the cheapest guard against a third.

---

## 1. What was the experiment

Score 1,096 **real BoltzGen designs** from two campaigns with FastRelax +
`InterfaceAnalyzer` plus ESM-IF1, and ask whether any metric separates designs whose
refold matches the backbone that was designed from designs whose refold does not.

Every previous experiment in this arc scored either point mutants of real complexes
(the 28-parent paired benchmark) or real antibodies against real antigens (the
six-target ASD benchmark). **Neither is a design.** This is the first time the arc's
metrics meet the objects the pipeline actually produces.

## 2. What were we testing

[PLAN.md](PLAN.md) §9 states the arc's central unproven assumption plainly: that a
metric threshold separating real binders from real non-binders also separates good
designs from bad ones. It is what Bennett et al. and the autoresearch filter paper both
implicitly rely on, and this project had never tested it.

[EXPERIMENT_04](EXPERIMENT_04_packstat_transfer.md) turned it into something
falsifiable. It concluded packstat is an **interface-integrity detector** — 23/28
(p=0.0005) at telling a binding-retained point mutant from a binding-destroyed one,
but ρ=+0.058 and below its own null at ranking graded affinity. It also said outright
that it **could not establish that interpretation**: the extremes test that would have
shown it had no power at 6 targets, where the null control passed too.

So the hypothesis here is specific: **if packstat detects structural integrity, it
should separate intact designs from wrecked ones.** Design pools are mostly structural
failures — median `filter_rmsd` 8.03 on one target and 17.29 on the other, against a
≤2.5 bar — so this is the regime where such a detector should be easiest to see working.

A second question comes free: `ipTM`, `pTM` and `delta_sasa_refolded` are already
computed for every design at zero cost. If Rosetta adds nothing over them, then
milestone 2's distillation has nothing to distil that is worth 5 minutes a complex.

### Null control

`ll_antigen` — ESM-IF1 scoring chain A. The antigen's **sequence** is byte-identical
across every design of a target; its **refolded structure** is not. So this measures
*global refold quality*, and it is **expected to fire**. That is its purpose here: it
separates "the whole structure is bad" from "the interface specifically is bad." A
metric that does not beat it has demonstrated nothing interface-specific.

This is a different and harder null than the previous experiments used, where the null
was constructed to be unable to carry *any* signal. Here it carries real signal by
construction, and the bar is to beat it.

## 3. What was the result

1,096 complexes scored (655 + 441), every row `ok`, both null controls at chance
(`ll_antigen` 0.512 on target 1 — it did *not* fire, see below). AUC = P(a random
structurally-consistent design outscores a random wrecked one); 0.5 is chance.

**Target A (committed), by pre-registered verdict:**

| metric | AUC | verdict | note |
|---|---|---|---|
| ipTM *(free)* | 0.708 | **PASS** | best, and already computed |
| delta_sasa_refolded *(free)* | 0.675 | **PASS** | already computed |
| −dG_separated | 0.666 | **PASS** | |
| buried SASA | 0.646 | WEAK | |
| interface H-bonds | 0.615 | WEAK | |
| −dG/dSASA | 0.584 | FAIL | |
| *ll_antigen* | *0.512* | *(null)* | at chance |
| **packstat** | **0.508** | **FAIL** | **at chance, below its own null** |
| shape complementarity | 0.495 | FAIL | |
| pTM *(free)* | 0.489 | FAIL | |

The verdict column applies the two-target rule, but **packstat FAILs on target 1
alone** (0.508, below the null's 0.512), so the committed headline does not depend on
the withheld target.

**The withheld second target reinforced it (directions only, per §4).** There, packstat
was **worse than chance** — structurally consistent designs scored *lower* on packstat
than wrecked ones — while ipTM, −dG_separated, buried SASA and delta_sasa_refolded all
separated *more* strongly than on target 1. packstat and shape complementarity were the
two worst metrics on both targets. Its numbers are in the gitignored results TSV for the withheld target;
`analyze_designs.py --include-withheld` prints them.

So the final verdicts (both targets): **PASS** — ipTM, delta_sasa_refolded,
−dG_separated. **WEAK** — buried SASA, interface H-bonds. **FAIL** — everything else,
packstat and shape complementarity most emphatically.

### Three facts that matter more than the table

**1. packstat does not detect structural integrity in designs.** On real designs it is
at chance (target 1) and worse than chance (target 2). EXPERIMENT_04 scoped packstat to
an "interface-integrity detector" because it could not rank affinity; this shows it is
not even that, once the backbone is generated rather than near-native.

**2. The two best metrics are free.** ipTM and delta_sasa_refolded are already in every
design's metrics CSV at zero cost, and they beat every Rosetta term. The 97 core-hours
of FastRelax added −dG_separated (PASS) and buried SASA (WEAK), neither of which beats
free ipTM. For this label, the Rosetta pass bought nothing.

**3. The null stayed at chance, which sharpens the reading.** I predicted `ll_antigen`
would fire here because a globally bad refold degrades the antigen too. It did not
(0.512 on target 1). So `filter_rmsd` is not simply "global structure quality" that
every metric inherits — an antigen-refold score cannot see it, and the passing metrics
are reading something interface- or complex-specific rather than riding a global-quality
channel.

## 4. What dataset did we use

Two BoltzGen antibody campaigns with their **full rank range** of refolded structures
retained — 1:1 against their `all_designs_metrics.csv` rows, spanning ranks 1–N rather
than only the top-10 the pipeline already selected.

| | Target A (committed) | Target B (withheld) |
|---|---|---|
| designs in pool | 1,000 | 3,000 |
| refold CIFs retained | 1,000 | 3,000 |
| pass `filter_rmsd` ≤ 2.5 | 165 (16%) | 111 (4%) |
| median `filter_rmsd` | 8.03 | 17.29 |
| median `design_to_target_iptm` | 0.339 | 0.233 |
| complex size | 419 res | 439 res |
| ρ(ipTM, −`filter_rmsd`) | +0.340 | +0.348 |
| **sampled for scoring** | **655** | **441** |

**That retention was not supposed to exist.** PLAN.md §8b records, as a blocker on
re-ranking work, that only ~10 designs per campaign-variant have a retained CIF and
those are the ones BoltzGen already ranked top-10 — so a re-ranking could never be
tested for surfacing a design from the unranked remainder. That is true for the 23
epitope-downstream campaigns (zero `refold_cif` directories) and **false for these two**.
§8b's blocker stands for what it describes; it simply does not cover every pool on disk.

**Two targets, because one proves nothing.** The targets are unrelated antigens. Each
is analysed *within* target, so antigen identity — the confound that invalidated
EXPERIMENT_02's cross-complex column and inflated EXPERIMENT_03's first build — is
controlled by construction.

Target 2 is also a usefully *harder* pool: 4% passing against 16%. If a metric works on
one and not the other, that is informative rather than noise.

### The label, and what it is not

`filter_rmsd` — does BoltzGen's own refold of the design match the backbone it designed?
Computed by the pipeline, independently of anything Rosetta or ESM-IF1 sees.

It is **structural self-consistency, not binding.** Nothing in this experiment can show
that a metric picks binders; there are no binding labels for designs anywhere in this
project, and §9's risk is untouched. What it can show is whether a metric detects a
design being structurally wrecked — which is precisely the property EXPERIMENT_04
attributed to packstat and could not verify.

It carries real independent information: ρ(ipTM, −`filter_rmsd`) is +0.340 and +0.348 on
the two targets, two unrelated antigens agreeing to within 0.008.

### Target 2 is withheld from this repository

Its campaign is gitignored at the user's instruction, and so are all its derived
results (gitignored by the `.gitignore` rule covering the withheld target's
derived files). Its numbers are computed locally and
**excluded from the committed report**; only the direction it showed is stated.

**This means the verdicts rest partly on data that cannot be checked from this
repository.** That is a real weakness in reviewability and it is stated here, in
`analyze_designs.py`'s generated report, and in the report's own header, rather than
left for a reader to discover. `analyze_designs.py --include-withheld` prints those
numbers to stdout only.

## 5. How did we test the hypothesis

1. **Pre-registered the criteria before launching anything** — in
   `build_design_sets.py`'s docstring and restated as `CRITERIA` in
   `analyze_designs.py`, so the verdict is computed rather than narrated:

   - **PASS** — AUC ≥ 0.65 on **both** targets *and* above the null on both
   - **WEAK** — AUC ≥ 0.60 on the same terms
   - **FAIL** — anything else, **including "works on one target only"**

2. **Stratified sampling**: all passers, plus 3× as many non-passers drawn evenly
   across deciles of `filter_rmsd` above the bar. Deterministic, no RNG, so the manifest
   reproduces without a seed.

3. **Chose AUC, not a sign test, and not a rank correlation.** The paired benchmark used
   a sign test because it had 28 matched pairs; here there are hundreds of unpaired
   designs per target. AUC (Mann-Whitney, tie-corrected) is unbiased under stratified
   sampling of the negative class in a way a rank correlation over the whole sample is
   not — so the correlation is reported as **secondary and flagged as biased**.

4. **Verified the structures before trusting them**, which caught a mistake of my own:
   a first check read `label_entity_id` and reported chains "1 2 3". The real layout is
   A (antigen, invariant across designs) + B/C (designed, varying), confirmed by
   extracting sequences from 5 designs per pool, so `INTERFACE = "A_BC"` applies
   unchanged. Chain lengths differ *between* designs (B=109/C=126 on one, B=125/C=115 on
   another) because CDR length is what is being designed — which makes the per-design
   coordinate/sequence length check load-bearing, not decorative.

5. **Verified FastRelax works on design CIFs** before committing 97 core-hours:
   `dG_separated` −80.3 and −68.5, not the +616 a clash-dominated unrelaxed structure
   gives. Measured 5.3 min/complex, against 8.3 predicted from the ASD timings.

6. **Unit-checked the AUC implementation** against hand-computable cases — perfect
   separation 1.0, reversed 0.0, all-tied 0.5, identical distributions p=1.000.

7. **Carried the null through every statistic**, and set the bar at beating it rather
   than at beating chance.

### A bug this surfaced, worth recording

`compute_esmif.py` defaulted to `--device cuda`, which **can never work**. ESM-IF1's
`inverse_folding.util.get_sequence_loss` calls `batch_converter(batch)` with no device,
so its tensors stay on CPU while the model sits on the GPU, and it then calls `.numpy()`
on the loss, which a CUDA tensor would refuse anyway. Every earlier run passed
`--device cpu` by hand, so the broken default survived unnoticed until milestone 7
called the script without it. Now defaults to the only value that works.

## 6. What were the findings

**1. packstat is finished as a selection metric.** This is the headline, because
packstat was the arc's one positive result — 23/28 on the paired benchmark, the sole
thing that ever beat ipTM. On real designs it is at chance (0.508 on the committed
target, below its own null) and worse than chance on the other. The 23/28 was specific
to point mutants of near-native crystal backbones; it does not survive the backbone
being generated. EXPERIMENT_04 had already shown it does not rank affinity; this shows
it does not even detect structural integrity in designs. There is no longer any
question left on which packstat beats chance.

**2. Milestone 2 should be cancelled, not rescoped a second time.** It was going to
distil packstat as a cheap interface-integrity filter. There is nothing to distil: the
target property it was supposed to approximate is itself at chance on the objects the
funnel actually produces. Cancelling it is the finding, not a side effect.

**3. ipTM is the best predictor here — but of self-consistency, not binding.** The
metric dismissed throughout the arc as "at chance" (14/28 on the paired benchmark)
is the single best metric on this one (0.708). The reconciliation is the whole point:
the labels are different properties. ipTM tracks *structural plausibility* — it passes
on `filter_rmsd` (a structural-consistency label) and fails on binding-retention and
affinity. This is exactly EXPERIMENT_01's reading — "ipTM measures whether a credible
interface can be modelled, not interface quality" — confirmed from the opposite
direction. **ipTM passing here is not a resurrection of ipTM as a binding signal.**

**4. The Rosetta pass bought nothing over free metrics, for this label.** Best free
metric (ipTM, 0.708) beats the best Rosetta metric (−dG_separated / buried SASA, 0.666 /
0.646). 97 core-hours confirmed a result that two already-computed columns gave for
free. Worth stating because it is the second time an expensive Rosetta pass in this arc
returned a negative-or-redundant result (EXPERIMENT_04 was the first), and it bears on
whether Rosetta scoring earns its place in any future funnel: on current evidence, for
these two questions, it does not.

**5. The unified picture across all five experiments.** No metric predicts *binding*
(affinity in EXPERIMENT_03/_04, retention in EXPERIMENT_02) in a way that generalises.
*Structural self-consistency* is predictable (EXPERIMENT_05) — but it is not binding,
and the metric that predicts it best is already free. Every expensive avenue the arc
opened to find a better-than-free selection signal has closed.

### What this does and does not license

**Licensed**: dropping packstat and the planned milestone-2 distillation; treating
Rosetta interface scoring as unproven value in this funnel; using ipTM / delta_sasa as
cheap structural-plausibility filters, which is roughly what BoltzGen's own ranking
already does.

**Not licensed**: any claim that ipTM selects *binders*. The label here is structural
self-consistency, measured by the pipeline, not binding — there are still no binding
labels for designs anywhere in this project (§9), and nothing here touches that gap.

### Limitations (as written in §5, before results — unchanged by them)

The §5 limitations stand. The two that cut hardest in light of the result: the second
target's pool is 96% failures, so its *positive* separations (ipTM etc.) risk conflating
"detects gross damage" with "detects anything", which is why the committed claims lean
on target 1; and `filter_rmsd` is not binding, so even the clean PASS verdicts are about
structural plausibility, not the property the project ultimately needs.

### Limitations (stated before the results, so they are not shaped by them)

- **Two targets.** Better than one, far from general.
- **`filter_rmsd` is not binding.** A metric that aces this has been shown to detect
  structural integrity and nothing more.
- **Target 2's pool is 96% failures.** If a metric separates well there, "detects gross
  damage" and "detects anything at all" are hard to distinguish. Target 1, at 16%
  passing, is the more informative of the two, and the requirement that a metric work on
  both is what guards against over-reading either.
- **The negatives are stratified**, so the secondary rank correlation is biased by
  construction and only the AUC column is pre-registered.
- **One target's numbers are not in this repository** (§4), so the verdicts are not
  fully checkable from it.
- **A metric could pass here and still be useless in a funnel**, because `filter_rmsd`
  is already computed for free by the pipeline. Beating it is not the test; adding
  information beyond what is already free is, and that is what the `(free)` rows in the
  results table are for.
