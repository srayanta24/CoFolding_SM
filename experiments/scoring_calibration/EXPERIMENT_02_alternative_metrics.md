# Experiment 02 — Is there a better selection metric than ipTM?

Status: **complete** · run 2026-10-18/19 · 332 structures × 3 independent metric families

Follow-up to [EXPERIMENT_01](EXPERIMENT_01_iptm_calibration.md), which found that ipTM
does not track binding affinity. This asks the obvious next question: does anything else?

Everything here reuses EXPERIMENT_01's structures. No new structure prediction was run.

> **Follow-up, 2026-10-24.** §6's conclusion that "cross-complex affinity ranking
> remains **untested**, not disproven" has now been tested directly, on a benchmark
> built to remove the antigen-identity confound — see
> [EXPERIMENT_03](EXPERIMENT_03_within_antigen_ranking.md). Short version: ipTM does
> carry some within-antigen signal (pooled ρ = +0.194), but the same null-control
> technique used here fires on that benchmark too (+0.105 pooled, +0.284 on one
> target), so most of it is not interpretable. ESM-IF1 reaches +0.078 against its
> published ≈0.28 on the same data, leaving this document's backbone-quality
> hypothesis as the leading explanation but unquantified.

> **Correction, 2026-10-19.** An earlier version of this document (and of
> IMPROVE_DESIGN.md §4) claimed packstat and shape complementarity are "already computed
> inside BoltzGen's analysis stage, so adopting them costs a `--metrics_override`." That
> is **wrong**, verified against a real campaign's 197-column metrics CSV: `packstat`
> and `sc_value` appear nowhere in it. Both come only from Rosetta's
> `InterfaceAnalyzerMover`, so adopting them means paying the 22.7 min/structure
> Rosetta cost — the very thing finding 3 rules out funnel-wide.
>
> What *is* free is narrower but still useful: **buried SASA (13/17, p=0.025) is
> available as `delta_sasa_refolded`**, and it is already weight-2 in BoltzGen's own
> default ranking. So the free change is re-weighting the existing six metrics —
> toward `delta_sasa_refolded`, away from `design_iptm` — not adding new columns.

---

## 1. What was the experiment

EXPERIMENT_01 left the funnel without a trustworthy selection signal. `design_to_target_iptm`
is weight-1 in BoltzGen's own ranking and the metric this project gates on, and it was
shown to be blind to the affinity continuum. The remedy is only useful if something
*else* works, so this experiment scores the same 332 complexes with three deliberately
**independent** families of metric and compares them head to head against ipTM:

| tier | metric family | independent of ipTM how? | cost |
|---|---|---|---|
| 0 | **ipSAE** | *Not* independent — same PAE matrix, reweighted | free (PAE already saved) |
| 1 | **Rosetta** interface energetics | physics on the coordinates | 17 h CPU |
| 2 | **ESM-IF1** inverse-folding likelihood | learned sequence-given-structure | 42 min CPU |

Tier 0 was included precisely *because* it is not independent: if reweighting the PAE
had worked, that would have been the cheapest possible fix.

## 2. What were we testing

Two questions, which turn out to need completely different treatment:

**H1 — cross-complex affinity ranking.** *Does the metric rank different
antibody–antigen complexes by their measured Kd?* Tested over 100 Kd-labelled complexes
spanning nine orders of magnitude.

**H2 — within-complex discrimination.** *On a single parent complex, does the metric
score a binding-retained point mutant above a binding-destroyed one?* Tested on the 17
parent complexes that carry both.

H2 is the operationally relevant one and this was not obvious at the outset. **A design
campaign is always a within-complex problem**: BoltzGen generates thousands of candidates
against *one* target and must rank them against each other. Cross-complex ranking is a
question nobody in this pipeline actually asks.

**A null control was added to tier 2** and became the most important part of the
experiment: ESM-IF1 also scores the **antigen** chain. The antigen is byte-identical
between a retained and a destroyed mutant of the same parent, so it *cannot* carry real
signal. Anything it appears to detect is measuring an artefact of the test.

## 3. What was the result

Metrics oriented so higher = predicted better binder. `ll_antigen` is the null control.

Regenerate with `analyze_metrics.py` → [results/metrics_comparison.md](results/metrics_comparison.md)
and its JSON sidecar. Every number below comes from there.

| metric | cross-complex ρ (want < 0) | unpaired AUC | **paired /17** | binomial p |
|---|---|---|---|---|
| ipTM (baseline) | +0.134 | 0.634 | 9/17 | 0.500 |
| ipSAE_min | +0.198 | 0.626 | 8/17 | 0.685 |
| −dG/dSASA (Rosetta) | +0.179 | 0.634 | 12/17 | 0.072 |
| −dG_separated | +0.145 | 0.644 | 12/17 | 0.072 |
| **packstat** | +0.003 | 0.624 | **14/17** | **0.006** |
| **shape complementarity** | +0.054 | 0.541 | **13/17** | **0.025** |
| **buried SASA** | +0.048 | 0.587 | **13/17** | **0.025** |
| interface H-bonds | +0.049 | 0.525 | 10/17 | 0.315 |
| −unsat H-bonds | +0.131 | 0.447 | 6/17 | 0.928 |
| ESM-IF1 ll_antibody | **−0.037** | 0.634 | 8/17 | 0.685 |
| ESM-IF1 ll_heavy | −0.014 | 0.602 | 8/17 | 0.685 |
| ESM-IF1 ll_light | −0.061 | 0.606 | 9/17 | 0.500 |
| *ESM-IF1 ll_antigen (null control)* | *+0.341* | *0.396* | *9/17* | *0.500* |

Two rows the first hand-transcribed pass missed, both recovered by writing the analysis
as a script:

- **Buried SASA also reaches 13/17 (p=0.025)**, tying shape complementarity. It had been
  dismissed on its weak unpaired AUC (0.587) — i.e. dismissed on a column now known to
  be confounded. There are **three** significant geometric metrics, not two.
- **−unsat H-bonds scores 6/17 (p=0.928)** — meaningfully *worse* than chance, i.e.
  binding-destroyed mutants tend to have *fewer* unsatisfied H-bonds than retained ones.
  Plausibly because a disrupted interface buries less surface and so has fewer polar
  groups to leave unsatisfied, making the raw count a size proxy rather than a quality
  measure. Not investigated further; flagged so nobody adopts it expecting the obvious
  sign.

**The control fires on two of the three columns.** A quantity that cannot possibly
carry signal shows the *strongest* cross-complex correlation of anything tested
(+0.341) and an unpaired AUC well away from chance (0.396). Both of those columns are
therefore measuring something other than what they claim to.

Only the paired column behaves correctly: the control sits at exactly chance (9/17),
which is what a null must do.

## 4. What dataset did we use

Identical to EXPERIMENT_01 — `sets/manifest.tsv`, 332 scored entries (32 leak-free
`clean_positive`, 100 Kd-labelled `contaminated_positive`, 100 `hard_negative`,
100 `matched_positive`). See that document for construction, scope limits and the
AB-Bind filtering. Every metric here was computed on the **same predicted structures**
produced by EXPERIMENT_01, so no tier can differ because it saw different inputs.

## 5. How did we test the hypothesis

**Tier 0, ipSAE** (`compute_ipsae.py`). Transcribed from the vendored reference
(`src/ipsae`, now registered in `setup_env.py`) rather than recalled; `d0` and
`ptm_func` verified character-for-character against it. One deliberate deviation: stock
ipsae.py scores chain *pairs*, but an antibody paratope spans heavy and light, so the
primary score groups B+C against A to stay comparable with `design_to_target_iptm`.
Run at PAE cutoffs 10 (Dunbrack's stated operating point) and 15.

**Tier 1, Rosetta** (`compute_rosetta.py`). FastRelax with
`constrain_relax_to_start_coords`, then `InterfaceAnalyzerMover("A_BC")`. Relax is not
optional, and this was measured rather than assumed:

    raw Boltz structure   dG_separated = +616.57   hbonds = 0
    after FastRelax       dG_separated =  -30.47   hbonds = 12

A positive 600-unit "interface energy" is not a weak interface, it is a meaningless
number. Scoring unrelaxed would have produced a full column of plausible-looking
garbage. 16 workers, ~15 min/structure, 17 h wall-clock, 332/332 succeeded.

**Tier 2, ESM-IF1** (`compute_esmif.py`). `score_sequence_in_complex` scores one chain
conditioned on the whole complex backbone, so a mutation is judged *in the presence of
the antigen*. Heavy and light scored separately plus their mean; antigen scored as the
null control.

**Statistics.** Distribution-free throughout. Cross-complex is Spearman against
log10(Kd); separation is Mann–Whitney U as an AUC; the paired test is a sign test over
parent complexes with an exact one-sided binomial p.

### ipSAE cutoff sensitivity

Dunbrack's PAE<10 operating point was calibrated on AlphaFold-with-MSA. Ours are no-MSA
Boltz-2 predictions with systematically higher PAE, so the cutoff was swept:

| cutoff | entries collapsing to exactly 0.0 | cross-complex ρ | unpaired AUC |
|---|---|---|---|
| 10 | **43%** | +0.186 | 0.632 |
| 15 | 22% | +0.198 | 0.626 |
| 15, single direction (ab→ag) | 3% | +0.204 | 0.617 |

Relaxing the cutoff halves the degenerate mass without improving ranking in either
column. The zeros arise because `ipsae_min` takes the minimum of two *asymmetric*
directions, and for many entries one direction has no interchain pair below the cutoff
at all. That is the metric behaving as designed — conservative — but it means ipSAE_min
carries a large block of tied values on this data, which is worth knowing before anyone
adopts it for ranking.

### Environment obstacles, recorded so they are not rediscovered

Each of these cost real time and none is documented upstream:

- **`.venvs/epiformer` has no pip.** It was created with `uv`, so installs need
  `uv pip install --python .venvs/epiformer/bin/python3 <pkg>`.
- **biotite 1.x broke fair-esm.** `esm.inverse_folding` imports `filter_backbone` at
  module level, renamed to `filter_peptide_backbone` in biotite 1.x. Shimmed in one line
  rather than downgrading, because pulling biotite back would likely have dragged numpy
  and therefore torch with it — the exact failure mode DESIGN.md §2 records this project
  hitting once before. Verified torch/numpy/CUDA unchanged after the install.
- **ESM-IF1's scorer ignores device.** `score_sequence_in_complex` builds its tensors on
  CPU regardless of where the model sits, so a CUDA model raises a device mismatch. CPU
  runs at 7.6 s/entry, fast enough that patching third-party internals was not worth it.
  Noted because the obvious "just move it to GPU" does not work.
- **Piping a progress stream through `grep` hides it.** `grep` buffers when not writing
  to a terminal, so `script.py | grep '^\['` showed nothing for 20 minutes while the
  job ran fine. Check the output file the script writes, not the pipe.

### Why two of the three analysis columns are invalid, demonstrated not asserted

The control's behaviour prompted two direct checks, and both confirmed a confound:

- **Unpaired AUC is confounded by parent composition.** `matched_positive` draws on 27
  parent complexes, `hard_negative` on 24, and only **17 are shared**. Half the parents
  appear in just one group, so the unpaired comparison partly measures *which complexes
  landed in which arm* rather than the mutation's effect. That is exactly why the
  antigen control scores 0.396 instead of 0.5.
- **Cross-complex ρ is confounded by antigen identity.** Among the 100 Kd-labelled
  entries there are 89 distinct antigens, and log10(Kd) spread *within* a repeated
  antigen is **1.73** against a total spread of **7.97**. Roughly four-fifths of the
  affinity variance is between antigens rather than within them, so **any antigen-only
  feature correlates with Kd spuriously** — which is precisely what `ll_antigen`'s
  +0.341 is.

This also retrospectively explains EXPERIMENT_01's most striking number. The
"monotonically backwards" dose-response was read there as a property of ipTM. It is
better explained as a property of **the benchmark**: cross-complex affinity ranking is
not cleanly testable on a corpus assembled this way, whatever metric is used.

**EXPERIMENT_01's headline needs this correction carried with it.** The claim "ipTM does
not track binding affinity" is not refuted — ipTM still fails the valid within-complex
test at exactly chance — but the *evidence* originally offered for it was partly an
artefact. The conclusion survives; one of its two supporting arguments does not.

## 6. What were the findings

**1. Within-complex, geometric interface quality beats everything else — including
ipTM, which is at chance.**

    packstat                14/17   p = 0.006
    shape complementarity   13/17   p = 0.025
    -dG/dSASA, -dG_sep      12/17   p = 0.072
    ipTM                     9/17   p = 0.500   <- chance
    ipSAE_min                8/17   p = 0.685   <- chance
    ESM-IF1 ll_antibody      8/17   p = 0.685   <- chance
    ll_antigen (control)     9/17   p = 0.500   <- chance, as required

The funnel is currently ranked by a metric that performs at chance on the only
comparison it is actually used for, while two cheap geometric terms do measurably
better.

**2. All three "independent" families failed to beat ipTM at the headline task, for
three different reasons** — and the reasons matter more than the shared outcome:

- **ipSAE** reweights the same PAE matrix, so it inherits ipTM's blindness by
  construction. 43% of entries collapse to exactly 0.0 at Dunbrack's PAE<10 cutoff,
  because that was calibrated on AlphaFold-with-MSA and our no-MSA Boltz-2 PAEs run
  higher. *Lesson: no PAE-derived metric can rescue a PAE-derived metric.*
- **Rosetta** is genuinely independent and does win the valid test — but only through
  its geometric terms (packstat, shape complementarity), not its energy terms, which
  sit at a marginal p = 0.072.
- **ESM-IF1** is the only metric with the correct *sign* on cross-complex ρ (−0.037),
  but the magnitude is negligible against AbBiBench's reported ≈0.28. Most likely cause:
  AbBiBench scores mutants on **crystal** structures, whereas these are no-MSA Boltz-2
  **predictions**, and an inverse-folding likelihood is only as good as the backbone it
  is conditioned on. Untested here.

**3. Cost rules out physics as a funnel-wide filter, independent of whether it works.**
Measured: Rosetta averaged **22.7 min/structure, 125.5 CPU-hours** total (17 h
wall-clock only because it ran on 16 cores). Generating one design costs ~3 min of GPU,
so rescoring consumes roughly **7.5× the compute of generation** — the two sit on
different hardware, so treat that as an order-of-magnitude statement rather than a
clean ratio. Either way, at BoltzGen's 60,000-design protocol it is impossible here; it
can only rescore a final top-N.

(An earlier draft said "~15 min/structure, 5×" from the single-threaded smoke test.
Under 16-way parallelism per-structure time rose to 22.7 min from memory-bandwidth
contention — worth knowing before anyone plans capacity from a single-threaded
measurement.)

ESM-IF1 at 7.6 s and ipSAE at ~0 are funnel-viable, but neither works. **packstat,
shape complementarity and buried SASA are the fortunate case: they win the valid test
*and* are already computed inside BoltzGen's analysis stage**, so adopting them costs
nothing but a `--metrics_override`.

**4. The null control was the highest-value part of the experiment.** It cost one extra
function call and it invalidated two of three analysis columns — including the one that
produced EXPERIMENT_01's headline. Without it, this document would have confidently
reported "Rosetta dG_separated achieves AUC 0.644, the best of any metric," which the
parent-composition confound shows is not a real result. Every future comparison in this
project should carry a quantity that cannot work, for exactly this reason.

### Limitations

- 17 parent complexes is thin. packstat's p = 0.006 is nominal and uncorrected for the
  ten metrics tested; under Bonferroni (α = 0.005) it would not survive. Treat it as a
  strong lead requiring confirmation, not a settled result.
- Point mutants are a far subtler perturbation than two independently generated designs.
  Whether a within-complex advantage transfers to real design diversity is untested —
  that is milestone 7, now with a concrete hypothesis.
- ESM-IF1's underperformance against its published numbers is attributed to predicted
  rather than crystal backbones, which is plausible but unverified. Re-running it on the
  deposited structures of the same complexes would settle it cheaply.
- Cross-complex affinity ranking remains **untested**, not disproven. Testing it needs a
  corpus with many antibodies per antigen across a wide affinity range — a different
  benchmark from this one.
