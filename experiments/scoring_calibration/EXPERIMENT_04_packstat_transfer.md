# Experiment 04 — Does packstat's 23/28 transfer to affinity ranking?

Status: **complete · 2026-10-31 · 1,201 core-hours of FastRelax**

Generated numbers: [`results/asd_comparison.md`](results/asd_comparison.md) /
[`.json`](results/asd_comparison.json), regenerable with `analyze_asd.py`.
Nothing below is hand-transcribed.

**One-line answer: no.** packstat ranks graded affinity at ρ = +0.058 pooled, which is
*below* its own null control (+0.070). The single metric this project had found to beat
ipTM does not generalise beyond the question it was found on.

---

## 1. What was the experiment

Score all 1,530 complexes of the six-target within-antigen benchmark
([EXPERIMENT_03](EXPERIMENT_03_within_antigen_ranking.md)) with the full Rosetta
interface panel — FastRelax followed by `InterfaceAnalyzer` — and ask whether the
Rosetta metrics, **packstat above all**, rank antibodies by measured affinity against a
fixed antigen.

This is the expensive half of EXPERIMENT_03, deliberately deferred until the cheap
metrics had been scored. It cost **1,201 core-hours** (1,530 complexes, mean 47.1
min each, range 1.4–144 min), and it is the largest single compute expenditure in the
project so far.

## 2. What were we testing

A specific, pre-registered claim about a specific metric.

packstat is the **only** metric in this project that has ever beaten ipTM on a test
that controls for complex identity: 23/28 on the 28-parent paired benchmark, p=0.0005,
surviving Bonferroni, 8/11 on the independent new parents, and verified not to be
riding the null's quality channel. Everything else tested — ipTM, pTM, ipSAE, ESM-IF1,
AntiFold, shape complementarity, buried SASA — performed at or below chance there.

Three consequences made it worth 1,201 core-hours to test again:

1. **It is the basis of milestone 2.** [`finetuning/PLAN.md`](../finetuning/PLAN.md)
   §4b proposes distilling relaxed packstat from unrelaxed structure, because packstat
   works and costs 22–90 min/complex against BoltzGen's 60,000-design protocol. If
   packstat only works on one benchmark, that milestone is distilling a narrow
   detector, not a selection signal.
2. **A single significant result on a single benchmark is the project's known failure
   mode.** Shape complementarity and buried SASA were both 13/17 (p=0.025) and were
   reported as findings; both collapsed to chance at 28 parents. packstat had not yet
   faced an independent benchmark.
3. **The two benchmarks ask genuinely different questions**, which makes this a real
   test rather than a replication:

| | paired benchmark (23/28) | this benchmark |
|---|---|---|
| contrast | binding-**retained** vs binding-**destroyed** point mutant | functional binders differing in affinity |
| label | binary, large effect | graded −log KD |
| controls for | complex identity (same parent) | antigen identity (same target) |
| n | 28 parents | 6 targets, 1,530 complexes |

The hypothesis under test was therefore: **packstat reads interface quality in a way
that generalises from "is this interface intact" to "is this interface better".**

Null control, as in every experiment here: `ll_antigen`, ESM-IF1 scoring the antigen
chain. Within a target the antigen is byte-identical across all entries, so it cannot
carry affinity information. It changed the conclusion in EXPERIMENT_02 and
EXPERIMENT_03, and it changes it again.

## 3. What was the result

Pooled Spearman against −log KD, z-scored within target so no target dominates. Higher
is better; every expected correlation is positive. `ρ partial` removes the null from
both sides.

| metric | ρ | ρ partial | verdict |
|---|---|---|---|
| pTM | +0.141 | +0.131 | best available, still weak |
| ipTM | +0.133 | +0.124 | |
| −dG_separated | +0.125 | +0.119 | best Rosetta term |
| shape complementarity | +0.112 | +0.104 | |
| ESM-IF1 ll_heavy | +0.111 | +0.105 | |
| −dG/dSASA | +0.100 | +0.095 | |
| *ll_antigen* | ***+0.070*** | — | **null control** |
| buried SASA | +0.065 | +0.062 | below null |
| complex pLDDT | +0.064 | +0.056 | below null |
| **packstat** | **+0.058** | **+0.054** | **below null** |
| interface H-bonds | +0.039 | +0.035 | below null |
| ESM-IF1 ll_light | −0.035 | −0.040 | below null |

**packstat is below the null control.** So is every other geometric Rosetta term except
shape complementarity. The best Rosetta term is an *energy* (−dG_separated), not a
packing measure — the reverse of the paired benchmark's ordering.

Nothing clears ρ = +0.15. AbBiBench's published baselines on this same data are
ProteinMPNN ρ ≈ 0.30 and ESM-IF1 ≈ 0.28, i.e. roughly double the best column here.

Per target, with the threshold for p<0.05 shown (|ρ| > 1.96/√(n−1)):

| metric | 1mhp (n=30) | 1n8z | 3gbn | 4fqi | aayl49 | aayl51 |
|---|---|---|---|---|---|---|
| threshold | 0.364 | 0.113 | 0.113 | 0.113 | 0.113 | 0.113 |
| ipTM | −0.110 | **+0.209** | **+0.289** | +0.024 | +0.083 | +0.042 |
| packstat | −0.055 | −0.013 | +0.108 | **+0.182** | −0.008 | +0.050 |
| shape compl. | −0.097 | −0.069 | **+0.401** | **+0.117** | +0.085 | +0.075 |
| *ll_antigen* | *+0.288* | *−0.094* | ***+0.241*** | *+0.051* | *+0.052* | *+0.100* |

packstat reaches significance on exactly one target of six, and it is not the same
target ipTM or shape complementarity succeed on.

**The null control is significant on 3gbn (+0.241)** — the one target where ipTM
(+0.289) and shape complementarity (+0.401) look strong. Their apparent signal there
sits on top of a channel that provably cannot encode affinity, so 3gbn is the target
whose numbers deserve least weight, not most.

### The obvious rescue, tested and refused

If packstat detects gross interface damage rather than affinity gradation, it should
recover when only the extremes are compared. Scored exactly as the paired benchmark is
— a sign test over groups — at three cut depths:

| cut | ipTM | packstat | shape compl. | *null* |
|---|---|---|---|---|
| top/bottom decile | 5/6 (p=0.109) | 5/6 (p=0.109) | 6/6 (p=0.016) | *5/6 (p=0.109)* |
| top/bottom quartile | 6/6 (p=0.016) | 4/6 (p=0.344) | 4/6 (p=0.344) | *5/6 (p=0.109)* |
| median split | 5/6 (p=0.109) | 4/6 (p=0.344) | 4/6 (p=0.344) | *5/6 (p=0.109)* |

**The rescue fails, and the null is why.** With 6 targets the sign test has so little
power that the null control wins 5/6 at two of three cut depths, with per-target effect
sizes as large as the real metrics. Shape complementarity's 6/6 (p=0.016) collapses to
4/6 one cut deeper. No row in this table is evidence of anything.

This is the identical lesson to the 13/17 → 16/28 collapse: **a sign test over a
handful of groups is not a finding, and the null control is what reveals it.** Having
been caught by this once, I ran the null before reading the table rather than after.

## 4. What dataset did we use

The six-target ASD within-antigen benchmark built in EXPERIMENT_03, unchanged — reused
deliberately so this is a new metric on a known benchmark rather than two things
changing at once.

| target | n | antigen | −log KD source |
|---|---|---|---|
| 1mhp | 30 | — | AbBiBench |
| 1n8z | 300 | HER2 | AbBiBench |
| 3gbn | 300 | influenza HA | AbBiBench |
| 4fqi | 300 | influenza HA | AbBiBench |
| aayl49 | 300 | — | AlphaSeq |
| aayl51 | 300 | — | AlphaSeq |

Each target is **one antigen, many antibodies, heavy chain only varying**. 2fjg and
1mlc are excluded: their "−log KD" values run to −4.1, which cannot be a Kd, and are
almost certainly ΔΔG-like with inverted sign (EXPERIMENT_03 §4 — the first of the four
ASD data traps).

Four result tables, all 1,530 rows and column-aligned:
`asd_scores.tsv` (BoltzGen confidences), `asd_esmif.tsv` (ESM-IF1 + the null),
`asd_ipsae.tsv` (ipSAE), `asd_rosetta.tsv` (**new** — the 1,201 core-hours).

## 5. How did we test the hypothesis

1. **Scored every complex with FastRelax + InterfaceAnalyzer.** `compute_rosetta.py`,
   16 workers, PyRosetta in `.venvs/germinal`. FastRelax is **not** optional: skipping
   it was tested on the paired benchmark and destroys the signal (23/28 → 15/28, raw-vs-
   relaxed rank correlation only +0.507), which is precisely why this cost 1,201
   core-hours instead of 1.
2. **Pooled with within-target z-scoring**, so a target with a different absolute
   metric scale cannot dominate the pooled ρ.
3. **Carried the null control through every statistic**, including the pooled table,
   the per-target table and the extremes test.
4. **Took the first-order partial** of each metric against affinity with the null
   removed from both sides. Here the partials barely move (+0.133 → +0.124 for ipTM),
   which is itself informative: the null's +0.070 is largely *orthogonal* noise rather
   than a shared channel, so the weak correlations are weak on their own account and
   not merely null-contaminated.
5. **Stated the per-target significance threshold** (|ρ| > 1.96/√(n−1)) so the
   per-target table could not be read as six independent successes.
6. **Pre-registered the extremes test as the rescue** before running it, so that a
   null-passing result could not be reinterpreted as support afterwards.

All of it lives in `analyze_asd.py` and regenerates from the TSVs. The project
convention — JSON is the contract, markdown is the view, nothing hand-transcribed —
exists because the 28-parent result once survived only as prose in a commit message.

## 6. What were the findings

**1. packstat does not generalise.** +0.058 pooled, below its own null, significant on
one target of six. The one metric that beat ipTM did so on one benchmark asking one
question, and that is now the honest scope of the 23/28 result.

**2. The precise scope of packstat's success is narrower than "interface quality".** It
discriminates binding-retained from binding-destroyed mutants of the same parent. It
does not rank functional binders by affinity. The mechanistic reading — that it detects
gross packing defects rather than gradations of fit — remains the most plausible one and
is consistent with the FastRelax result (packstat needs clashes resolved before it reads
anything), but **this benchmark cannot establish it**, because the extremes test that
would have shown it has no power at 6 targets.

**3. Milestone 2 needs rescoping, not cancelling.** Distilling relaxed packstat from
unrelaxed structure is still well-posed — the target is deterministic, needs no binding
labels, and ~1,900 training examples now exist. What changes is the claim: it would
yield a cheap **interface-integrity detector**, useful as an early funnel filter to
discard wrecked designs, not a ranker for the final selection. The bar stated in §4b —
reproduce ~23/28 on the paired test — is still the right bar, and it is now explicitly
a bar on *that* question only.

**4. The within-antigen affinity-ranking question remains unanswered by anything we
have.** Best column is pTM at +0.141 against published baselines of ≈0.30. Four metric
families — BoltzGen confidences, ipSAE, Rosetta energies and geometry, inverse-folding
likelihoods — have now been tested on this benchmark and none is usable for ranking
affinity. This is the fifth experiment to reach the same conclusion: **this pipeline has
no trustworthy selection signal.**

**5. The six-target benchmark is underpowered for group-wise tests** and should only be
used for pooled rank correlation, where n = 1,530. Any future sign test on it must
report the null alongside, and 3gbn's numbers must be discounted specifically.

**6. The 1,201 core-hours were worth spending**, and this is worth stating plainly
because the result is negative. Had packstat been assumed to generalise, milestone 2
would have trained a model to approximate it, evaluated it on the paired benchmark it
was derived from, reproduced ~23/28, and been reported as a cheap general-purpose
selection metric. The error would have been invisible — exactly the shape of the four
ASD data traps and the two schema misalignments.

### Where this leaves the programme

The scoring-calibration arc is **complete**: five experiments, two benchmarks, four
metric families, and a consistent answer. No cheap unsupervised scalar ranks antibody
designs. The remaining honest options are the ones
[`IMPROVE_DESIGN.md`](../../IMPROVE_DESIGN.md) §5 lists under the training branch, all
of which require putting binding knowledge into weights — and
[`finetuning/MILESTONE_01`](../finetuning/MILESTONE_01_label_validation.md) established
that the supervised data to do that does not exist in `databases/` either.

That is not a dead end, but it is a genuine constraint, and it is better stated than
worked around.

### Limitations

- **Six targets, two of them (aayl49/aayl51) from AlphaSeq rather than measured Kd.**
  Different assay, different noise floor.
- **1mhp has n=30**, so its threshold for significance is |ρ| > 0.364 and nothing there
  can be read.
- **Predicted backbones throughout.** EXPERIMENT_03 showed inverse-folding scores
  degrade badly on large antigens from predicted structure; the Rosetta terms are
  computed on the same predicted-then-relaxed structures and inherit that.
- **A negative result on a weak benchmark is weaker evidence than a positive one.**
  packstat failing here is consistent with "packstat is narrow" and with "this
  benchmark cannot resolve any metric". The pooled ρ ≈ 0.30 published baselines argue
  the benchmark has *some* resolving power, but we did not reproduce those baselines
  ourselves, so that argument is borrowed rather than demonstrated.
