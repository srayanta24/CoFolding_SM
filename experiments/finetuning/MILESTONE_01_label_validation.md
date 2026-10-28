# Milestone 1 — label validation: branch A is blocked

Status: **complete · 2026-10-28 · no GPU used**

[PLAN.md](PLAN.md)'s milestone 1 existed to answer one question before any training:
**does ASD's `bool` "non-binder" label mean what we need it to mean, and are its 832
both-class antigens genuinely diverse?**

Both answers are no. Branch A should not proceed as specified. This cost nothing to
establish, which is the entire point of putting it first.

---

## 1. The 832 antigens are 6 clusters

Clustered with MMseqs2 at the same 40% identity / 80% coverage used by
`databases/splits/`:

| cluster | antigens | source |
|---|---|---|
| 1 | **800** | HIV |
| 2 | 19 | covid-19 |
| 3 | 10 | covid-19 |
| 4, 5, 6 | 1 each | covid-19 / met |

**832 antigens collapse to 6 clusters.** 800 of them are HIV Env strain variants —
near-identical sequences, not independent targets.

A held-out-by-cluster split would train on 5 clusters and test on 1. That is not
meaningfully more diversity than AbBiBench's five antigens, which is the sole reason
branch A was chosen over the graded-affinity option in the first place.

## 2. The dominant label is the wrong property

Tracing `source_url` for every row in the three negative-bearing sources:

| source | negatives | primary source | what the label means |
|---|---|---|---|
| **hiv** | 22,708 | [DeepAAI, Nat Mach Intell 2022](https://www.nature.com/articles/s42256-022-00553-w) | **neutralization**, not binding |
| **covid-19** | ~27,696 | [AntiBinder, Brief Bioinform 2025](https://academic.oup.com/bib/article/26/1/bbaf008/7964432) + [CoV-AbDab](https://academic.oup.com/bioinformatics/article/37/5/734/5893556) | binder/non-binder, provenance partly undocumented |
| **met** | 2,484 | [Nat Commun 2022](https://www.nature.com/articles/s41467-022-31457-3) | FACS-sorted binding — **real**, 1 antigen |

Three problems, in order of severity:

**The largest negative source is labelled for neutralization.** DeepAAI predicts
*neutralizability*; its data is "Ab–Ag neutralization and non-neutralization". An
antibody can bind without neutralizing, so a substantial fraction of its 22,708
"non-binders" are binders. Training a binding model on them teaches the wrong property.

**The second-largest cites a method paper, not a primary source.** AntiBinder is an
antibody–antigen interaction *predictor*; ASD takes its evaluation set as ground truth.
Reading it for negative-generation methodology: it "references four datasets with both
interaction pairs and noninteraction pairs but provides no methodology section
explaining negative generation." Negatives of undocumented provenance, possibly
computational.

**The per-dataset antigen counts confirm the collapse.** AntiBinder's own figures:
COVID-19 27,324 pairs / **32 antigens**; MET 4,000 / **1 antigen**; HIV 24,004 / 940
antigens (strain variants). Consistent with the clustering above.

## 3. MET is the one clean subset, and it is unusable here

The only part of `bool` whose labels mean exactly what we need: a yeast-display library
of emibetuzumab variants sorted by FACS for high and low c-Met binding. One assay, one
paper, 2,484 / 1,516 balance. Better provenance than anything else in ASD.

Three blockers, checked directly:

| | |
|---|---|
| **no light chains** | 0 of 4,000 rows. Heavy variable domain only (115 aa). Our entire validated pipeline is Fab-based, so nothing here compares to the 28-parent or ASD benchmarks. |
| **antigen is 1,390 aa** | full-length c-Met → **1,505-residue complex**, above the 1,200 cost guard. At observed scaling that is ~15-20 min/complex, so 4,000 ≈ 1,000+ GPU-hours; even a 300 subsample is 75-100. |
| **one antigen, one lineage** | 4,000 distinct heavies, but 35/115 variable positions and 0.947 mean pairwise identity. Cannot train anything transferable. |

Its honest role would be a **high-power single-system test** — 4,000 measured points
against the 28 parents our paired result rests on. Not worth the GPU now: EXPERIMENT_03
already showed predicted-backbone quality degrades inverse-folding scores badly on large
antigens, and 1,390 residues would be the largest antigen we have attempted.

**Worth recording as a route back in**: truncating c-Met to its antibody-binding domain
(the SEMA/PSI region emibetuzumab targets) would cut the complex to a tractable size and
make this a genuinely useful benchmark. That is a real option, not a dead end.

## 4. Verdict

**Branch A is blocked.** It rested on "832 antigens of genuine diversity with
binder/non-binder labels". The diversity is ~6 clusters and the dominant label is
neutralization. Neither half survives.

What remains of Option 3, in order of attractiveness:

1. **Branch B** (structure-conditioned, SAbDab `train_era` sequence recovery). Small
   expected return — it is what AntiFold already did, and AntiFold gained only 13/28 →
   16/28, still below the null. But it is leak-free, pipeline-compatible and cheap.
2. **AbBiBench graded affinity**, leave-one-target-out over 5 targets. Narrow and
   honest, and directly comparable to published baselines. The 5-antigen limit that
   made it unattractive is no longer *worse* than branch A's effective 6 clusters.
3. **Branch D** (fine-tune BoltzGen's own inverse folder). Unchanged blockers: 8 GPUs,
   no PEFT, data adapter needed.
4. **MET with a truncated antigen** — speculative, but the only clean-label data we have.

**The honest summary is that Option 3 has no well-supported training set.** Every
supervised binding signal in `databases/` is either too narrow (5 antigens or 1), the
wrong property (neutralization), or of undocumented provenance. That is a finding about
the data, not a failure of the plan, and it is better known now than after training a
model on it.

## 5. The ASD pattern, stated as a rule

This is the **fourth** ASD data trap found in a week:

1. `-log KD` is mislabelled on 2 of 8 targets (values down to −4.1, i.e. not a Kd).
2. Graded affinity exists for 5 antigens (`-log KD`) and 3 (`alphaseq`).
3. All 4,040 structure-bearing `bool` rows are positives — no structural negatives exist.
4. The `bool` negatives are mostly neutralization labels, and its antigen diversity
   collapses 832 → 6 clusters.

Each would have produced a confidently wrong result, and each was caught by a cheap
check. **ASD's aggregation is convenient, but its labels, provenance and diversity must
be verified every single time before use.** `databases/asd/README.md` should carry this
warning.
