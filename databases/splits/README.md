# `databases/splits/` — the leak-free evaluation splits

Status: **verified · 2026-10-31** · built by
[`databases/src/build_splits.py`](../src/build_splits.py) · rebuild with the
[`splits` skill](../../.claude/skills/splits/SKILL.md)

Partitions every antigen-bound structure in [`databases/sabdab/`](../sabdab/) into a
set we must assume BoltzGen already memorised and a set it cannot have seen. **Any
number this project reports about a trained or tuned model has to be traceable to one
of these files**, or it is unfalsifiable — BoltzGen was trained on the PDB, so scoring
it on a pre-2023 complex measures recall, not generalisation.

Unlike every other `databases/<name>/` folder, everything here is **derived** — no
downloads, no external source. It is reproducible from `databases/sabdab/` alone, and
the dev/test assignment is hash-based rather than random-seeded, so a rebuild from a
clean directory reproduces these exact files.

---

## 1. The files

| File | Rows | What it is |
|---|---|---|
| `train_era.txt` | **8,072** | Released **on or before 2023-06-01**. Presumptively inside BoltzGen's training data. Use for training; never for evaluation. |
| `dev.txt` | **100** | Post-cutoff, antigen-cluster-clean. **Iterate against this.** |
| `test.txt` | **751** | Post-cutoff, antigen-cluster-clean. **Reserved** — touch once, at the end. |
| `test_cluster_disjoint.txt` | **554** | Subset of `test.txt` sharing no antigen cluster with `dev.txt`. See §4 — use this one after any dev-based tuning. |
| `excluded_ambiguous.txt` | **2,244** | Post-cutoff, but the antigen clusters with a `train_era` antigen at 40% identity. Excluded rather than trusted. |
| `excluded_no_sequence.txt` | **291** | Post-cutoff, but we could not extract an antigen sequence, so redundancy was never checked. Excluded rather than assumed safe. |
| `antigen_clusters.tsv` | 9,241 | MMseqs2 assignment, `representative → member`. 1,826 clusters. Kept so `excluded_ambiguous` entries can be audited by hand. |
| `splits_summary.json` | — | Every count below, plus all 2,217 extraction failures with reasons. The machine-readable contract. |

8,072 + 100 + 751 + 2,244 + 291 = **11,458**, the total number of unique antigen-bound
structures in SAbDab. Every structure lands in exactly one bucket.

`dev` + `test` = **851 confirmed binders** — real, solved complexes that BoltzGen has
not seen. That number is the one quoted in
[`scoring_calibration/PLAN.md`](../../experiments/scoring_calibration/PLAN.md) §5.

## 2. How the split is built, and why both halves are needed

**Two independent leakage checks, combined.** Either one alone is insufficient.

**Check 1 — temporal.** Cutoff `2023-06-01`. This is not a guess: it mirrors
BoltzGen's own training filter exactly, verified identical across every training config
in `src/boltzgen/` (`DateFilter(date="2023-06-01", ref="released")`). Structures on or
before it are presumptively in its training data.

**Check 2 — antigen sequence-identity clustering.** MMseqs2 `easy-cluster` at **40%
identity**, on antigen sequences only. Temporal separation alone is not enough: *the
same antigen reappears in later PDB entries* — a revised crystal form, a different Fab
against the same target, a new complex partner. A post-cutoff structure whose antigen
clusters with a pre-cutoff antigen is a near-duplicate wearing a later date, so it goes
to `excluded_ambiguous` rather than into the test set.

That check is why 2,244 structures are discarded — **66% of everything post-cutoff**
(3,386 structures), and 20% of SAbDab overall. That discard rate is the headline cost
of the guarantee, and it is worth paying: without check 2 the usable post-cutoff pool
would be 3,095 rather than 851 — **3.6× larger and 73% near-duplicates of training
antigens.** Most apparent post-cutoff data is a restatement of something BoltzGen has
already seen, which is exactly the trap a date filter alone walks into.

**The conservative default, stated explicitly.** `excluded_no_sequence` exists because
a structure whose antigen sequence failed to parse was *never checked for redundancy*.
It is not evidence of cleanliness, so those 291 are excluded. The rule throughout is
**unverified means excluded, never assumed clean.**

### Why 2,217 extraction failures produce only 291 exclusions

The two numbers look inconsistent and are not. Antigen sequences are only needed for
the *clustering* check, which only gates the post-cutoff pool:

| extraction failures | 2,217 |
|---|---|
| …that are `train_era` | 1,926 — kept anyway; training data needs no leak check |
| …that are post-cutoff | 291 — become `excluded_no_sequence` |
| …that reached `dev`/`test` | **0** |

So no structure with an unverified antigen is in any evaluation set. Of the failures,
1,257 are rows where SAbDab records the antigen chain as the literal string `NA`, 511
name a chain absent from the mmCIF, and 449 are multi-chain antigens (`A|B`) whose
chain set did not intersect any single `_entity_poly` entity.

### Known caveat: sequences come from `_entity_poly`

The parser reads the `_entity_poly` loop, which gives the **construct** sequence — it
can include signal peptides and expression tags that are not in the resolved
coordinates. This project has been bitten by exactly that elsewhere: in
`scoring_calibration` it produced antibody sequences with signal peptides attached
(`MPMGSLQPLATLYLLGMLVASVLA|EVQLVESGGG…`), and the fix was to rebuild from `_atom_site`.

**Here it is acceptable, and the reason is specific**: these sequences are used *only*
for 40%-identity clustering, never as model input. A ~20-residue leader on a
200-residue antigen cannot move a pairwise identity across the 40% threshold. If
anything it makes clustering marginally more permissive, i.e. it errs toward moving
structures into `excluded_ambiguous` — the conservative direction. Do **not** reuse
`antigen_clusters.tsv`'s underlying sequences as model input without re-extracting from
`_atom_site`.

## 3. The rule for which comparisons are legitimate

This is the rule other plans refer back to here.

| Claim | Requires |
|---|---|
| "model generalises" | evaluated on `dev` or `test` only |
| "model learned X" | trained on `train_era`, evaluated post-cutoff |
| anything tuned, then reported | tuned on `dev`, reported on `test_cluster_disjoint` (§4) |
| "BoltzGen scores N on real binders" | state the bucket; a `train_era` number measures memorisation and must say so |

**Never report a post-tuning number on the same file you tuned against.** And when a
figure mixes buckets, the mix has to be stated — the memorisation gap in
[EXPERIMENT_01](../../experiments/scoring_calibration/EXPERIMENT_01_iptm_calibration.md)
exists precisely because pre- and post-cutoff complexes score differently, so a blended
number is not interpretable.

## 4. The dev/test boundary is weaker than it looks — found 2026-10-31

Verified while writing this file, and it qualifies §3.

**`dev` and `test` are each cluster-disjoint from `train_era` by construction — that
much holds, and it is checked:**

| | structures | clusters | clusters shared with `train_era` |
|---|---|---|---|
| `train_era` | 8,072 | 1,409 | — |
| `dev` | 100 | 89 | **0** ✓ |
| `test` | 751 | 387 | **0** ✓ |

**But they are not cluster-disjoint from each other.** The dev/test split is a hash of
the *structure* id, applied after clustering, so clusters straddle the boundary:

| | |
|---|---|
| clusters in both `dev` and `test` | **59** of dev's 89 |
| `dev` structures in a shared cluster | **70 / 100** |
| `test` structures in a shared cluster | **197 / 751** (26%) |

So iterating on `dev` and then reporting on the full `test` leaks: 26% of `test` holds
antigens homologous to ones already tuned against. The `train_era` guarantee — the one
that matters most, since it is what makes any claim about BoltzGen meaningful — is
unaffected.

**Remedy, and why it is a new file rather than a fix.** Re-splitting by cluster would
renumber `dev` and `test` and silently invalidate every result already reported against
them, including EXPERIMENT_01's calibration and the epitope-prediction milestones. So
the overlap is recorded and `test_cluster_disjoint.txt` is emitted alongside: the
**554** test structures (328 clusters) sharing no cluster with `dev`. Verified to share
zero clusters with `dev` *and* zero with `train_era`.

- Reporting a number that followed **no** dev-based tuning → `test.txt` is fine.
- Reporting a number that followed dev-based tuning → use **`test_cluster_disjoint.txt`**.

Both are listed in `splits_summary.json` under `dev_test_cluster_overlap`, so the
distinction is machine-checkable rather than a convention someone has to remember.

## 5. Cross-reference: which other databases land where

Generated by `cross_reference_sources()`; `unknown` means the PDB id is not an
antigen-bound SAbDab entry at all, so it was never bucketed.

### AACDB — 3,674 unique PDB ids

| bucket | count |
|---|---|
| **`train_era`** | **3,628** (98.7%) |
| `unknown` | 23 |
| `excluded_ambiguous` | 18 |
| `test` | **4** |
| `excluded_no_sequence` | 1 |

**AACDB is almost entirely training-era.** This is the single most consequential number
here. It is what makes AACDB usable as a *training* label source — and it is why it
cannot be used for evaluation: only 4 of its 3,674 structures are in `test`, and none
in `dev`.

[`epitope_prediction/PLAN.md`](../../experiments/epitope_prediction/PLAN.md) §58 relies
on this directly: its labels are restricted to the 3,628 `train_era` structures, which
is what keeps `dev`/`test` genuinely held out for the epitope models. The 4 `test`
entries must be excluded from any training set drawn from AACDB.

### AB-Bind — 27 unique non-homology-model PDB ids

| bucket | count |
|---|---|
| `train_era` | 20 |
| `unknown` | 7 |

Entirely training-era. AB-Bind's ΔΔG mutants are therefore a **memorisation-era**
resource: fine for calibrating a score's dynamic range, never evidence of
generalisation. (`HM_`-prefixed rows are homology models with no PDB entry and are
dropped before tallying.)

## 6. Rebuilding

```bash
python3 databases/src/build_splits.py      # needs databases/sabdab/ populated
```

Deterministic: the temporal check is a date comparison, clustering is MMseqs2 at a
fixed threshold, and dev/test is `sha256(pdb_id) % 100 < 15`. No random seed, so the
same inputs give byte-identical outputs. MMseqs2 is fetched on demand by
`_common.ensure_mmseqs2()`.

`.mmseqs_work/` is deleted after the run; only `antigen_clusters.tsv` is kept from it,
because auditing an `excluded_ambiguous` decision by hand needs the cluster assignment
and nothing else.

**If `databases/sabdab/` is ever updated, everything here is stale.** The counts in
this file are tied to the 11,458-structure snapshot of 2026-08-16. Re-running will
move structures between buckets and change `dev`/`test` membership — which, per §4,
invalidates comparisons against previously reported numbers. Treat a rebuild as a
versioning event, not a refresh.
