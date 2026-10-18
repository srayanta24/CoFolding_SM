"""Pass/fail classification for BoltzGen per-design metrics.

Every campaign already writes a 197-column metrics CSV
(data/designs/<name>/run*/final_ranked_designs/all_designs_metrics.csv). This module
defines what "good" means for the handful of columns worth summarizing, and is explicit
about where each threshold comes from. Don't blur the provenance kinds: a heuristic
threshold can silently contradict BoltzGen's own ranking (see
liability_num_violations below) and callers need to know which kind they're looking at.

Provenance kinds, in decreasing order of how much weight to put on them:

``measured``
    Measured on this pipeline, on this hardware, against known-real binders —
    see ``MEASURED_REFERENCE`` below and
    ``experiments/scoring_calibration/EXPERIMENT_01_iptm_calibration.md``. The
    strongest kind available, because it involves no transfer assumption from
    another model, another machine, or another task.
``boltzgen``
    Read directly out of BoltzGen's own running code (file:line cited in the
    threshold's ``doc``). This is what the tool itself does, not an interpretation
    of what it ought to do.
``literature``
    A published, empirically validated operating point from the antibody-design
    literature, with the paper and the validation cited.
``readme``
    This project's own README.md guidance. Useful, but written for the *cofolding*
    task (Skill A) and not automatically transferable to de novo design scoring —
    which is exactly the mistake the 2026-10-17 recalibration below corrected.
``heuristic``
    Invented here because nothing above covers that column. Treat as a placeholder.

Recalibration, 2026-10-17 (IMPROVE_DESIGN.md sec 4, step 1)
-----------------------------------------------------------
Two thresholds moved, because the originals were borrowed from the wrong task:

- ``design_to_target_iptm`` pass mark 0.8 -> 0.6. The 0.8 came from README.md's
  cofolding guidance ("above ~0.8 suggests a confident, likely-correct interface"),
  which describes evaluating a *known* complex. For *de novo design* the empirically
  validated operating point is ipTM > 0.6 (Bennett et al., Nature 2025 -- AUC 0.86
  for predicting real binding success, with only 9% of their ordered VHH designs
  clearing it). At 0.8 every campaign on disk scores a 0% pass rate, which told us
  nothing; at 0.6 the metric discriminates.
- ``filter_rmsd`` pass mark 3.0 -> 2.5, which is simply BoltzGen's own default
  (``refolding_rmsd_threshold: float = 2.5``, filter.py:162). Our 3.0 was *looser*
  than the tool's own filter, so this module was reporting designs as passing
  self-consistency that BoltzGen itself had already rejected.

Worth stating plainly, since it bears on how any of these numbers should be read: a
campaign's raw pass rate is the pass rate of an *unfiltered* design pool. BoltzGen's
method is extreme oversampling plus hard filtering (its own protocol generates tens of
thousands and advances ~15), so low raw pass rates at this project's 50-3,000 design
scale are expected and are not by themselves evidence the pipeline underperforms. See
IMPROVE_DESIGN.md sec 2.

Recalibration, 2026-10-18 (scoring_calibration PLAN.md milestone 5)
-------------------------------------------------------------------
EXPERIMENT_01 cofolded 32 leak-free, crystallographically confirmed antibody-antigen
binders through this exact pipeline. ``design_to_target_iptm`` and ``complex_plddt``
are now anchored to that measured distribution instead of to imported or invented
numbers, using one consistent scheme for both:

    pass        >= the MEDIAN real binder   ("as good as a typical real binder")
    borderline  >= the 25th percentile      ("inside the range real binders occupy")
    fail        <  the 25th percentile      ("worse than 3 in 4 real binders")

This makes every verdict interpretable as a statement about real antibodies rather
than about an arbitrary constant. It also moves ipTM's pass mark UP, from 0.6 to
0.699 — Bennett et al.'s 0.6 is the point below which they stopped testing, not the
point at which something resembles a real binder, and it now lands inside the
borderline band. Campaign pass rates will fall as a result; that is the measurement
being honest, not a regression.

**The caveat that matters most, from the same experiment.** ipTM does NOT track
binding affinity: Spearman vs log10(Kd) = +0.134 across 100 real complexes spanning
nine orders of magnitude, with the trend mildly *backwards*. Clearing the ipTM bar
means "a credible interface can be modelled here," NOT "this binds tightly." Do not
read a high ipTM as predicted potency, and do not rank candidates by it expecting the
ordering to reflect affinity. See IMPROVE_DESIGN.md sec 4's filtration subsection for
what to use instead.
"""

from dataclasses import dataclass
from typing import Callable, Literal

Verdict = Literal["pass", "borderline", "fail", "unknown"]
Source = Literal["measured", "boltzgen", "literature", "readme", "heuristic"]

# Percentiles of each metric over the 32 leak-free, crystallographically confirmed
# antibody-antigen binders of EXPERIMENT_01's `clean_positive` set, cofolded through
# this pipeline (Boltz-2, no MSA) on this hardware. Exposed so reports and downstream
# analysis can phrase a design's score as "the Nth percentile of real binders" rather
# than as a bare number. n=32 (dev only, per split discipline): the medians are usable,
# the tails are thin. Regenerate with experiments/scoring_calibration/analyze.py.
MEASURED_REFERENCE: dict[str, dict[str, float]] = {
    "design_to_target_iptm": {
        "p05": 0.443, "p10": 0.487, "p25": 0.578, "p50": 0.699,
        "p75": 0.798, "p90": 0.885, "min": 0.368, "max": 0.967, "n": 32,
    },
    "complex_plddt": {
        "p05": 0.623, "p10": 0.679, "p25": 0.751, "p50": 0.848,
        "p75": 0.902, "p90": 0.925, "min": 0.581, "max": 0.943, "n": 32,
    },
}


@dataclass(frozen=True)
class Threshold:
    metric: str
    source: Source
    doc: str
    classify: Callable[[float], Verdict]


# What BoltzGen's `pass_filters` column actually means, read off
# task/filter/filter.py:233-275 for the antibody-anything/nanobody-anything protocol
# (which sets filter_cysteine=true and leaves filter_designfolding / filter_bindingsite
# / filter_target_aligned off). Recorded here because "98/300 passed filters" is
# meaningless without knowing which filters those were.
#
# The load-bearing absence: `design_to_target_iptm` is NOT among them. BoltzGen hard-
# filters on structural validity and self-consistency only, and treats ipTM purely as a
# *ranking* metric (see BOLTZGEN_RANKING_WEIGHTS). The literature's suggestion to use
# ipTM > 0.6 as an additional hard filter is therefore a real, testable change to the
# funnel, not something BoltzGen already does -- that's step 2 in IMPROVE_DESIGN.md.
BOLTZGEN_HARD_FILTERS: dict[str, str] = {
    "has_x": "<= 0 — no unknown/X residues in the designed sequence.",
    "filter_rmsd": "<= 2.5 A — all-atom RMSD, design vs. its independent refold.",
    "filter_rmsd_design": "<= 2.5 A — same check against the design-stage structure.",
    "CYS_fraction": "<= 0 — no designed cysteines (antibody/nanobody/peptide only).",
}

# BoltzGen's default ranking metrics and weights (paper Algorithm 2, 'Top-k binder
# design selection by worst weighted metric rank'). The score for a design is
# max_j(rank_ij / w_j) -- i.e. its *worst* weighted rank across all metrics, so this
# prioritizes designs with no bad axis rather than designs with one great axis.
# All of it is overridable at the CLI via --metrics_override with no model change,
# which is what makes stage-5 experiments free. Lower rank is better.
BOLTZGEN_RANKING_WEIGHTS: dict[str, int] = {
    "design_iptm": 1,
    "design_ptm": 2,
    "neg_min_design_to_target_pae": 1,
    "plip_hbonds_refolded": 2,
    "plip_saltbridge_refolded": 2,
    "delta_sasa_refolded": 2,
}


def _iptm_classify(v: float) -> Verdict:
    # MEASURED (EXPERIMENT_01): anchored to the real-binder distribution rather than
    # to an imported constant. pass = at or above the median real binder (0.699);
    # borderline = inside their interquartile range (>= p25, 0.578); fail = below
    # what 3 in 4 real binders achieve.
    #
    # For orientation against the previous marks: 0.8 (the pre-step-1 value, from
    # README's cofolding guidance) admits only 22% of real binders; 0.6 (Bennett
    # et al.'s AlphaFold3 operating point, used between step 1 and now) admits 72%
    # and sits inside the borderline band here.
    #
    # Reminder, because it is easy to forget once this returns "pass": a high ipTM
    # does NOT imply tight binding. See the module docstring.
    if v >= MEASURED_REFERENCE["design_to_target_iptm"]["p50"]:
        return "pass"
    if v >= MEASURED_REFERENCE["design_to_target_iptm"]["p25"]:
        return "borderline"
    return "fail"


def _filter_rmsd_classify(v: float) -> Verdict:
    # BOLTZGEN: refolding_rmsd_threshold defaults to 2.5 in
    # src/boltzgen/src/boltzgen/task/filter/filter.py:162, and is applied to
    # filter_rmsd and filter_rmsd_design as a hard filter (filter.py:235-244). Using
    # anything looser here means reporting designs as self-consistent that BoltzGen's
    # own pipeline already dropped.
    #
    # Note which RMSD this is: filtering.yaml sets from_inverse_folded=false, so
    # filter_rmsd is the ALL-ATOM rmsd, not the backbone bb_rmsd (filter.py:365-369).
    #
    # The 6.0 borderline band is HEURISTIC -- BoltzGen has no second tier, it is a
    # straight pass/fail at 2.5. Kept only so the report can distinguish "just missed"
    # from "wildly inconsistent" (this project has seen values up to ~22 A).
    if v <= 2.5:
        return "pass"
    if v <= 6.0:
        return "borderline"
    return "fail"


def _complex_plddt_classify(v: float) -> Verdict:
    # MEASURED (EXPERIMENT_01), upgraded from heuristic. Same scheme as ipTM: pass at
    # the median real binder (0.848), borderline at their p25 (0.751).
    #
    # Worth recording rather than quietly overwriting: the old heuristic was a guess
    # that rescaled README's single-chain pLDDT bands 85/70 -> 0.85/0.70, and it turned
    # out to be close. 0.85 admits 50% of real binders (the median is 0.848) and 0.70
    # admits 88%. The guess was sound; it is now simply measured instead of assumed.
    if v >= MEASURED_REFERENCE["complex_plddt"]["p50"]:
        return "pass"
    if v >= MEASURED_REFERENCE["complex_plddt"]["p25"]:
        return "borderline"
    return "fail"


def _liability_violations_classify(v: float) -> Verdict:
    # HEURISTIC, unchanged: no README cutoff and no BoltzGen filter exists for this
    # column. The worked example's own rank-1 winning design (hel_18) carries 16
    # violations without being flagged as bad, so this threshold is intentionally loose
    # — it exists to catch outliers, not to second-guess BoltzGen's own ranking.
    if v <= 20:
        return "pass"
    if v <= 35:
        return "borderline"
    return "fail"


THRESHOLDS: dict[str, Threshold] = {
    "design_to_target_iptm": Threshold(
        "design_to_target_iptm", "measured",
        "EXPERIMENT_01: pass at the median of 32 real leak-free binders cofolded "
        "through this pipeline (0.699), borderline at their p25 (0.578). Was 0.8 "
        "(README cofolding guidance, admits 22% of real binders) until 2026-10-17, "
        "then 0.6 (Bennett et al., admits 72%) until 2026-10-18. CAVEAT: ipTM does "
        "not track affinity (Spearman vs log10(Kd) = +0.134, wrong sign) — passing "
        "means a credible interface can be modelled, not that it binds tightly.",
        _iptm_classify,
    ),
    "filter_rmsd": Threshold(
        "filter_rmsd", "boltzgen",
        "BoltzGen's own default: refolding_rmsd_threshold=2.5 (filter.py:162), applied "
        "as a hard filter. All-atom RMSD, since filtering.yaml sets "
        "from_inverse_folded=false. Was 3.0 until 2026-10-17 — looser than the tool's "
        "own filter. The 6.0 borderline band remains heuristic.",
        _filter_rmsd_classify,
    ),
    "complex_plddt": Threshold(
        "complex_plddt", "measured",
        "EXPERIMENT_01: pass at the median of 32 real leak-free binders (0.848), "
        "borderline at their p25 (0.751). Previously a heuristic rescaling of "
        "README's single-chain pLDDT bands to 0.85/0.70 — which measurement shows "
        "was a good guess (0.85 admits 50% of real binders, 0.70 admits 88%).",
        _complex_plddt_classify,
    ),
    "liability_num_violations": Threshold(
        "liability_num_violations", "heuristic",
        "No README cutoff and no BoltzGen filter exists. Loose by design: this "
        "project's own rank-1 design (hel_18) has 16 violations and is not a failure.",
        _liability_violations_classify,
    ),
}


def classify(metric: str, value: float | None) -> Verdict:
    if value is None:
        return "unknown"
    threshold = THRESHOLDS.get(metric)
    if threshold is None:
        return "unknown"
    return threshold.classify(value)


def core_metric_columns() -> list[str]:
    """The subset of the 197 CSV columns worth surfacing in a benchmark report.

    Not "all columns" — this is a deliberate allowlist: interface confidence, the
    self-consistency checks, developability, and BoltzGen's own combined ranking signal.

    Added 2026-10-17: `delta_sasa_refolded` and `filter_rmsd_design`. Both are things
    BoltzGen itself acts on — the former carries weight 2 in its default ranking
    (BOLTZGEN_RANKING_WEIGHTS), the latter is one of its four hard filters
    (BOLTZGEN_HARD_FILTERS) — and neither was being reported at all, which made our
    summaries blind to two of the signals actually driving the tool's own selection.
    """
    return [
        "id",
        "final_rank",
        "design_to_target_iptm",
        "min_design_to_target_pae",
        "design_ptm",
        "filter_rmsd",
        "filter_rmsd_design",
        "delta_sasa_refolded",
        "complex_plddt",
        "liability_score",
        "liability_num_violations",
        "plip_hbonds_refolded",
        "plip_saltbridge_refolded",
        "num_filters_passed",
        "pass_filters",
        "quality_score",
    ]
