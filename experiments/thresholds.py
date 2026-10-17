"""Pass/fail classification for BoltzGen per-design metrics.

Every campaign already writes a 197-column metrics CSV
(data/designs/<name>/run*/final_ranked_designs/all_designs_metrics.csv). This module
defines what "good" means for the handful of columns worth summarizing, and is explicit
about where each threshold comes from. Don't blur the provenance kinds: a heuristic
threshold can silently contradict BoltzGen's own ranking (see
liability_num_violations below) and callers need to know which kind they're looking at.

Provenance kinds, in decreasing order of how much weight to put on them:

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
"""

from dataclasses import dataclass
from typing import Callable, Literal

Verdict = Literal["pass", "borderline", "fail", "unknown"]
Source = Literal["boltzgen", "literature", "readme", "heuristic"]


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
    # LITERATURE: Bennett et al., "Atomically accurate de novo design of antibodies
    # with RFdiffusion", Nature 2025. ipTM is predictive of real binding success at
    # AUC 0.86, and they propose ipTM > 0.6 as a filter (noting only 9% of their
    # ordered VHH designs clear it). The <0.5 floor is retained from README.md's
    # "below ~0.5 means don't trust the binding pose at all"; the 0.5-0.6 band between
    # the two is the honest "neither source endorses this" gap.
    if v > 0.6:
        return "pass"
    if v < 0.5:
        return "fail"
    return "borderline"


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
    # HEURISTIC, unchanged by the 2026-10-17 recalibration: no README equivalence and
    # no BoltzGen filter uses this column, so nothing better-grounded was available.
    # README's 85/70 pLDDT bands are documented for predict_structure.py's standalone
    # single-chain fold (0-100 scale), a different quantity than this CSV's
    # complex_plddt, which is on a 0-1 fractional scale (verified directly: observed
    # range ~0.65-0.80 across the campaigns on disk). Rescaled 85/70 -> 0.85/0.70 as a
    # plausible-but-unverified guess pending real calibration data.
    if v >= 0.85:
        return "pass"
    if v >= 0.70:
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
        "design_to_target_iptm", "literature",
        "Bennett et al. Nature 2025: ipTM > 0.6 as a de novo design filter, AUC 0.86 "
        "for predicting binding success. Was 0.8 (README cofolding guidance) until "
        "2026-10-17 — that is the wrong task's threshold and scored 0% everywhere.",
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
        "complex_plddt", "heuristic",
        "No README equivalence and no BoltzGen filter for this column; borrowed from "
        "predict_structure.py's unrelated single-chain pLDDT bands (85/70) as a guess.",
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
