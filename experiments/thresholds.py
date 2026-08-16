"""Pass/fail classification for BoltzGen per-design metrics.

Every campaign already writes a 197-column metrics CSV
(data/designs/<name>/run*/final_ranked_designs/all_designs_metrics.csv). This module
defines what "good" means for the handful of columns worth summarizing, and is explicit
about where each threshold comes from — README.md's own documented guidance, or a
heuristic invented here because README doesn't cover that column. Don't blur the two:
a heuristic threshold can silently contradict BoltzGen's own ranking (see
liability_num_violations below) and callers need to know which kind they're looking at.
"""

from dataclasses import dataclass
from typing import Callable, Literal

Verdict = Literal["pass", "borderline", "fail", "unknown"]
Source = Literal["readme", "heuristic"]


@dataclass(frozen=True)
class Threshold:
    metric: str
    source: Source
    doc: str
    classify: Callable[[float], Verdict]


def _iptm_classify(v: float) -> Verdict:
    # README.md: >0.8 confident interface; <0.5 don't trust the binding pose;
    # 0.3-0.6 "genuinely ambiguous, shouldn't be over-interpreted from a single seed".
    if v > 0.8:
        return "pass"
    if v < 0.5:
        return "fail"
    return "borderline"


def _filter_rmsd_classify(v: float) -> Verdict:
    # README.md: self-consistency RMSD between the design and its independent refold.
    # <2-3A good; this project's own low-ranked results have gone up to ~18A.
    if v <= 3.0:
        return "pass"
    if v <= 6.0:
        return "borderline"
    return "fail"


def _complex_plddt_classify(v: float) -> Verdict:
    # HEURISTIC, not README-sourced: README's 85/70 pLDDT bands are documented for
    # predict_structure.py's standalone single-chain fold (0-100 scale), a different
    # quantity than this CSV's complex_plddt, which is on a 0-1 fractional scale
    # (verified directly: observed range ~0.65-0.80 across all three campaigns on
    # disk). Rescaled 85/70 -> 0.85/0.70 as a plausible-but-unverified guess pending
    # real calibration data.
    if v >= 0.85:
        return "pass"
    if v >= 0.70:
        return "borderline"
    return "fail"


def _liability_violations_classify(v: float) -> Verdict:
    # HEURISTIC, not README-sourced: README has no numeric cutoff anywhere for this
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
        "design_to_target_iptm", "readme",
        "README.md: >0.8 confident interface, <0.5 don't trust, 0.3-0.6 ambiguous.",
        _iptm_classify,
    ),
    "filter_rmsd": Threshold(
        "filter_rmsd", "readme",
        "README.md: self-consistency RMSD, <2-3A good, up to ~18A seen in low-ranked "
        "results in this project.",
        _filter_rmsd_classify,
    ),
    "complex_plddt": Threshold(
        "complex_plddt", "heuristic",
        "No README equivalence for this column exists; borrowed from "
        "predict_structure.py's unrelated single-chain pLDDT bands (85/70) as a guess.",
        _complex_plddt_classify,
    ),
    "liability_num_violations": Threshold(
        "liability_num_violations", "heuristic",
        "No README cutoff exists. Loose by design: this project's own rank-1 design "
        "(hel_18) has 16 violations and is not considered a failure.",
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
    self-consistency check, developability, and BoltzGen's own combined ranking signal.
    """
    return [
        "id",
        "final_rank",
        "design_to_target_iptm",
        "min_design_to_target_pae",
        "design_ptm",
        "filter_rmsd",
        "complex_plddt",
        "liability_score",
        "liability_num_violations",
        "plip_hbonds_refolded",
        "plip_saltbridge_refolded",
        "num_filters_passed",
        "pass_filters",
        "quality_score",
    ]
