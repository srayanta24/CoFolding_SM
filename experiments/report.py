#!/usr/bin/env python3
"""Render a cross-campaign antibody design benchmark report.

Combines experiments/aggregate_metrics.py's per-campaign/cross-campaign summaries
(and, once available, experiments/score_reference.py's known-binder baseline) into one
markdown report plus a JSON sidecar. Per UI_DESIGN.md's stated contract for the
not-yet-built dashboard ("the UI does not recompute metrics, it only renders what the
eval harness already produced"), the JSON is the thing anything downstream should
consume — the markdown is for humans.

Usage:
    python3 experiments/report.py
    python3 experiments/report.py --with-reference hel
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _common import REPO_ROOT  # noqa: E402

import aggregate_metrics  # noqa: E402
import thresholds  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def _fmt(v, digits=3):
    return "n/a" if v is None else f"{v:.{digits}f}"


def render_markdown(
    per_campaign: dict, cross_campaign: dict, reference_baselines: dict | None = None
) -> str:
    lines = [
        f"# Antibody design benchmark report — {date.today().isoformat()}",
        "",
        "Aggregates the per-design metrics BoltzGen already computes for each campaign "
        "(`data/designs/*/run*/final_ranked_designs/all_designs_metrics.csv`) — no "
        "confidence metric here is recomputed, only summarized. Thresholds and their "
        "provenance (README-sourced vs. heuristic) are documented in "
        "`experiments/thresholds.py`.",
        "",
        "## Per-campaign summary",
        "",
    ]

    for name, c in sorted(per_campaign.items()):
        meta = c["meta"]
        lines.append(f"### `{name}` — {meta['target']} ({meta['structure_source']})")
        lines.append("")
        lines.append(
            f"{c['num_designs']} designs generated, "
            f"{c['num_passed_filters']} passed BoltzGen's own filters "
            f"({c['num_passed_filters'] / c['num_designs']:.1%})."
        )
        lines.append("")
        lines.append("| metric | mean | median | min | max | pass rate | source |")
        lines.append("|---|---|---|---|---|---|---|")
        for metric in ["design_to_target_iptm", "filter_rmsd", "complex_plddt", "liability_num_violations"]:
            s = c["metrics"][metric]
            th = thresholds.THRESHOLDS[metric]
            if s["n"] == 0:
                lines.append(f"| `{metric}` | n/a | n/a | n/a | n/a | n/a | {th.source} |")
                continue
            lines.append(
                f"| `{metric}` | {_fmt(s['mean'])} | {_fmt(s['median'])} | "
                f"{_fmt(s['min'])} | {_fmt(s['max'])} | {s['pass_rate']:.1%} | {th.source} |"
            )
        lines.append("")

    lines.append("## Cross-campaign summary")
    lines.append("")
    lines.append(f"Across {len(per_campaign)} campaigns:")
    lines.append("")
    lines.append("| metric | mean of campaign means | mean pass rate |")
    lines.append("|---|---|---|")
    for metric, s in cross_campaign.items():
        if metric not in thresholds.THRESHOLDS:
            continue
        lines.append(f"| `{metric}` | {_fmt(s['mean_of_campaign_means'])} | {s['mean_pass_rate']:.1%} |")
    lines.append("")

    if reference_baselines:
        lines.append("## Known-binder reference baselines")
        lines.append("")
        lines.append(
            "Real, experimentally known antibody-antigen complexes scored through the "
            "same cofolding pipeline (`scripts/run_design.py`) used to externally "
            "validate designs — calibrates what a genuine true positive scores on this "
            "pipeline, since BoltzGen's own metrics are self-reported and generic "
            "literature ipTM thresholds may not transfer directly. This is NOT a "
            "structural/epitope-overlap check (see experiments/README.md for that "
            "caveat) — a design can score well here while binding a different surface "
            "than the reference complex."
        )
        lines.append("")
        for name, baseline in sorted(reference_baselines.items()):
            lines.append(f"### `{name}`")
            lines.append("")
            for backend, metrics in baseline.items():
                lines.append(f"- **{backend}**: {metrics}")
            lines.append("")

    lines.append(
        "---\n\n_Caveat (README.md): these are computational predictions scored by "
        "each model's own confidence, not validated binders. Campaign scale here "
        "(50-1000 designs) is far below BoltzGen's own guidance for realistic "
        "hit-finding (10,000-60,000) — treat pass rates as demo-scale, not "
        "representative campaign hit rates._"
    )
    return "\n".join(lines)


def write_report(
    reference_baselines: dict | None = None, out_dir: Path = REPORTS_DIR
) -> tuple[Path, Path]:
    per_campaign = aggregate_metrics.per_campaign_summary()
    cross_campaign = aggregate_metrics.cross_campaign_summary(per_campaign)

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    md_path = out_dir / f"{stamp}_benchmark.md"
    json_path = out_dir / f"{stamp}_benchmark.json"

    md_path.write_text(render_markdown(per_campaign, cross_campaign, reference_baselines))
    json_path.write_text(
        json.dumps(
            {
                "date": stamp,
                "per_campaign": per_campaign,
                "cross_campaign": cross_campaign,
                "reference_baselines": reference_baselines or {},
            },
            indent=2,
        )
    )
    return md_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--with-reference", action="append", default=[],
        help="Reference complex name(s) from reference_targets.py to score and include "
             "as a baseline (requires running scripts/run_design.py — slow). Can repeat.",
    )
    args = parser.parse_args()

    reference_baselines = {}
    if args.with_reference:
        import score_reference

        for name in args.with_reference:
            out_dir = score_reference.score_reference(name)
            reference_baselines[name] = score_reference.load_reference_confidence(out_dir)

    md_path, json_path = write_report(reference_baselines or None)
    print(f"[report] wrote {md_path}")
    print(f"[report] wrote {json_path}")


if __name__ == "__main__":
    main()
