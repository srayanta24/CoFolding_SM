#!/usr/bin/env python3
"""Aggregate per-design metrics across every antibody design campaign on disk.

Each BoltzGen campaign already writes its own rich per-design CSV
(data/designs/<name>/run*/final_ranked_designs/all_designs_metrics.csv). Nothing in the
project layer combines these across campaigns or defines what counts as a "good"
design — that's what this module does, on top of the CSVs BoltzGen already produced
(it does not recompute or re-derive any confidence metric).

Usage:
    python3 experiments/aggregate_metrics.py
"""

import csv
import sys
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _common import REPO_ROOT  # noqa: E402

import thresholds  # noqa: E402

DESIGNS_DIR = REPO_ROOT / "data" / "designs"

# Hardcoded, not parsed from campaign YAML/Hydra configs: this project has no YAML
# dependency in the stdlib-only script layer, and there are only a handful of
# campaigns to track by hand. structure_source matters because it's a real confound —
# ctrop2_antibody's target structure was Boltz-2-predicted, not experimentally real,
# which is a meaningfully different starting point than trop2_antibody's fetched 7e5n.
CAMPAIGN_META: dict[str, dict] = {
    "hel_antibody": {"target": "HEL (hen egg lysozyme)", "structure_source": "real_pdb (1dpx, apo)"},
    "ctrop2_antibody": {"target": "TROP2", "structure_source": "predicted (boltz2)"},
    "trop2_antibody": {"target": "TROP2", "structure_source": "real_pdb (7e5n)"},
}

NUMERIC_METRICS = [
    "design_to_target_iptm",
    "min_design_to_target_pae",
    "design_ptm",
    "filter_rmsd",
    "complex_plddt",
    "liability_score",
    "liability_num_violations",
    "plip_hbonds_refolded",
    "plip_saltbridge_refolded",
    "quality_score",
]


def find_metrics_csvs(root: Path = DESIGNS_DIR) -> list[Path]:
    return sorted(root.glob("*/run*/final_ranked_designs/all_designs_metrics.csv"))


def campaign_name_from_path(csv_path: Path) -> str:
    # .../<campaign>/run1/final_ranked_designs/all_designs_metrics.csv
    return csv_path.parents[2].name


def _parse_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_campaign(csv_path: Path) -> list[dict]:
    columns = set(thresholds.core_metric_columns())
    rows = []
    with open(csv_path, newline="") as f:
        for raw_row in csv.DictReader(f):
            row = {k: v for k, v in raw_row.items() if k in columns}
            for metric in NUMERIC_METRICS:
                if metric in row:
                    row[metric] = _parse_float(row[metric])
            rows.append(row)
    return rows


def summarize(rows: list[dict], metric: str) -> dict:
    values = [r[metric] for r in rows if r.get(metric) is not None]
    if not values:
        return {"n": 0, "n_total": len(rows)}
    verdicts = [thresholds.classify(metric, v) for v in values]
    return {
        "n": len(values),
        "n_total": len(rows),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "pass_rate": verdicts.count("pass") / len(verdicts),
        "fail_rate": verdicts.count("fail") / len(verdicts),
    }


def per_campaign_summary(root: Path = DESIGNS_DIR) -> dict[str, dict]:
    summary = {}
    for csv_path in find_metrics_csvs(root):
        name = campaign_name_from_path(csv_path)
        rows = load_campaign(csv_path)
        summary[name] = {
            "csv_path": str(csv_path.relative_to(REPO_ROOT)),
            "meta": CAMPAIGN_META.get(name, {"target": "unknown", "structure_source": "unknown"}),
            "num_designs": len(rows),
            "num_passed_filters": sum(1 for r in rows if r.get("pass_filters") == "True"),
            "metrics": {m: summarize(rows, m) for m in NUMERIC_METRICS},
        }
    return summary


def cross_campaign_summary(per_campaign: dict[str, dict]) -> dict[str, dict]:
    cross = {}
    for metric in NUMERIC_METRICS:
        campaign_means = [
            c["metrics"][metric]["mean"]
            for c in per_campaign.values()
            if c["metrics"][metric]["n"] > 0
        ]
        pass_rates = [
            c["metrics"][metric]["pass_rate"]
            for c in per_campaign.values()
            if c["metrics"][metric]["n"] > 0
        ]
        if not campaign_means:
            continue
        cross[metric] = {
            "num_campaigns": len(campaign_means),
            "mean_of_campaign_means": mean(campaign_means),
            "mean_pass_rate": mean(pass_rates),
        }
    return cross


def main() -> None:
    per_campaign = per_campaign_summary()
    cross = cross_campaign_summary(per_campaign)
    import json
    print(json.dumps({"per_campaign": per_campaign, "cross_campaign": cross}, indent=2))


if __name__ == "__main__":
    main()
