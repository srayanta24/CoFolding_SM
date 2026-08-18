#!/usr/bin/env python3
"""Evaluation for the GBT baseline and both GNN ensembles: ROC AUC, precision/recall,
calibration diagnostics, and a confidence-sanity check -- do residues where the
ensemble disagrees more (lower confidence) actually have lower accuracy? If not, the
confidence signal isn't doing its job and that needs to be surfaced, not hidden
(PLAN.md's explicit ask). Always evaluated on data/interface_labels.py's from-scratch
labels (never AACDB's), per the coverage-gap fix.

Usage:
    python3 experiments/epitope_prediction/eval/classifier_metrics.py --split dev
    python3 experiments/epitope_prediction/eval/classifier_metrics.py --split test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
from dataset import load_cached  # noqa: E402
from gnn import ENSEMBLE_SIZE, ensemble_predict, load_ensemble  # noqa: E402


def eval_baseline(clf, entries: list[dict]) -> dict:
    X = np.concatenate([e["node_features"].numpy() for e in entries], axis=0)
    y = np.concatenate([e["labels"].numpy() for e in entries], axis=0)
    probs = clf.predict_proba(X)[:, 1]
    return {"y": y, "propensity": probs, "confidence": None}


def eval_ensemble(feature_set: str, entries: list[dict], device: str) -> dict:
    models = load_ensemble(feature_set, device)
    all_y, all_prop, all_conf = [], [], []
    for e in entries:
        prop, conf = ensemble_predict(models, e, device)
        all_y.append(e["labels"].numpy())
        all_prop.append(prop.numpy())
        all_conf.append(conf.numpy())
    return {
        "y": np.concatenate(all_y),
        "propensity": np.concatenate(all_prop),
        "confidence": np.concatenate(all_conf),
    }


def calibration_curve(y: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> list[tuple[float, float, int]]:
    """(mean_predicted, observed_fraction, n) per bin -- a lightweight reliability
    curve without needing sklearn's calibration_curve (avoids an edge case with empty
    bins raising)."""
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        out.append((probs[mask].mean(), y[mask].mean(), int(mask.sum())))
    return out


def confidence_sanity_check(y: np.ndarray, propensity: np.ndarray, confidence: np.ndarray, n_bins: int = 5) -> None:
    """Bins residues by confidence, reports prediction error (|y - propensity|) per
    bin. A real confidence signal should show LOWER error in HIGHER-confidence bins --
    if it doesn't, say so plainly rather than silently reporting AUC alone."""
    error = np.abs(y - propensity)
    order = np.argsort(confidence)
    bin_edges = np.array_split(order, n_bins)
    print("  confidence-sanity check (should show decreasing error at higher confidence):")
    prev_mean_error = None
    monotonic = True
    for i, idx in enumerate(bin_edges):
        mean_conf = confidence[idx].mean()
        mean_err = error[idx].mean()
        print(f"    bin {i+1}/{n_bins}: mean_confidence={mean_conf:.3f}  mean_|error|={mean_err:.3f}  n={len(idx)}")
        if prev_mean_error is not None and mean_err > prev_mean_error + 1e-9:
            monotonic = False
        prev_mean_error = mean_err
    if not monotonic:
        print("    WARNING: error does not decrease monotonically with confidence -- "
              "the confidence signal may not be reliable, don't trust it blindly for "
              "residue-count selection without investigating further.")


def report(name: str, result: dict) -> None:
    y, propensity, confidence = result["y"], result["propensity"], result["confidence"]
    auc = roc_auc_score(y, propensity)
    brier = brier_score_loss(y, propensity)
    precision, recall, thresholds = precision_recall_curve(y, propensity)

    print(f"\n=== {name} ===")
    print(f"  AUC: {auc:.4f}  (SEMA-2.0 reference: 0.76)")
    print(f"  Brier score: {brier:.4f} (lower is better-calibrated)")
    print(f"  n={len(y)}, positive rate={y.mean():.1%}")

    print("  calibration (mean predicted vs. observed fraction, 10 bins):")
    for mean_pred, observed, n in calibration_curve(y, propensity):
        print(f"    predicted={mean_pred:.3f}  observed={observed:.3f}  n={n}")

    # Precision at a couple of fixed recall points, for a threshold-independent-ish read.
    for target_recall in (0.3, 0.5):
        idx = np.argmin(np.abs(recall - target_recall))
        print(f"  at recall~{recall[idx]:.2f}: precision={precision[idx]:.3f} (threshold={thresholds[min(idx, len(thresholds)-1)]:.3f})")

    if confidence is not None:
        confidence_sanity_check(y, propensity, confidence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    entries = load_cached(args.split)
    print(f"[classifier_metrics] evaluating on {args.split}.txt ({len(entries)} structures)", file=sys.stderr)

    import pickle
    baseline_path = Path(__file__).resolve().parent.parent / "model" / "checkpoints" / "baseline.pkl"
    if baseline_path.exists():
        with open(baseline_path, "rb") as f:
            clf = pickle.load(f)
        report("GBT baseline", eval_baseline(clf, entries))

    checkpoint_dir = Path(__file__).resolve().parent.parent / "model" / "checkpoints"
    if (checkpoint_dir / f"model_A_seed{ENSEMBLE_SIZE - 1}.pt").exists():
        report("Model A (geometric only)", eval_ensemble("A", entries, device))
    if (checkpoint_dir / f"model_B_seed{ENSEMBLE_SIZE - 1}.pt").exists():
        report("Model B (geometric + ESM2)", eval_ensemble("B", entries, device))


if __name__ == "__main__":
    main()
