#!/usr/bin/env python3
"""Gradient-boosted-trees sanity floor: hand-engineered geometric features only (no
graph, no ESM2), flattened across all residues in all structures. A real number before
either GNN is trained, not skipped -- per PLAN.md sec 4.

Usage:
    python3 experiments/epitope_prediction/model/baseline.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_cached  # noqa: E402

CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "baseline.pkl"


def flatten(entries: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.concatenate([e["node_features"].numpy() for e in entries], axis=0)
    y = np.concatenate([e["labels"].numpy() for e in entries], axis=0)
    return X, y


def train_baseline(train_entries: list[dict]) -> HistGradientBoostingClassifier:
    X, y = flatten(train_entries)
    print(f"[baseline] training on {X.shape[0]} residues, {y.sum():.0f} positive ({y.mean():.1%})", file=sys.stderr)
    clf = HistGradientBoostingClassifier(max_iter=200, max_depth=6, random_state=0)
    clf.fit(X, y)
    return clf


def evaluate(clf: HistGradientBoostingClassifier, entries: list[dict], name: str) -> float:
    X, y = flatten(entries)
    probs = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)
    print(f"[baseline] {name} AUC: {auc:.4f} (n={len(y)}, positive={y.mean():.1%})")
    return auc


def main() -> None:
    train_entries = load_cached("train")
    dev_entries = load_cached("dev")

    clf = train_baseline(train_entries)
    evaluate(clf, dev_entries, "dev")

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"[baseline] saved to {CHECKPOINT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
