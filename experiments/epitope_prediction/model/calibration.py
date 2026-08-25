#!/usr/bin/env python3
"""Post-hoc calibration for the ensemble's propensity output -- absent entirely until
now (no Platt/isotonic/temperature scaling existed anywhere in this pipeline). Isotonic
regression via scikit-learn (already installed for the GBT baseline, zero new deps),
fit on the same deterministic internal-val slice gnn.py's train_ensemble() already
carves out for early stopping.

Mild double-dipping (same slice used for both early stopping and calibration fitting)
is a standard, accepted pattern here -- it never touches databases/splits/dev.txt or
test.txt, so it doesn't threaten those numbers' validity as a held-out yardstick.

Isotonic (monotonic) calibration cannot change AUC or precision-at-fixed-recall by
construction -- it only remaps propensity to be a better-calibrated probability, so a
flat AUC after calibration isn't a failure, it's expected.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def fit_calibrator(models: list, val_entries: list[dict], device: str) -> IsotonicRegression:
    from gnn import ensemble_predict  # local import to avoid a circular import (gnn.py imports this module too)

    all_y, all_prop = [], []
    for e in val_entries:
        prop, _ = ensemble_predict(models, e, device)
        all_y.append(e["labels"].numpy())
        all_prop.append(prop.numpy())
    y = np.concatenate(all_y)
    propensity = np.concatenate(all_prop)

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(propensity, y)
    return calibrator


def apply_calibration(calibrator: IsotonicRegression, propensity: np.ndarray) -> np.ndarray:
    return calibrator.predict(propensity)


def save_calibrator(calibrator: IsotonicRegression, feature_set: str) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_DIR / f"calibrator_{feature_set}.pkl", "wb") as f:
        pickle.dump(calibrator, f)


def load_calibrator(feature_set: str) -> IsotonicRegression | None:
    path = CHECKPOINT_DIR / f"calibrator_{feature_set}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
