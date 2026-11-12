"""Tests for saved model artifact, metadata, and capacity-clipped predictions."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.evaluate import clip_predictions


def test_saved_model_bundle_and_metadata_valid() -> None:
    cfg = load_config()
    bundle_path = PROJECT_ROOT / cfg["paths"]["model_bundle"]
    meta_path = PROJECT_ROOT / cfg["paths"]["model_metadata"]
    assert bundle_path.exists(), f"Missing model bundle: {bundle_path}"
    assert meta_path.exists(), f"Missing model metadata: {meta_path}"

    bundle = joblib.load(bundle_path)
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)

    assert meta["seed"] == cfg["project"]["seed"]
    assert meta["horizon_hours"] == 48
    assert bundle["best_model_name"] == meta["best_model_name"]

    val_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "val.parquet").head(25)
    X_sample = val_df[bundle["feature_columns"]]
    cap = val_df["capacity"].to_numpy(dtype=float)

    raw_occ = bundle["best_model"].predict(X_sample)
    pred_occ, pred_tix, _ = clip_predictions(raw_occ, cap)

    assert np.isfinite(pred_occ).all()
    assert np.isfinite(pred_tix).all()
    assert (pred_occ >= 0.0).all() and (pred_occ <= 1.0).all()
    assert (pred_tix >= 0.0).all() and (pred_tix <= cap).all()

