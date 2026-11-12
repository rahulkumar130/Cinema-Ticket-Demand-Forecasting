"""Tests for chronological train/validation/test splitting."""

from __future__ import annotations

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.split import get_rolling_origin_cv, split_chronological


def test_chronological_split_has_no_overlap_or_future_leak() -> None:
    cfg = load_config()
    feat_df = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["features_data"])
    train_df, val_df, test_df, dates = split_chronological(
        feat_df,
        train_frac=float(cfg["split"]["train_frac"]),
        val_frac=float(cfg["split"]["val_frac"]),
    )

    assert len(train_df) + len(val_df) + len(test_df) == len(feat_df)
    assert train_df["show_date"].max() <= dates.train_end_date
    assert val_df["show_date"].min() > dates.train_end_date
    assert val_df["show_date"].max() <= dates.val_end_date
    assert test_df["show_date"].min() > dates.val_end_date

    # Zero show_id overlap across splits
    assert set(train_df["show_id"]).isdisjoint(set(val_df["show_id"]))
    assert set(train_df["show_id"]).isdisjoint(set(test_df["show_id"]))
    assert set(val_df["show_id"]).isdisjoint(set(test_df["show_id"]))


def test_rolling_origin_cv_is_strictly_chronological() -> None:
    cfg = load_config()
    feat_df = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["features_data"])
    train_df, _, _, _ = split_chronological(feat_df)

    cv = get_rolling_origin_cv(n_splits=4)
    for tr_idx, va_idx in cv.split(train_df):
        assert train_df.iloc[tr_idx]["show_datetime"].max() <= train_df.iloc[va_idx]["show_datetime"].min()

