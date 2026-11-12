"""Tests proving T-48h feature engineering has zero target or future leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PROJECT_ROOT, load_config
from src.features import ALL_FEATURE_COLUMNS, FEATURE_AVAILABILITY, build_features


def test_every_feature_has_documented_availability() -> None:
    forbidden = {"tickets_sold", "occupancy_rate", "sold_out", "show_id", "show_datetime"}
    assert set(ALL_FEATURE_COLUMNS).isdisjoint(forbidden)
    for col in ALL_FEATURE_COLUMNS:
        assert col in FEATURE_AVAILABILITY, f"Missing availability doc for {col}"


def test_causal_features_match_bruteforce_strict_t48h() -> None:
    cfg = load_config()
    clean_df = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["cleaned_data"]).head(2500).copy()
    feat_df = build_features(clean_df, horizon_hours=48)

    rng = np.random.default_rng(42)
    sample_positions = rng.choice(len(feat_df), size=25, replace=False)

    for pos in sample_positions:
        row = feat_df.iloc[pos]
        cutoff = row["show_datetime"] - pd.Timedelta(hours=48)
        hist_movie = feat_df[
            (feat_df["movie_id"] == row["movie_id"]) & (feat_df["show_datetime"] < cutoff)
        ]

        assert row["shows_so_far_movie"] == len(hist_movie)
        if len(hist_movie) == 0:
            assert np.isnan(row["movie_occ_mean_all"])
        else:
            assert row["movie_occ_mean_all"] == pytest.approx(hist_movie["occupancy_rate"].mean(), abs=1e-9)

        hist_7d = hist_movie[hist_movie["show_datetime"] >= cutoff - pd.Timedelta(days=7)]
        if len(hist_7d) == 0:
            assert np.isnan(row["movie_occ_mean_7d"])
        else:
            assert row["movie_occ_mean_7d"] == pytest.approx(hist_7d["occupancy_rate"].mean(), abs=1e-9)


def test_future_target_mutation_does_not_affect_t48h_features() -> None:
    """Perturbing targets within (t - 48h, +inf) must not change features at show t."""
    cfg = load_config()
    clean_df = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["cleaned_data"]).head(2000).copy()

    target_idx = 1200
    target_dt = pd.to_datetime(clean_df.iloc[target_idx]["show_datetime"])
    cutoff = target_dt - pd.Timedelta(hours=48)

    baseline_feats = build_features(clean_df, horizon_hours=48).iloc[target_idx][ALL_FEATURE_COLUMNS]

    perturbed_df = clean_df.copy()
    future_or_horizon_mask = pd.to_datetime(perturbed_df["show_datetime"]) >= cutoff
    perturbed_df.loc[future_or_horizon_mask, "occupancy_rate"] = 0.99
    perturbed_df.loc[future_or_horizon_mask, "sold_out"] = 1

    perturbed_feats = build_features(perturbed_df, horizon_hours=48).iloc[target_idx][ALL_FEATURE_COLUMNS]
    pd.testing.assert_series_equal(baseline_feats, perturbed_feats)

