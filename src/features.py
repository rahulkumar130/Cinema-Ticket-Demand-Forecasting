"""T-48h-safe feature engineering for show-level cinema ticket demand forecasting.

Strict temporal contract:
    Every historical/lagged feature for a target show at `show_datetime = t`
    may ONLY aggregate completed shows with `history_show_datetime < t - 48 hours`.
    Same-day shows, T-24h shows, and target values are strictly excluded.

Usage:
    python -m src.features
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config

# Explicit feature registry with availability documentation (used in leakage audit & API)
FEATURE_AVAILABILITY: dict[str, str] = {
    # Calendar features (known when schedule is published, weeks ahead)
    "weekday": "Schedule creation (static calendar)",
    "month": "Schedule creation (static calendar)",
    "is_weekend": "Schedule creation (static calendar)",
    "is_tuesday": "Schedule creation (static calendar)",
    "is_holiday": "Schedule creation (public holiday calendar)",
    "is_special_event": "Schedule creation (cinema promo calendar)",
    "slot": "Schedule creation (showtime slot)",
    "show_hour": "Schedule creation (showtime)",
    # Screen & pricing features (known at T-48h)
    "screen_id": "Schedule creation (assigned screen)",
    "screen_type": "Schedule creation (screen format)",
    "capacity": "Schedule creation (physical hall seats)",
    "ticket_price": "T-48h pricing schedule",
    # Movie metadata features (known at T-48h)
    "genre": "Movie metadata (pre-release)",
    "language": "Movie metadata (pre-release)",
    "rating": "Movie metadata (pre-release)",
    "runtime_min": "Movie metadata (pre-release)",
    "movie_age_days": "Schedule creation (show_date - release_date)",
    # Historical demand features (strictly < show_datetime - 48h)
    "shows_so_far_movie": "T-48h history (count of movie shows < t - 48h)",
    "is_new_movie": "T-48h history / schedule (age <= 3d or 0 prior shows < t - 48h)",
    "is_low_history": "T-48h history (shows_so_far_movie <= 5)",
    "movie_occ_mean_all": "T-48h history (expanding movie occupancy < t - 48h)",
    "movie_occ_mean_7d": "T-48h history (rolling 7d movie occupancy in [t-9d, t-48h))",
    "movie_occ_mean_14d": "T-48h history (rolling 14d movie occupancy in [t-16d, t-48h))",
    "movie_sold_out_rate_all": "T-48h history (expanding movie sold-out rate < t - 48h)",
    "movie_slot_occ_mean": "T-48h history (expanding movie x slot occupancy < t - 48h)",
    "movie_weekday_occ_mean": "T-48h history (expanding movie x weekday occupancy < t - 48h)",
    # Cold-start priors (strictly < show_datetime - 48h across catalog)
    "genre_slot_occ_prior": "T-48h history (expanding genre x slot occupancy < t - 48h)",
    "genre_occ_prior": "T-48h history (expanding genre occupancy < t - 48h)",
    "slot_weekday_occ_prior": "T-48h history (expanding slot x weekday occupancy < t - 48h)",
    "movie_effective_occ_prior": "T-48h hierarchical prior (7d -> all -> genre_slot -> slot_weekday)",
}

CATEGORICAL_FEATURES = ["slot", "screen_id", "screen_type", "genre", "language", "rating"]
NUMERIC_FEATURES = [
    "weekday",
    "month",
    "is_weekend",
    "is_tuesday",
    "is_holiday",
    "is_special_event",
    "show_hour",
    "capacity",
    "ticket_price",
    "runtime_min",
    "movie_age_days",
    "shows_so_far_movie",
    "is_new_movie",
    "is_low_history",
    "movie_occ_mean_all",
    "movie_occ_mean_7d",
    "movie_occ_mean_14d",
    "movie_sold_out_rate_all",
    "movie_slot_occ_mean",
    "movie_weekday_occ_mean",
    "genre_slot_occ_prior",
    "genre_occ_prior",
    "slot_weekday_occ_prior",
    "movie_effective_occ_prior",
]
ALL_FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _compute_group_cutoff_stats(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    horizon_hours: int,
    window_days: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute strictly causal (< t - horizon) group mean and count via searchsorted.

    Uses sorted timestamps and prefix sums per group so no row can ever read
    its own value, a concurrent show, or any show within `horizon_hours`.
    """
    n = len(df)
    means = np.full(n, np.nan, dtype=float)
    counts = np.zeros(n, dtype=float)

    horizon_ns = np.int64(horizon_hours) * np.int64(3600) * np.int64(1_000_000_000)
    window_ns = (
        np.int64(window_days) * np.int64(86400) * np.int64(1_000_000_000)
        if window_days is not None
        else None
    )

    ts_all = df["show_datetime"].astype("datetime64[ns]").astype("int64").to_numpy()
    vals_all = df[value_col].to_numpy(dtype=float)

    for _, idx_arr in df.groupby(group_cols, sort=False).indices.items():
        idx = np.asarray(idx_arr, dtype=int)
        # Sort indices chronologically within group
        order = np.argsort(ts_all[idx], kind="mergesort")
        s_idx = idx[order]
        s_ts = ts_all[s_idx]
        s_vals = vals_all[s_idx]

        prefix = np.concatenate(([0.0], np.cumsum(s_vals)))
        cutoffs = s_ts - horizon_ns
        right_pos = np.searchsorted(s_ts, cutoffs, side="left")

        if window_ns is None:
            left_pos = np.zeros_like(right_pos)
        else:
            left_pos = np.searchsorted(s_ts, cutoffs - window_ns, side="left")

        cnt = (right_pos - left_pos).astype(float)
        sums = prefix[right_pos] - prefix[left_pos]

        grp_means = np.where(cnt > 0, sums / np.maximum(cnt, 1.0), np.nan)
        means[s_idx] = grp_means
        counts[s_idx] = cnt

    return means, counts


def build_features(
    clean_df: pd.DataFrame,
    horizon_hours: int = 48,
    new_movie_max_age_days: int = 3,
    low_history_max_shows: int = 5,
) -> pd.DataFrame:
    """Build all T-48h-safe features from a cleaned show-level DataFrame."""
    df = clean_df.copy()
    df["show_datetime"] = pd.to_datetime(df["show_datetime"])
    df["release_date"] = pd.to_datetime(df["release_date"])
    df = df.sort_values(["show_datetime", "screen_id"]).reset_index(drop=True)

    # 1. Calendar features
    df["weekday"] = df["show_datetime"].dt.weekday.astype(int)
    df["month"] = df["show_datetime"].dt.month.astype(int)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_tuesday"] = (df["weekday"] == 1).astype(int)
    df["show_hour"] = (
        df["show_datetime"].dt.hour + df["show_datetime"].dt.minute / 60.0
    ).round(2)

    # 2. Movie age (clipped at 0 for any pre-release preview)
    df["movie_age_days"] = (
        (df["show_datetime"].dt.normalize() - df["release_date"].dt.normalize()).dt.days
    ).clip(lower=0)

    # 3. Strictly causal (< t - horizon_hours) historical demand features
    occ_all, cnt_all = _compute_group_cutoff_stats(
        df, ["movie_id"], "occupancy_rate", horizon_hours=horizon_hours, window_days=None
    )
    occ_7d, _ = _compute_group_cutoff_stats(
        df, ["movie_id"], "occupancy_rate", horizon_hours=horizon_hours, window_days=7
    )
    occ_14d, _ = _compute_group_cutoff_stats(
        df, ["movie_id"], "occupancy_rate", horizon_hours=horizon_hours, window_days=14
    )
    so_all, _ = _compute_group_cutoff_stats(
        df, ["movie_id"], "sold_out", horizon_hours=horizon_hours, window_days=None
    )
    occ_movie_slot, _ = _compute_group_cutoff_stats(
        df, ["movie_id", "slot"], "occupancy_rate", horizon_hours=horizon_hours, window_days=None
    )
    occ_movie_wd, _ = _compute_group_cutoff_stats(
        df, ["movie_id", "weekday"], "occupancy_rate", horizon_hours=horizon_hours, window_days=None
    )

    # 4. Cold-start priors strictly < t - horizon_hours
    genre_slot_prior, _ = _compute_group_cutoff_stats(
        df, ["genre", "slot"], "occupancy_rate", horizon_hours=horizon_hours, window_days=None
    )
    genre_prior, _ = _compute_group_cutoff_stats(
        df, ["genre"], "occupancy_rate", horizon_hours=horizon_hours, window_days=None
    )
    slot_wd_prior, _ = _compute_group_cutoff_stats(
        df, ["slot", "weekday"], "occupancy_rate", horizon_hours=horizon_hours, window_days=None
    )

    df["shows_so_far_movie"] = cnt_all.astype(int)
    df["is_new_movie"] = (
        (df["movie_age_days"] <= new_movie_max_age_days) | (df["shows_so_far_movie"] == 0)
    ).astype(int)
    df["is_low_history"] = (df["shows_so_far_movie"] <= low_history_max_shows).astype(int)

    df["movie_occ_mean_all"] = occ_all
    df["movie_occ_mean_7d"] = occ_7d
    df["movie_occ_mean_14d"] = occ_14d
    df["movie_sold_out_rate_all"] = so_all
    df["movie_slot_occ_mean"] = occ_movie_slot
    df["movie_weekday_occ_mean"] = occ_movie_wd

    df["genre_slot_occ_prior"] = genre_slot_prior
    df["genre_occ_prior"] = genre_prior
    df["slot_weekday_occ_prior"] = slot_wd_prior

    # Hierarchical cold-start prior combining movie history with genre/slot fallbacks
    df["movie_effective_occ_prior"] = (
        df["movie_occ_mean_7d"]
        .combine_first(df["movie_occ_mean_all"])
        .combine_first(df["genre_slot_occ_prior"])
        .combine_first(df["genre_occ_prior"])
        .combine_first(df["slot_weekday_occ_prior"])
    )

    return df


def run_feature_pipeline(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load clean parquet, engineer T-48h features, and save to disk."""
    cfg = config or load_config()
    clean_path = PROJECT_ROOT / cfg["paths"]["cleaned_data"]
    feat_path = PROJECT_ROOT / cfg["paths"]["features_data"]

    clean_df = pd.read_parquet(clean_path)
    feat_df = build_features(
        clean_df,
        horizon_hours=int(cfg["project"]["horizon_hours"]),
        new_movie_max_age_days=int(cfg["features"]["new_movie_max_age_days"]),
        low_history_max_shows=int(cfg["features"]["low_history_max_shows"]),
    )

    feat_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(feat_path, index=False)
    return feat_df


def main() -> None:
    feat_df = run_feature_pipeline()
    print(f"Engineered {len(ALL_FEATURE_COLUMNS)} features for {len(feat_df):,} shows.")
    print(f"New-movie shows: {feat_df['is_new_movie'].sum():,} ({feat_df['is_new_movie'].mean()*100:.2f}%)")
    print(f"Low-history shows: {feat_df['is_low_history'].sum():,} ({feat_df['is_low_history'].mean()*100:.2f}%)")
    print(f"Saved features -> {load_config()['paths']['features_data']}")


if __name__ == "__main__":
    main()

