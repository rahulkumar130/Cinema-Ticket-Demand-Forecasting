"""Data cleaning, outlier treatment, train-only imputation, and schema validation.

Pipeline responsibilities:
1. Deduplicate exact/key-duplicate show records.
2. Treat out-of-range feature outliers (POS glitches) by setting them to NaN
   prior to train-only imputation; handle corrupted target rows explicitly.
3. Fit imputation statistics strictly on the chronological training window
   (no validation/test leakage) and transform the full dataset.
4. Compute `occupancy_rate = tickets_sold / capacity` and `sold_out` flag.
5. Validate final schema and domain ranges using Pandera.

Usage:
    python -m src.data_prep
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from src.config import PROJECT_ROOT, load_config
from src.split import compute_split_dates

VALID_SLOTS = ["matinee", "evening", "late"]
VALID_SCREEN_TYPES = ["Standard", "Dolby", "IMAX", "VIP"]
VALID_GENRES = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Animation", "Romance", "Thriller"]
VALID_RATINGS = ["G", "PG", "PG-13", "R"]
VALID_LANGUAGES = ["English", "Hindi", "Spanish", "Regional"]


CLEAN_SCHEMA = DataFrameSchema(
    columns={
        "show_id": Column(str, nullable=False, unique=True),
        "show_date": Column(str, nullable=False),
        "show_time": Column(str, nullable=False),
        "show_datetime": Column("datetime64[ns]", nullable=False),
        "slot": Column(str, Check.isin(VALID_SLOTS), nullable=False),
        "screen_id": Column(str, nullable=False),
        "screen_type": Column(str, Check.isin(VALID_SCREEN_TYPES), nullable=False),
        "capacity": Column(int, Check.in_range(50, 500), nullable=False),
        "movie_id": Column(str, nullable=False),
        "movie_title": Column(str, nullable=False),
        "genre": Column(str, Check.isin(VALID_GENRES), nullable=False),
        "language": Column(str, Check.isin(VALID_LANGUAGES), nullable=False),
        "rating": Column(str, Check.isin(VALID_RATINGS), nullable=False),
        "runtime_min": Column(float, Check.in_range(60.0, 240.0), nullable=False),
        "release_date": Column("datetime64[ns]", nullable=False),
        "ticket_price": Column(float, Check.in_range(4.0, 60.0), nullable=False),
        "is_holiday": Column(int, Check.isin([0, 1]), nullable=False),
        "is_special_event": Column(int, Check.isin([0, 1]), nullable=False),
        "tickets_sold": Column(int, Check.ge(0), nullable=False),
        "occupancy_rate": Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "sold_out": Column(int, Check.isin([0, 1]), nullable=False),
    },
    checks=[
        Check(lambda df: (df["tickets_sold"] <= df["capacity"]).all(), error="tickets_sold exceeds capacity"),
        Check(
            lambda df: not df.duplicated(subset=["screen_id", "show_datetime"]).any(),
            error="Duplicate (screen_id, show_datetime) found",
        ),
    ],
    strict=False,
)


@dataclass
class ImputationStats:
    """Statistics fitted strictly on the training split for missing-value imputation."""

    price_by_screen_type: dict[str, float]
    global_price_median: float
    runtime_by_genre: dict[str, float]
    global_runtime_median: float
    rating_by_genre: dict[str, str]
    global_rating_mode: str
    global_language_mode: str


def clean_structural_and_outliers(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Deduplicate, parse dates, and neutralize impossible POS outliers.

    Outlier policy (documented for viva & report):
    - `ticket_price` outside [4.0, 60.0]: set to NaN (imputed via screen_type median).
    - `runtime_min` outside [60, 240]: set to NaN (imputed via genre median).
    - `tickets_sold < 0` or `tickets_sold > capacity`: dropped as corrupted POS target rows
      so double-counted/glitch records do not create false `sold_out` labels.
    """
    df = raw_df.copy()
    stats: dict[str, int] = {"raw_rows": len(df)}

    # 1. Parse timestamps and sort chronologically
    df["show_datetime"] = pd.to_datetime(df["show_datetime"])
    df["release_date"] = pd.to_datetime(df["release_date"])

    # 2. Deduplicate exact rows and conflicting (screen_id, show_datetime) slots
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["show_id"], keep="first")
    df = df.drop_duplicates(subset=["screen_id", "show_datetime"], keep="first")
    stats["duplicates_removed"] = before_dedup - len(df)

    # 3. Target sanity: drop corrupted negative or over-capacity POS target records
    neg_mask = df["tickets_sold"] < 0
    stats["negative_tickets_dropped"] = int(neg_mask.sum())
    over_cap_mask = df["tickets_sold"] > df["capacity"]
    stats["over_capacity_dropped"] = int(over_cap_mask.sum())
    df = df.loc[~(neg_mask | over_cap_mask)].copy()
    df["tickets_sold"] = df["tickets_sold"].astype(int)

    # 4. Feature outliers -> NaN (so train-only imputer fills them cleanly)
    bad_price_mask = (df["ticket_price"] < 4.0) | (df["ticket_price"] > 60.0)
    stats["price_outliers_nullified"] = int(bad_price_mask.sum())
    df.loc[bad_price_mask, "ticket_price"] = np.nan

    bad_runtime_mask = (df["runtime_min"] < 60.0) | (df["runtime_min"] > 240.0)
    stats["runtime_outliers_nullified"] = int(bad_runtime_mask.sum())
    df.loc[bad_runtime_mask, "runtime_min"] = np.nan

    # Sort chronologically for deterministic downstream splitting and lag generation
    df = df.sort_values(["show_datetime", "screen_id"]).reset_index(drop=True)
    stats["rows_after_structural_clean"] = len(df)
    return df, stats


def fit_imputation_stats(train_df: pd.DataFrame) -> ImputationStats:
    """Fit medians and modes strictly on the training slice."""
    price_by_screen = train_df.groupby("screen_type")["ticket_price"].median().dropna().to_dict()
    global_price = float(train_df["ticket_price"].median())

    runtime_by_genre = train_df.groupby("genre")["runtime_min"].median().dropna().to_dict()
    global_runtime = float(train_df["runtime_min"].median())

    rating_by_genre: dict[str, str] = {}
    for genre, grp in train_df.groupby("genre"):
        modes = grp["rating"].dropna().mode()
        if not modes.empty:
            rating_by_genre[str(genre)] = str(modes.iloc[0])

    global_rating = str(train_df["rating"].dropna().mode().iloc[0])
    global_language = str(train_df["language"].dropna().mode().iloc[0])

    return ImputationStats(
        price_by_screen_type={str(k): float(v) for k, v in price_by_screen.items()},
        global_price_median=global_price,
        runtime_by_genre={str(k): float(v) for k, v in runtime_by_genre.items()},
        global_runtime_median=global_runtime,
        rating_by_genre=rating_by_genre,
        global_rating_mode=global_rating,
        global_language_mode=global_language,
    )


def apply_imputation_stats(df: pd.DataFrame, imputer: ImputationStats) -> pd.DataFrame:
    """Apply pre-fitted train-only imputation statistics to any DataFrame."""
    out = df.copy()

    # Ticket price: group median by screen_type, fallback to global median
    price_fallback = out["screen_type"].map(imputer.price_by_screen_type).fillna(imputer.global_price_median)
    out["ticket_price"] = out["ticket_price"].fillna(price_fallback).astype(float)

    # Runtime: group median by genre, fallback to global median
    runtime_fallback = out["genre"].map(imputer.runtime_by_genre).fillna(imputer.global_runtime_median)
    out["runtime_min"] = out["runtime_min"].fillna(runtime_fallback).astype(float)

    # Rating: group mode by genre, fallback to global mode
    rating_fallback = out["genre"].map(imputer.rating_by_genre).fillna(imputer.global_rating_mode)
    out["rating"] = out["rating"].fillna(rating_fallback).astype(str)

    # Language: global mode
    out["language"] = out["language"].fillna(imputer.global_language_mode).astype(str)

    # Derived targets & censoring indicator
    out["capacity"] = out["capacity"].astype(int)
    out["tickets_sold"] = out["tickets_sold"].astype(int)
    out["occupancy_rate"] = (out["tickets_sold"] / out["capacity"]).clip(0.0, 1.0)
    out["sold_out"] = (out["tickets_sold"] >= out["capacity"]).astype(int)
    return out


def validate_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate DataFrame against the Pandera schema."""
    return CLEAN_SCHEMA.validate(df, lazy=True)


def run_data_prep(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run full Phase 2 cleaning pipeline with train-only imputation."""
    cfg = config or load_config()
    raw_path = PROJECT_ROOT / cfg["paths"]["raw_data"]
    clean_path = PROJECT_ROOT / cfg["paths"]["cleaned_data"]

    raw_df = pd.read_csv(raw_path)
    missing_before = raw_df.isna().sum().to_dict()

    df_struct, prep_stats = clean_structural_and_outliers(raw_df)

    # Identify chronological train cutoff by date before fitting imputer (zero leakage)
    dates = compute_split_dates(
        df_struct,
        train_frac=float(cfg["split"]["train_frac"]),
        val_frac=float(cfg["split"]["val_frac"]),
    )
    train_slice = df_struct[df_struct["show_date"].astype(str) <= dates.train_end_date]

    imputer = fit_imputation_stats(train_slice)
    df_clean = apply_imputation_stats(df_struct, imputer)
    df_clean = validate_clean_data(df_clean)

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(clean_path, index=False)

    summary = {
        **prep_stats,
        "final_clean_rows": len(df_clean),
        "train_imputation_cutoff": dates.train_end_date,
        "missing_before": {k: int(v) for k, v in missing_before.items() if v > 0},
        "missing_after": int(df_clean.isna().sum().sum()),
        "sold_out_shows": int(df_clean["sold_out"].sum()),
        "sold_out_pct": round(float(df_clean["sold_out"].mean() * 100), 2),
    }
    return df_clean, summary


def main() -> None:
    df_clean, summary = run_data_prep()
    print("Data cleaning & validation complete:")
    print(json.dumps(summary, indent=2))
    print(f"Saved clean dataset ({len(df_clean):,} rows) -> {load_config()['paths']['cleaned_data']}")


if __name__ == "__main__":
    main()

