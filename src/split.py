"""Chronological train / validation / test splitting and TimeSeriesSplit helper.

Guarantees:
1. Strict chronological ordering by `show_date` (no same-day straddling).
2. Train < Validation < Test with zero temporal overlap.
3. Persists resolved split dates back to `config.yaml` for auditability.

Usage:
    python -m src.split
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit

from src.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config


@dataclass(frozen=True)
class SplitDates:
    """Inclusive end dates for chronological train / validation / test partitions."""

    train_start_date: str
    train_end_date: str
    val_end_date: str
    test_end_date: str


def compute_split_dates(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> SplitDates:
    """Compute day-level chronological boundaries targeting train_frac / val_frac."""
    sorted_dates = pd.to_datetime(df["show_date"]).sort_values().reset_index(drop=True)
    n = len(sorted_dates)
    if n < 10:
        raise ValueError(f"Dataset too small to split chronologically ({n} rows).")

    train_idx = int(n * train_frac) - 1
    val_idx = int(n * (train_frac + val_frac)) - 1

    train_start = sorted_dates.iloc[0].strftime("%Y-%m-%d")
    train_end = sorted_dates.iloc[train_idx].strftime("%Y-%m-%d")
    val_end = sorted_dates.iloc[val_idx].strftime("%Y-%m-%d")
    test_end = sorted_dates.iloc[-1].strftime("%Y-%m-%d")

    if not (train_start <= train_end < val_end < test_end):
        raise ValueError(
            f"Invalid split boundaries: {train_start} <= {train_end} < {val_end} < {test_end}"
        )

    return SplitDates(
        train_start_date=train_start,
        train_end_date=train_end,
        val_end_date=val_end,
        test_end_date=test_end,
    )


def split_chronological(
    df: pd.DataFrame,
    split_dates: SplitDates | None = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitDates]:
    """Split a show-level DataFrame into chronological train, val, and test sets."""
    ordered = df.sort_values(["show_datetime", "screen_id"]).reset_index(drop=True)
    dates = split_dates or compute_split_dates(ordered, train_frac=train_frac, val_frac=val_frac)

    show_dates = ordered["show_date"].astype(str)
    train_df = ordered[show_dates <= dates.train_end_date].copy()
    val_df = ordered[(show_dates > dates.train_end_date) & (show_dates <= dates.val_end_date)].copy()
    test_df = ordered[show_dates > dates.val_end_date].copy()

    return train_df, val_df, test_df, dates


def write_split_dates_to_config(
    dates: SplitDates,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> None:
    """Persist resolved chronological split boundaries into config.yaml."""
    cfg = load_config(config_path)
    cfg["split"]["train_end_date"] = dates.train_end_date
    cfg["split"]["val_end_date"] = dates.val_end_date
    cfg["split"]["test_end_date"] = dates.test_end_date
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def get_rolling_origin_cv(n_splits: int = 4) -> TimeSeriesSplit:
    """Return expanding-window TimeSeriesSplit for chronological CV on train/val."""
    return TimeSeriesSplit(n_splits=n_splits)


def run_split(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitDates]:
    """Load engineered features, split chronologically, and persist split dates."""
    cfg = config or load_config()
    feat_path = PROJECT_ROOT / cfg["paths"]["features_data"]
    df = pd.read_parquet(feat_path)

    train_df, val_df, test_df, dates = split_chronological(
        df,
        train_frac=float(cfg["split"]["train_frac"]),
        val_frac=float(cfg["split"]["val_frac"]),
    )
    write_split_dates_to_config(dates)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "val.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)
    return train_df, val_df, test_df, dates


def main() -> None:
    train_df, val_df, test_df, dates = run_split()
    total = len(train_df) + len(val_df) + len(test_df)
    print("Chronological split complete:")
    print(f"  Train: {len(train_df):,} rows ({len(train_df)/total*100:.1f}%) | {dates.train_start_date} .. {dates.train_end_date}")
    print(f"  Val:   {len(val_df):,} rows ({len(val_df)/total*100:.1f}%) | > {dates.train_end_date} .. {dates.val_end_date}")
    print(f"  Test:  {len(test_df):,} rows ({len(test_df)/total*100:.1f}%) | > {dates.val_end_date} .. {dates.test_end_date}")


if __name__ == "__main__":
    main()

