"""Tests for synthetic data cleaning, outlier handling, and Pandera validation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import PROJECT_ROOT, load_config
from src.data_prep import (
    apply_imputation_stats,
    clean_structural_and_outliers,
    fit_imputation_stats,
    validate_clean_data,
)


def test_clean_dataset_exists_and_passes_pandera() -> None:
    cfg = load_config()
    clean_path = PROJECT_ROOT / cfg["paths"]["cleaned_data"]
    assert clean_path.exists(), f"Missing cleaned parquet at {clean_path}"

    df = pd.read_parquet(clean_path)
    assert len(df) >= 20000, f"Expected >= 20,000 rows, got {len(df)}"
    validated = validate_clean_data(df)
    assert validated["occupancy_rate"].between(0.0, 1.0).all()
    assert (validated["tickets_sold"] <= validated["capacity"]).all()
    assert validated.isna().sum().sum() == 0


def test_structural_cleaning_removes_dups_and_fixes_outliers() -> None:
    sample = pd.DataFrame(
        [
            {
                "show_id": "S1",
                "show_date": "2024-01-01",
                "show_time": "18:00",
                "show_datetime": "2024-01-01 18:00:00",
                "slot": "evening",
                "screen_id": "SCR-01",
                "screen_type": "Standard",
                "capacity": 100,
                "movie_id": "MOV-001",
                "movie_title": "Test Movie",
                "genre": "Action",
                "language": "English",
                "rating": "PG-13",
                "runtime_min": 120.0,
                "release_date": "2024-01-01",
                "ticket_price": 12.0,
                "is_holiday": 0,
                "is_special_event": 0,
                "tickets_sold": 80,
            },
            # Exact duplicate of S1
            {
                "show_id": "S1",
                "show_date": "2024-01-01",
                "show_time": "18:00",
                "show_datetime": "2024-01-01 18:00:00",
                "slot": "evening",
                "screen_id": "SCR-01",
                "screen_type": "Standard",
                "capacity": 100,
                "movie_id": "MOV-001",
                "movie_title": "Test Movie",
                "genre": "Action",
                "language": "English",
                "rating": "PG-13",
                "runtime_min": 120.0,
                "release_date": "2024-01-01",
                "ticket_price": 12.0,
                "is_holiday": 0,
                "is_special_event": 0,
                "tickets_sold": 80,
            },
            # Outlier price/runtime + over-capacity tickets
            {
                "show_id": "S2",
                "show_date": "2024-01-01",
                "show_time": "20:30",
                "show_datetime": "2024-01-01 20:30:00",
                "slot": "evening",
                "screen_id": "SCR-01",
                "screen_type": "Standard",
                "capacity": 100,
                "movie_id": "MOV-001",
                "movie_title": "Test Movie",
                "genre": "Action",
                "language": None,
                "rating": None,
                "runtime_min": 10.0,
                "release_date": "2024-01-01",
                "ticket_price": 999.0,
                "is_holiday": 0,
                "is_special_event": 0,
                "tickets_sold": 95,
            },
            # Corrupted over-capacity tickets row
            {
                "show_id": "S4",
                "show_date": "2024-01-01",
                "show_time": "21:30",
                "show_datetime": "2024-01-01 21:30:00",
                "slot": "evening",
                "screen_id": "SCR-02",
                "screen_type": "Standard",
                "capacity": 100,
                "movie_id": "MOV-001",
                "movie_title": "Test Movie",
                "genre": "Action",
                "language": "English",
                "rating": "PG-13",
                "runtime_min": 120.0,
                "release_date": "2024-01-01",
                "ticket_price": 12.0,
                "is_holiday": 0,
                "is_special_event": 0,
                "tickets_sold": 125,
            },
            # Corrupted negative tickets row
            {
                "show_id": "S3",
                "show_date": "2024-01-01",
                "show_time": "22:45",
                "show_datetime": "2024-01-01 22:45:00",
                "slot": "late",
                "screen_id": "SCR-01",
                "screen_type": "Standard",
                "capacity": 100,
                "movie_id": "MOV-001",
                "movie_title": "Test Movie",
                "genre": "Action",
                "language": "English",
                "rating": "PG-13",
                "runtime_min": 120.0,
                "release_date": "2024-01-01",
                "ticket_price": 12.0,
                "is_holiday": 0,
                "is_special_event": 0,
                "tickets_sold": -4,
            },
        ]
    )

    cleaned_struct, stats = clean_structural_and_outliers(sample)
    assert stats["duplicates_removed"] == 1
    assert stats["negative_tickets_dropped"] == 1
    assert stats["over_capacity_dropped"] == 1
    assert stats["price_outliers_nullified"] == 1
    assert stats["runtime_outliers_nullified"] == 1
    assert len(cleaned_struct) == 2

    # Fit imputer strictly on S1 (row 0) and apply to S1+S2
    imputer = fit_imputation_stats(cleaned_struct.iloc[:1])
    imputed = apply_imputation_stats(cleaned_struct, imputer)
    validated = validate_clean_data(imputed)
    assert validated.loc[1, "ticket_price"] == pytest.approx(12.0)
    assert validated.loc[1, "runtime_min"] == pytest.approx(120.0)
    assert validated.loc[1, "tickets_sold"] == 95
    assert validated.loc[1, "sold_out"] == 0

