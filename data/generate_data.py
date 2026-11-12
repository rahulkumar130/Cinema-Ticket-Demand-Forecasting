"""Synthetic cinema show-level ticket demand generator.

Generates 12-18 months (default 540 days -> ~21,600 shows) of realistic,
deterministic show-level cinema data with latent demand drivers, capacity
censoring (sold-out shows), and injected data-quality anomalies for cleaning.

Usage:
    python -m data.generate_data
    python -m data.generate_data --seed 42 --n-days 540 --n-movies 65 --n-screens 8
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config, set_global_seed

GENRES = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Animation", "Romance", "Thriller"]
LANGUAGES = ["English", "Hindi", "Spanish", "Regional"]
RATINGS = ["G", "PG", "PG-13", "R"]
SLOTS = [
    ("11:00", "matinee"),
    ("14:30", "matinee"),
    ("18:00", "evening"),
    ("20:30", "evening"),
    ("22:45", "late"),
]

# Fixed public holidays (month, day) repeated across years in the simulation
PUBLIC_HOLIDAYS_MD = {
    (1, 1),    # New Year's Day
    (1, 26),   # Republic / National Holiday
    (2, 14),   # Valentine's Day
    (5, 1),    # Labour Day
    (7, 4),    # Mid-summer Holiday
    (8, 15),   # Independence Holiday
    (10, 31),  # Halloween (boosts Horror)
    (11, 12),  # Festive Holiday
    (12, 25),  # Christmas Day
    (12, 31),  # New Year's Eve
}


def build_screens(n_screens: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create screen catalog with varying capacities and formats."""
    templates = [
        ("SCR-01", "Standard", 90),
        ("SCR-02", "Standard", 115),
        ("SCR-03", "Standard", 145),
        ("SCR-04", "Dolby", 175),
        ("SCR-05", "Standard", 200),
        ("SCR-06", "Dolby", 235),
        ("SCR-07", "VIP", 105),
        ("SCR-08", "IMAX", 300),
        ("SCR-09", "Standard", 160),
        ("SCR-10", "IMAX", 280),
    ]
    chosen = templates[:n_screens]
    rows = []
    for screen_id, screen_type, cap in chosen:
        jitter = int(rng.integers(-5, 6))
        rows.append(
            {
                "screen_id": screen_id,
                "screen_type": screen_type,
                "capacity": max(60, cap + jitter),
            }
        )
    return pd.DataFrame(rows)


def build_movies(
    n_movies: int,
    start_date: datetime,
    n_days: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create movie catalog with metadata and a hidden latent popularity factor."""
    genre_probs = [0.20, 0.16, 0.14, 0.11, 0.12, 0.11, 0.08, 0.08]
    lang_probs = [0.55, 0.25, 0.10, 0.10]

    # Mix of ~12% blockbusters, ~25% sleepers/hits, and regular releases
    is_blockbuster = rng.random(n_movies) < 0.12
    # Ensure at least 4 blockbusters exist
    if is_blockbuster.sum() < 4:
        is_blockbuster[:4] = True
        rng.shuffle(is_blockbuster)

    rows = []
    # First 8 movies are already in theaters at start_date (warm catalog)
    for idx in range(n_movies):
        genre = str(rng.choice(GENRES, p=genre_probs))
        language = str(rng.choice(LANGUAGES, p=lang_probs))

        if genre == "Animation":
            rating = str(rng.choice(["G", "PG"], p=[0.45, 0.55]))
            runtime_min = int(rng.integers(88, 118))
        elif genre == "Horror":
            rating = str(rng.choice(["PG-13", "R"], p=[0.30, 0.70]))
            runtime_min = int(rng.integers(90, 125))
        else:
            rating = str(rng.choice(RATINGS, p=[0.08, 0.24, 0.44, 0.24]))
            runtime_min = int(rng.integers(95, 170))

        if idx < 8:
            rel_offset = int(rng.integers(-21, -2))
        else:
            rel_offset = int(rng.integers(0, max(1, n_days - 14)))

        release_date = start_date + timedelta(days=rel_offset)

        # Latent popularity (hidden from ML features): log-normal base + blockbuster boost
        base_pop = float(np.clip(rng.normal(0.50, 0.14), 0.18, 0.85))
        if is_blockbuster[idx]:
            base_pop = float(np.clip(rng.normal(0.86, 0.07), 0.75, 0.99))

        rows.append(
            {
                "movie_id": f"MOV-{idx + 1:03d}",
                "movie_title": f"{genre} Feature #{idx + 1:02d}",
                "genre": genre,
                "language": language,
                "rating": rating,
                "runtime_min": runtime_min,
                "release_date": release_date.strftime("%Y-%m-%d"),
                "_hidden_popularity": base_pop,
                "_is_blockbuster": bool(is_blockbuster[idx]),
            }
        )
    return pd.DataFrame(rows)


def _compute_show_occupancy(
    *,
    movie_row: pd.Series,
    screen_row: pd.Series,
    show_dt: datetime,
    slot: str,
    ticket_price: float,
    is_holiday: bool,
    is_special_event: bool,
    rng: np.random.Generator,
) -> float:
    """Compute realistic latent occupancy rate before capacity capping."""
    rel_dt = datetime.strptime(movie_row["release_date"], "%Y-%m-%d")
    age_days = (show_dt.date() - rel_dt.date()).days

    # 1. Base latent movie popularity (0.18 .. 0.99)
    pop = float(movie_row["_hidden_popularity"])

    # 2. Age decay curve: strong opening weekend, gradual decay over 6-8 weeks
    if age_days <= 2:
        age_factor = 1.32
    elif age_days <= 6:
        age_factor = 1.06
    else:
        age_factor = float(np.exp(-0.042 * (age_days - 6)))
        age_factor = max(0.18, age_factor)

    # 3. Calendar effects
    dow = show_dt.weekday()  # 0=Mon .. 6=Sun
    dow_mult = {
        0: 0.62,  # Mon
        1: 0.74,  # Tue (midweek discount bump)
        2: 0.66,  # Wed
        3: 0.75,  # Thu
        4: 1.08,  # Fri
        5: 1.32,  # Sat
        6: 1.18,  # Sun
    }[dow]
    holiday_mult = 1.28 if is_holiday else 1.0
    event_mult = 1.24 if is_special_event else 1.0

    # 4. Showtime slot & genre interactions
    slot_mult = {"matinee": 0.68, "evening": 1.18, "late": 0.82}[slot]
    genre = str(movie_row["genre"])
    genre_effect = {
        "Action": 1.08,
        "Sci-Fi": 1.06,
        "Animation": 1.04,
        "Comedy": 1.00,
        "Thriller": 0.98,
        "Horror": 0.95,
        "Drama": 0.90,
        "Romance": 0.92,
    }[genre]

    if genre == "Animation" and slot == "matinee" and dow in (5, 6):
        slot_mult *= 1.38
    elif genre == "Horror" and slot == "late" and dow in (4, 5):
        slot_mult *= 1.32
    elif genre == "Drama" and slot == "late":
        slot_mult *= 0.82

    # 5. Screen format & price elasticity
    screen_type = str(screen_row["screen_type"])
    screen_mult = {"Standard": 1.0, "Dolby": 1.08, "IMAX": 1.15, "VIP": 1.04}[screen_type]
    base_ref_price = {"Standard": 12.0, "Dolby": 15.0, "IMAX": 18.0, "VIP": 21.0}[screen_type]
    price_effect = 1.0 - 0.025 * (ticket_price - base_ref_price)
    price_effect = float(np.clip(price_effect, 0.72, 1.22))

    # Combine multiplicatively around a realistic baseline occupancy (~0.56)
    expected_occ = (
        0.56
        * (0.45 + 0.95 * pop)
        * age_factor
        * dow_mult
        * holiday_mult
        * event_mult
        * slot_mult
        * genre_effect
        * screen_mult
        * price_effect
    )

    # Add realistic show-level noise (normal + small day-to-day jitter)
    noisy_occ = expected_occ + float(rng.normal(0.0, 0.065))
    return float(max(0.02, noisy_occ))


def _inject_quality_issues(
    df: pd.DataFrame,
    missing_rate: float,
    duplicate_rate: float,
    outlier_rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject realistic missingness, duplicates, and anomalous outliers."""
    out = df.copy()
    n = len(out)

    # 1. Missing values (~1-2% across nullable metadata/price columns)
    nullable_cols = ["ticket_price", "runtime_min", "rating", "language"]
    for col in nullable_cols:
        mask = rng.random(n) < (missing_rate / len(nullable_cols) * 1.6)
        out.loc[mask, col] = np.nan

    # 2. Outlier anomalies (bad POS prices, impossible runtimes, negative/over-cap tickets)
    n_outliers = max(3, int(n * outlier_rate))
    outlier_idx = rng.choice(out.index, size=n_outliers, replace=False)
    for i, idx in enumerate(outlier_idx):
        kind = i % 4
        if kind == 0:
            out.loc[idx, "ticket_price"] = float(rng.choice([-5.0, 149.0, 299.0]))
        elif kind == 1:
            out.loc[idx, "runtime_min"] = int(rng.choice([12, 540]))
        elif kind == 2:
            out.loc[idx, "tickets_sold"] = int(out.loc[idx, "capacity"]) + int(rng.integers(5, 35))
        else:
            out.loc[idx, "tickets_sold"] = -int(rng.integers(1, 10))

    # 3. Duplicate rows (exact duplicate POS records)
    n_dups = max(2, int(n * duplicate_rate))
    dup_rows = out.sample(n=n_dups, random_state=int(rng.integers(0, 10000)))
    out = pd.concat([out, dup_rows], ignore_index=True)

    return out


def generate_dataset(
    seed: int = 42,
    start_date_str: str = "2024-01-01",
    n_days: int = 540,
    n_movies: int = 65,
    n_screens: int = 8,
    missing_rate: float = 0.015,
    duplicate_rate: float = 0.005,
    outlier_rate: float = 0.003,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Generate the full synthetic show-level dataset and optionally save to CSV."""
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    screens = build_screens(n_screens=n_screens, rng=rng)
    movies = build_movies(n_movies=n_movies, start_date=start_date, n_days=n_days, rng=rng)

    # Pre-sample ~2% special event days (festivals, cinema day promos)
    special_event_days = set(
        rng.choice(np.arange(n_days), size=max(5, int(n_days * 0.02)), replace=False).tolist()
    )

    records = []
    show_counter = 1

    for day_offset in range(n_days):
        cur_date = start_date + timedelta(days=day_offset)
        cur_date_str = cur_date.strftime("%Y-%m-%d")
        is_holiday = (cur_date.month, cur_date.day) in PUBLIC_HOLIDAYS_MD
        is_special_event = day_offset in special_event_days

        # Active movies: released on or before today and within active run window (<= 65 days)
        rel_dates = pd.to_datetime(movies["release_date"])
        age_series = (pd.Timestamp(cur_date) - rel_dates).dt.days
        active_movies = movies[(age_series >= 0) & (age_series <= 65)]
        if active_movies.empty:
            active_movies = movies.iloc[:4]

        # Weight movie scheduling probability so fresher/popular movies get more screens
        ages = (pd.Timestamp(cur_date) - pd.to_datetime(active_movies["release_date"])).dt.days.values
        sched_weights = active_movies["_hidden_popularity"].values * np.exp(-0.03 * np.maximum(0, ages))
        sched_weights = sched_weights / sched_weights.sum()

        for _, screen_row in screens.iterrows():
            # Assign 1-2 movies per screen across its 5 daily slots
            chosen_indices = rng.choice(
                active_movies.index,
                size=min(2, len(active_movies)),
                replace=False,
                p=sched_weights,
            )
            for slot_idx, (show_time, slot_name) in enumerate(SLOTS):
                movie_idx = chosen_indices[0] if slot_idx in (2, 3) else chosen_indices[-1]
                movie_row = active_movies.loc[movie_idx]

                show_dt = datetime.strptime(f"{cur_date_str} {show_time}", "%Y-%m-%d %H:%M")

                # Price depends on screen format, slot, weekend/holiday, and promo event
                base_price = {"Standard": 12.0, "Dolby": 15.0, "IMAX": 18.0, "VIP": 21.0}[
                    screen_row["screen_type"]
                ]
                slot_adj = {"matinee": -2.0, "evening": 1.0, "late": 0.0}[slot_name]
                wknd_adj = 1.5 if cur_date.weekday() >= 4 or is_holiday else 0.0
                tue_disc = -2.5 if cur_date.weekday() == 1 else 0.0
                event_adj = -3.0 if is_special_event else 0.0
                ticket_price = round(
                    max(6.0, base_price + slot_adj + wknd_adj + tue_disc + event_adj + float(rng.normal(0, 0.5))),
                    2,
                )

                latent_occ = _compute_show_occupancy(
                    movie_row=movie_row,
                    screen_row=screen_row,
                    show_dt=show_dt,
                    slot=slot_name,
                    ticket_price=ticket_price,
                    is_holiday=is_holiday,
                    is_special_event=is_special_event,
                    rng=rng,
                )
                cap = int(screen_row["capacity"])
                unconstrained_tickets = int(round(latent_occ * cap))
                tickets_sold = int(np.clip(unconstrained_tickets, 0, cap))

                records.append(
                    {
                        "show_id": f"SHW-{show_counter:06d}",
                        "show_date": cur_date_str,
                        "show_time": show_time,
                        "show_datetime": show_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "slot": slot_name,
                        "screen_id": screen_row["screen_id"],
                        "screen_type": screen_row["screen_type"],
                        "capacity": cap,
                        "movie_id": movie_row["movie_id"],
                        "movie_title": movie_row["movie_title"],
                        "genre": movie_row["genre"],
                        "language": movie_row["language"],
                        "rating": movie_row["rating"],
                        "runtime_min": int(movie_row["runtime_min"]),
                        "release_date": movie_row["release_date"],
                        "ticket_price": ticket_price,
                        "is_holiday": int(is_holiday),
                        "is_special_event": int(is_special_event),
                        "tickets_sold": tickets_sold,
                    }
                )
                show_counter += 1

    raw_df = pd.DataFrame(records)
    raw_df = _inject_quality_issues(
        raw_df,
        missing_rate=missing_rate,
        duplicate_rate=duplicate_rate,
        outlier_rate=outlier_rate,
        rng=rng,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(output_path, index=False)

    return raw_df


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    dg = cfg["data_generation"]
    parser = argparse.ArgumentParser(description="Generate synthetic cinema demand data.")
    parser.add_argument("--seed", type=int, default=dg["seed"])
    parser.add_argument("--start-date", type=str, default=dg["start_date"])
    parser.add_argument("--n-days", type=int, default=dg["n_days"])
    parser.add_argument("--n-movies", type=int, default=dg["n_movies"])
    parser.add_argument("--n-screens", type=int, default=dg["n_screens"])
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / cfg["paths"]["raw_data"],
    )
    return parser.parse_args()


def main() -> None:
    cfg = load_config()
    dg = cfg["data_generation"]
    args = parse_args()
    df = generate_dataset(
        seed=args.seed,
        start_date_str=args.start_date,
        n_days=args.n_days,
        n_movies=args.n_movies,
        n_screens=args.n_screens,
        missing_rate=dg["missing_rate"],
        duplicate_rate=dg["duplicate_rate"],
        outlier_rate=dg["outlier_rate"],
        output_path=args.output,
    )
    print(f"Generated {len(df):,} raw show records -> {args.output}")
    print(f"Date range: {df['show_date'].min()} to {df['show_date'].max()}")
    print(f"Unique movies: {df['movie_id'].nunique()} | Unique screens: {df['screen_id'].nunique()}")


if __name__ == "__main__":
    main()

