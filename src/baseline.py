"""Capacity-adjusted historical average baseline model.

Predicts show occupancy_rate using hierarchical historical mean occupancy
learned strictly from the training data:
    1. (movie_id, weekday, slot)
    2. (movie_id, slot)
    3. (genre, slot)
    4. slot
    5. global training mean

Predicted occupancy is multiplied by screen capacity and clipped to [0, capacity].

Usage:
    python -m src.baseline
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CapacityAdjustedHistoricalBaseline:
    """Hierarchical lookup baseline for capacity-adjusted show demand."""

    global_mean_: float = 0.27
    slot_means_: dict[str, float] = field(default_factory=dict)
    genre_slot_means_: dict[tuple[str, str], float] = field(default_factory=dict)
    movie_slot_means_: dict[tuple[str, str], float] = field(default_factory=dict)
    movie_wd_slot_means_: dict[tuple[str, int, str], float] = field(default_factory=dict)

    def fit(self, train_df: pd.DataFrame) -> "CapacityAdjustedHistoricalBaseline":
        """Fit hierarchical occupancy averages on training shows only."""
        self.global_mean_ = float(train_df["occupancy_rate"].mean())
        self.slot_means_ = {
            str(k): float(v)
            for k, v in train_df.groupby("slot")["occupancy_rate"].mean().items()
        }
        self.genre_slot_means_ = {
            (str(g), str(s)): float(v)
            for (g, s), v in train_df.groupby(["genre", "slot"])["occupancy_rate"].mean().items()
        }
        self.movie_slot_means_ = {
            (str(m), str(s)): float(v)
            for (m, s), v in train_df.groupby(["movie_id", "slot"])["occupancy_rate"].mean().items()
        }
        self.movie_wd_slot_means_ = {
            (str(m), int(w), str(s)): float(v)
            for (m, w, s), v in train_df.groupby(["movie_id", "weekday", "slot"])[
                "occupancy_rate"
            ]
            .mean()
            .items()
        }
        return self

    def predict_occupancy(self, df: pd.DataFrame) -> np.ndarray:
        """Predict clipped occupancy_rate in [0, 1] using hierarchical fallbacks."""
        preds = np.empty(len(df), dtype=float)
        for i, row in enumerate(df.itertuples(index=False)):
            key_mws = (str(row.movie_id), int(row.weekday), str(row.slot))
            key_ms = (str(row.movie_id), str(row.slot))
            key_gs = (str(row.genre), str(row.slot))
            key_s = str(row.slot)

            if key_mws in self.movie_wd_slot_means_:
                occ = self.movie_wd_slot_means_[key_mws]
            elif key_ms in self.movie_slot_means_:
                occ = self.movie_slot_means_[key_ms]
            elif key_gs in self.genre_slot_means_:
                occ = self.genre_slot_means_[key_gs]
            elif key_s in self.slot_means_:
                occ = self.slot_means_[key_s]
            else:
                occ = self.global_mean_

            preds[i] = float(np.clip(occ, 0.0, 1.0))
        return preds

    def predict_tickets(self, df: pd.DataFrame) -> np.ndarray:
        """Convert predicted occupancy into capacity-clipped ticket counts."""
        occ = self.predict_occupancy(df)
        cap = df["capacity"].to_numpy(dtype=float)
        return np.clip(occ * cap, 0.0, cap)


def main() -> None:
    from src.config import PROJECT_ROOT

    proc = PROJECT_ROOT / "data" / "processed"
    train_df = pd.read_parquet(proc / "train.parquet")
    val_df = pd.read_parquet(proc / "val.parquet")

    baseline = CapacityAdjustedHistoricalBaseline().fit(train_df)
    pred_occ = baseline.predict_occupancy(val_df)
    pred_tix = baseline.predict_tickets(val_df)

    mae_tix = float(np.mean(np.abs(val_df["tickets_sold"].to_numpy() - pred_tix)))
    mae_occ = float(np.mean(np.abs(val_df["occupancy_rate"].to_numpy() - pred_occ)) * 100)
    print(f"Baseline (Val) -> MAE: {mae_tix:.2f} tickets | {mae_occ:.2f} occ pts")


if __name__ == "__main__":
    main()

