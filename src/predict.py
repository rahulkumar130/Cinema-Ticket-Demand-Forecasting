"""Shared inference module that loads the saved model bundle and scores a show.

Used identically by the FastAPI service (`api/main.py`) and Streamlit fallback
(`app/streamlit_app.py`). Never hard-codes predictions.
"""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import PROJECT_ROOT, load_config

SLOT_TO_HOUR = {"matinee": 14.5, "evening": 18.0, "late": 22.75}
DISCLAIMER_TEXT = (
    "Decision-support forecast only — requires human operations review "
    "before assigning screens or staff."
)


@lru_cache(maxsize=1)
def load_artifacts(
    bundle_path: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Any, list[str]]:
    """Load saved joblib bundle, metadata JSON, and initialize local SHAP explainer once."""
    cfg = load_config()
    b_path = bundle_path or (PROJECT_ROOT / cfg["paths"]["model_bundle"])
    m_path = metadata_path or (PROJECT_ROOT / cfg["paths"]["model_metadata"])

    bundle = joblib.load(b_path)
    with m_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    pipe = bundle["best_model"]
    pre = pipe.named_steps["pre"]
    reg = pipe.named_steps["reg"]
    best_name = bundle["best_model_name"]

    trans_names: list[str] = []
    for name, trans, cols in pre.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            trans_names.extend(trans.get_feature_names_out(cols).tolist())
        elif hasattr(trans, "named_steps") and "ohe" in trans.named_steps:
            trans_names.extend(trans.named_steps["ohe"].get_feature_names_out(cols).tolist())
        else:
            trans_names.extend(list(cols))

    if best_name in ("HistGradientBoosting", "RandomForest"):
        explainer = shap.TreeExplainer(reg)
    else:
        explainer = None

    return bundle, metadata, explainer, trans_names


def _build_feature_row(payload: dict[str, Any], bundle: dict[str, Any]) -> tuple[pd.DataFrame, bool, str | None]:
    """Assemble the exact 30-column DataFrame expected by the saved sklearn Pipeline."""
    priors = bundle["reference_priors"]
    known_movies: dict[str, Any] = priors.get("movies", {})

    show_dt = datetime.strptime(payload["show_date"], "%Y-%m-%d")
    rel_dt = datetime.strptime(payload["release_date"], "%Y-%m-%d")
    age_days = max(0, (show_dt.date() - rel_dt.date()).days)

    weekday = show_dt.weekday()
    month = show_dt.month
    is_weekend = int(weekday >= 5)
    is_tuesday = int(weekday == 1)

    slot = str(payload["slot"]).lower()
    show_hour = float(payload.get("show_hour") or SLOT_TO_HOUR.get(slot, 18.0))

    movie_id = str(payload.get("movie_id") or "UNKNOWN").strip()
    genre = str(payload["genre"])
    shows_so_far = int(payload.get("shows_so_far_movie", 0))
    prior_occ_override = payload.get("prior_movie_occupancy")

    is_known_movie = movie_id in known_movies and shows_so_far > 0

    # Cold-start priors from training reference catalog
    genre_slot_prior = float(priors["genre_slot_occ"].get(f"{genre}|{slot}", priors["global_occ_mean"]))
    genre_prior = float(priors["genre_occ"].get(genre, priors["global_occ_mean"]))
    slot_wd_prior = float(priors["slot_weekday_occ"].get(f"{slot}|{weekday}", priors["global_occ_mean"]))

    if prior_occ_override is not None:
        m_occ_7d = float(prior_occ_override)
        m_occ_14d = float(prior_occ_override)
        m_occ_all = float(prior_occ_override)
        m_so_rate = 0.0
        m_slot_occ = float(prior_occ_override)
        m_wd_occ = float(prior_occ_override)
    elif is_known_movie:
        info = known_movies[movie_id]
        m_occ_7d = float(info["occ_mean_recent"])
        m_occ_14d = float(info["occ_mean_recent"])
        m_occ_all = float(info["occ_mean_all"])
        m_so_rate = float(info["sold_out_rate"])
        m_slot_occ = float(info["occ_mean_recent"])
        m_wd_occ = float(info["occ_mean_recent"])
    else:
        m_occ_7d = np.nan
        m_occ_14d = np.nan
        m_occ_all = np.nan
        m_so_rate = np.nan
        m_slot_occ = np.nan
        m_wd_occ = np.nan

    effective_prior = (
        m_occ_7d
        if not np.isnan(m_occ_7d)
        else (m_occ_all if not np.isnan(m_occ_all) else genre_slot_prior)
    )

    is_new_movie = int(age_days <= 3 or shows_so_far == 0)
    is_low_history = int(shows_so_far <= 5)

    warnings = []
    if movie_id not in known_movies and prior_occ_override is None:
        warnings.append("Unknown movie ID — using genre/slot cold-start priors.")
    if is_new_movie:
        warnings.append(f"New release / zero T-48h history (age={age_days}d, prior_shows={shows_so_far}).")
    elif is_low_history:
        warnings.append(f"Low history depth ({shows_so_far} shows <= 5) — wider uncertainty expected.")

    cold_start_warning = bool(warnings)
    warning_reason = " ".join(warnings) if warnings else None

    row = {
        "slot": slot,
        "screen_id": str(payload.get("screen_id") or "SCR-01"),
        "screen_type": str(payload["screen_type"]),
        "genre": genre,
        "language": str(payload.get("language") or "English"),
        "rating": str(payload.get("rating") or "PG-13"),
        "weekday": weekday,
        "month": month,
        "is_weekend": is_weekend,
        "is_tuesday": is_tuesday,
        "is_holiday": int(bool(payload.get("is_holiday", False))),
        "is_special_event": int(bool(payload.get("is_special_event", False))),
        "show_hour": show_hour,
        "capacity": int(payload["capacity"]),
        "ticket_price": float(payload["ticket_price"]),
        "runtime_min": float(payload.get("runtime_min") or 120.0),
        "movie_age_days": age_days,
        "shows_so_far_movie": shows_so_far,
        "is_new_movie": is_new_movie,
        "is_low_history": is_low_history,
        "movie_occ_mean_all": m_occ_all,
        "movie_occ_mean_7d": m_occ_7d,
        "movie_occ_mean_14d": m_occ_14d,
        "movie_sold_out_rate_all": m_so_rate,
        "movie_slot_occ_mean": m_slot_occ,
        "movie_weekday_occ_mean": m_wd_occ,
        "genre_slot_occ_prior": genre_slot_prior,
        "genre_occ_prior": genre_prior,
        "slot_weekday_occ_prior": slot_wd_prior,
        "movie_effective_occ_prior": effective_prior,
    }
    return pd.DataFrame([row])[bundle["feature_columns"]], cold_start_warning, warning_reason


def predict_single_show(payload: dict[str, Any]) -> dict[str, Any]:
    """Generate forecast, CQR 80% interval, cold-start flag, and SHAP drivers for one show."""
    bundle, metadata, explainer, trans_names = load_artifacts()
    X_row, cold_start_warning, warning_reason = _build_feature_row(payload, bundle)
    capacity = int(payload["capacity"])

    pipe = bundle["best_model"]
    raw_occ = float(pipe.predict(X_row)[0])
    pred_occ = float(np.clip(raw_occ, 0.0, 1.0))
    pred_tix = float(np.clip(pred_occ * capacity, 0.0, capacity))

    cqr_delta = float(bundle["cqr_delta"])
    q10_raw = float(bundle["q10_model"].predict(X_row)[0]) - cqr_delta
    q90_raw = float(bundle["q90_model"].predict(X_row)[0]) + cqr_delta
    low_occ = float(np.clip(min(q10_raw, pred_occ), 0.0, 1.0))
    high_occ = float(np.clip(max(q90_raw, pred_occ), 0.0, 1.0))
    low_tix = float(np.clip(low_occ * capacity, 0.0, capacity))
    high_tix = float(np.clip(high_occ * capacity, 0.0, capacity))

    # Top contributing raw features via SHAP
    top_factors: list[dict[str, Any]] = []
    if explainer is not None:
        x_trans = pipe.named_steps["pre"].transform(X_row)
        sv = explainer.shap_values(x_trans)[0]
        grouped: dict[str, float] = {f: 0.0 for f in bundle["feature_columns"]}
        cat_cols = bundle["categorical_features"]
        for val, tname in zip(sv, trans_names):
            matched = next((cat for cat in cat_cols if tname.startswith(f"{cat}_")), tname)
            if matched in grouped:
                grouped[matched] += float(val)
        ranked = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        top_factors = [
            {
                "feature": k,
                "impact_occ_pts": round(v * 100.0, 2),
                "impact_tickets": round(v * capacity, 1),
                "direction": "up" if v >= 0 else "down",
            }
            for k, v in ranked
        ]

    if high_occ >= 0.95 or pred_occ >= 0.90:
        sold_out_risk = "High"
    elif high_occ >= 0.75:
        sold_out_risk = "Moderate"
    else:
        sold_out_risk = "Low"

    return {
        "model_name": bundle["best_model_name"],
        "predicted_tickets": round(pred_tix, 1),
        "predicted_tickets_rounded": int(round(pred_tix)),
        "predicted_occupancy_pct": round(pred_occ * 100.0, 1),
        "interval_80_tickets": [round(low_tix, 1), round(high_tix, 1)],
        "interval_80_occupancy_pct": [round(low_occ * 100.0, 1), round(high_occ * 100.0, 1)],
        "capacity": capacity,
        "cold_start_warning": cold_start_warning,
        "warning_reason": warning_reason,
        "sold_out_risk": sold_out_risk,
        "top_factors": top_factors,
        "model_version_created_at": metadata.get("created_at_utc"),
        "disclaimer": DISCLAIMER_TEXT,
    }

