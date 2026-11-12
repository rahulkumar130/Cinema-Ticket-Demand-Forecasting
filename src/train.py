"""Reproducible training pipeline, validation hyperparameter search, and artifact saving.

Workflow:
1. Load chronological `train.parquet` and `val.parquet` (test set is NEVER read here).
2. Fit Baseline (`CapacityAdjustedHistoricalBaseline`) on train and score on val.
3. Run a small, documented grid search on validation across 3 model families:
   - Model A: Ridge Regression (linear, interpretable)
   - Model B: Random Forest Regressor
   - Model C: HistGradientBoostingRegressor
4. Compute 4-fold rolling-origin TimeSeriesSplit CV MAE on train for each family's best config.
5. Log all experiment runs to `reports/experiments.csv`.
6. Select winner by lowest validation MAE (tickets), train 10th/90th quantile models
   on train, and calibrate 80% prediction intervals via Conformalized Quantile Regression (CQR)
   on validation.
7. Save `models/model_bundle.joblib` and `models/model_metadata.json`.

Usage:
    python -m src.train
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.baseline import CapacityAdjustedHistoricalBaseline
from src.config import PROJECT_ROOT, load_config, set_global_seed
from src.evaluate import compute_core_metrics
from src.features import ALL_FEATURE_COLUMNS, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.split import get_rolling_origin_cv


def build_preprocessor() -> ColumnTransformer:
    """Create train-only ColumnTransformer for numeric & categorical features."""
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _rolling_cv_mae_tickets(
    pipe_factory,
    train_df: pd.DataFrame,
    n_splits: int = 4,
) -> float:
    """Compute expanding-window TimeSeriesSplit CV MAE (tickets) on train_df."""
    cv = get_rolling_origin_cv(n_splits=n_splits)
    X_tr_all = train_df[ALL_FEATURE_COLUMNS]
    y_tr_all = train_df["occupancy_rate"].to_numpy(dtype=float)
    tix_all = train_df["tickets_sold"].to_numpy(dtype=float)
    cap_all = train_df["capacity"].to_numpy(dtype=float)

    fold_maes = []
    for tr_idx, va_idx in cv.split(X_tr_all):
        model = pipe_factory()
        model.fit(X_tr_all.iloc[tr_idx], y_tr_all[tr_idx])
        raw_occ = model.predict(X_tr_all.iloc[va_idx])
        pred_tix = np.clip(np.clip(raw_occ, 0.0, 1.0) * cap_all[va_idx], 0.0, cap_all[va_idx])
        fold_maes.append(float(np.mean(np.abs(tix_all[va_idx] - pred_tix))))
    return round(float(np.mean(fold_maes)), 4)


def _extract_reference_priors(train_df: pd.DataFrame) -> dict[str, Any]:
    """Extract lightweight lookup priors from training data for API/UI cold-start fallback."""
    genre_slot = {
        f"{g}|{s}": round(float(v), 4)
        for (g, s), v in train_df.groupby(["genre", "slot"])["occupancy_rate"].mean().items()
    }
    genre_mean = {
        str(g): round(float(v), 4)
        for g, v in train_df.groupby("genre")["occupancy_rate"].mean().items()
    }
    slot_wd = {
        f"{s}|{int(w)}": round(float(v), 4)
        for (s, w), v in train_df.groupby(["slot", "weekday"])["occupancy_rate"].mean().items()
    }
    screen_caps = {
        str(sid): {"screen_type": str(grp["screen_type"].iloc[0]), "capacity": int(grp["capacity"].iloc[0])}
        for sid, grp in train_df.groupby("screen_id")
    }
    movie_summary = {}
    for mid, grp in train_df.groupby("movie_id"):
        tail = grp.sort_values("show_datetime").tail(10)
        movie_summary[str(mid)] = {
            "movie_title": str(grp["movie_title"].iloc[-1]),
            "genre": str(grp["genre"].iloc[-1]),
            "language": str(grp["language"].iloc[-1]),
            "rating": str(grp["rating"].iloc[-1]),
            "runtime_min": float(grp["runtime_min"].iloc[-1]),
            "occ_mean_recent": round(float(tail["occupancy_rate"].mean()), 4),
            "occ_mean_all": round(float(grp["occupancy_rate"].mean()), 4),
            "sold_out_rate": round(float(grp["sold_out"].mean()), 4),
            "total_train_shows": int(len(grp)),
        }

    return {
        "global_occ_mean": round(float(train_df["occupancy_rate"].mean()), 4),
        "genre_slot_occ": genre_slot,
        "genre_occ": genre_mean,
        "slot_weekday_occ": slot_wd,
        "screens": screen_caps,
        "movies": movie_summary,
    }


def run_training(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Train baseline + candidate models, tune on val, calibrate CQR intervals, and save artifact."""
    cfg = config or load_config()
    seed = int(cfg["project"]["seed"])
    set_global_seed(seed)

    ev = cfg["evaluation"]
    proc_dir = PROJECT_ROOT / "data" / "processed"
    train_df = pd.read_parquet(proc_dir / "train.parquet")
    val_df = pd.read_parquet(proc_dir / "val.parquet")

    X_train = train_df[ALL_FEATURE_COLUMNS]
    y_train_occ = train_df["occupancy_rate"].to_numpy(dtype=float)

    X_val = val_df[ALL_FEATURE_COLUMNS]
    y_val_tix = val_df["tickets_sold"].to_numpy(dtype=float)
    y_val_occ = val_df["occupancy_rate"].to_numpy(dtype=float)
    cap_val = val_df["capacity"].to_numpy(dtype=float)

    experiments: list[dict[str, Any]] = []

    # 1. Baseline
    baseline = CapacityAdjustedHistoricalBaseline().fit(train_df)
    base_val_occ = baseline.predict_occupancy(val_df)
    base_val_metrics = compute_core_metrics(
        y_true_tix=y_val_tix,
        raw_pred_occ=base_val_occ,
        y_true_occ=y_val_occ,
        capacity=cap_val,
        under_cost=float(ev["under_forecast_cost_per_ticket"]),
        over_cost=float(ev["over_forecast_cost_per_ticket"]),
        occ_tolerance=float(ev["occupancy_tolerance"]),
    )
    experiments.append(
        {
            "model_family": "Baseline",
            "params": "{}",
            "cv_mae_tickets": np.nan,
            **{f"val_{k}": v for k, v in base_val_metrics.items()},
        }
    )

    # 2. Candidate model grids (tuned strictly on val_df)
    search_space: dict[str, list[tuple[dict[str, Any], Any]]] = {
        "Ridge": [
            ({"alpha": alpha}, lambda a=alpha: Pipeline([("pre", build_preprocessor()), ("reg", Ridge(alpha=a))]))
            for alpha in [0.5, 5.0, 25.0]
        ],
        "RandomForest": [
            (
                {"n_estimators": 150, "max_depth": md, "min_samples_leaf": msl, "random_state": seed},
                lambda d=md, l=msl: Pipeline(
                    [
                        ("pre", build_preprocessor()),
                        (
                            "reg",
                            RandomForestRegressor(
                                n_estimators=150,
                                max_depth=d,
                                min_samples_leaf=l,
                                n_jobs=-1,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            )
            for md in [10, 14]
            for msl in [5, 15]
        ],
        "HistGradientBoosting": [
            (
                {"max_iter": 250, "max_depth": md, "learning_rate": lr, "min_samples_leaf": msl, "random_state": seed},
                lambda d=md, rate=lr, l=msl: Pipeline(
                    [
                        ("pre", build_preprocessor()),
                        (
                            "reg",
                            HistGradientBoostingRegressor(
                                max_iter=250,
                                max_depth=d,
                                learning_rate=rate,
                                min_samples_leaf=l,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            )
            for md in [4, 6]
            for lr in [0.05, 0.08]
            for msl in [20, 40]
        ],
    }

    best_per_family: dict[str, tuple[float, dict[str, Any], Pipeline, Any]] = {}

    for family, configs in search_space.items():
        for params, factory in configs:
            pipe = factory()
            pipe.fit(X_train, y_train_occ)
            val_raw_occ = pipe.predict(X_val)
            m = compute_core_metrics(
                y_true_tix=y_val_tix,
                raw_pred_occ=val_raw_occ,
                y_true_occ=y_val_occ,
                capacity=cap_val,
                under_cost=float(ev["under_forecast_cost_per_ticket"]),
                over_cost=float(ev["over_forecast_cost_per_ticket"]),
                occ_tolerance=float(ev["occupancy_tolerance"]),
            )
            experiments.append(
                {
                    "model_family": family,
                    "params": json.dumps(params),
                    "cv_mae_tickets": np.nan,
                    **{f"val_{k}": v for k, v in m.items()},
                }
            )
            if family not in best_per_family or m["mae_tickets"] < best_per_family[family][0]:
                best_per_family[family] = (m["mae_tickets"], params, pipe, factory)

    # Compute TimeSeriesSplit CV MAE on train for the winning config of each family
    for family, (_, params, _, factory) in best_per_family.items():
        cv_mae = _rolling_cv_mae_tickets(factory, train_df, n_splits=4)
        for exp in experiments:
            if exp["model_family"] == family and exp["params"] == json.dumps(params):
                exp["cv_mae_tickets"] = cv_mae

    exp_df = pd.DataFrame(experiments)
    exp_path = PROJECT_ROOT / cfg["paths"]["experiment_log"]
    exp_path.parent.mkdir(parents=True, exist_ok=True)
    exp_df.to_csv(exp_path, index=False)

    # Select overall winner by validation MAE (tickets)
    best_model_name = min(best_per_family.keys(), key=lambda f: best_per_family[f][0])
    best_val_mae, best_params, best_pipe, _ = best_per_family[best_model_name]

    # 3. Quantile regression (10th / 90th) + Conformalized Quantile Regression (CQR) calibration on val
    q10_pipe = Pipeline(
        [
            ("pre", build_preprocessor()),
            (
                "reg",
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=0.10,
                    max_iter=250,
                    max_depth=5,
                    learning_rate=0.05,
                    min_samples_leaf=30,
                    random_state=seed,
                ),
            ),
        ]
    )
    q90_pipe = Pipeline(
        [
            ("pre", build_preprocessor()),
            (
                "reg",
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=0.90,
                    max_iter=250,
                    max_depth=5,
                    learning_rate=0.05,
                    min_samples_leaf=30,
                    random_state=seed,
                ),
            ),
        ]
    )
    q10_pipe.fit(X_train, y_train_occ)
    q90_pipe.fit(X_train, y_train_occ)

    q10_val = q10_pipe.predict(X_val)
    q90_val = q90_pipe.predict(X_val)
    conformity_scores = np.maximum(q10_val - y_val_occ, y_val_occ - q90_val)
    n_val = len(y_val_occ)
    q_level = min(1.0, 0.80 * (1.0 + 1.0 / n_val))
    cqr_delta = float(np.quantile(conformity_scores, q_level))

    # 4. Save model bundle & metadata
    candidate_models = {family: best_per_family[family][2] for family in best_per_family}
    bundle = {
        "best_model_name": best_model_name,
        "best_model": best_pipe,
        "candidate_models": candidate_models,
        "baseline": baseline,
        "q10_model": q10_pipe,
        "q90_model": q90_pipe,
        "cqr_delta": cqr_delta,
        "feature_columns": ALL_FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "reference_priors": _extract_reference_priors(train_df),
    }

    bundle_path = PROJECT_ROOT / cfg["paths"]["model_bundle"]
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "horizon_hours": int(cfg["project"]["horizon_hours"]),
        "split_dates": cfg["split"],
        "best_model_name": best_model_name,
        "best_params": best_params,
        "best_val_mae_tickets": best_val_mae,
        "cqr_delta_occ": round(cqr_delta, 5),
        "feature_columns": ALL_FEATURE_COLUMNS,
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    meta_path = PROJECT_ROOT / cfg["paths"]["model_metadata"]
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return {
        "experiments": exp_df,
        "best_model_name": best_model_name,
        "best_val_mae_tickets": best_val_mae,
        "cqr_delta": cqr_delta,
        "bundle_path": str(bundle_path),
        "metadata_path": str(meta_path),
    }


def main() -> None:
    res = run_training()
    print("=== Validation Experiment Summary ===")
    cols = ["model_family", "params", "cv_mae_tickets", "val_mae_tickets", "val_rmse_tickets", "val_wape_pct", "val_mae_occ_pts"]
    print(res["experiments"][cols].sort_values("val_mae_tickets").to_string(index=False))
    print(f"\nSelected best model: {res['best_model_name']} (Val MAE = {res['best_val_mae_tickets']:.3f} tickets)")
    print(f"CQR 80% calibration delta: {res['cqr_delta']:+.4f} occupancy")
    print(f"Saved model bundle -> {res['bundle_path']}")
    print(f"Saved metadata     -> {res['metadata_path']}")


if __name__ == "__main__":
    main()

