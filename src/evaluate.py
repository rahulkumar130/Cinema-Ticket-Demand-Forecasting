"""Evaluation metrics, slice analysis, and untouched test-set reporting.

Computes ticket-level, occupancy-level, capacity-aware, censoring, cold-start,
asymmetric business cost, and prediction-interval calibration metrics.

Usage:
    python -m src.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, root_mean_squared_error

from src.config import PROJECT_ROOT, load_config


def clip_predictions(
    raw_occ: np.ndarray,
    capacity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Clip raw occupancy predictions to [0, 1] and convert to capacity-clipped tickets.

    Returns:
        clipped_occ, clipped_tickets, pct_outside_bounds_before_clipping
    """
    raw_tix = raw_occ * capacity
    outside = (raw_tix < 0.0) | (raw_tix > capacity)
    outside_pct = float(np.mean(outside) * 100.0)

    clipped_occ = np.clip(raw_occ, 0.0, 1.0)
    clipped_tix = np.clip(clipped_occ * capacity, 0.0, capacity)
    return clipped_occ, clipped_tix, outside_pct


def compute_core_metrics(
    *,
    y_true_tix: np.ndarray,
    raw_pred_occ: np.ndarray,
    y_true_occ: np.ndarray,
    capacity: np.ndarray,
    under_cost: float = 8.0,
    over_cost: float = 3.0,
    occ_tolerance: float = 0.10,
    lower_occ: np.ndarray | None = None,
    upper_occ: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute core ticket, occupancy, capacity-aware, and business-cost metrics."""
    pred_occ, pred_tix, outside_pct = clip_predictions(raw_pred_occ, capacity)

    mae_tix = float(mean_absolute_error(y_true_tix, pred_tix))
    rmse_tix = float(root_mean_squared_error(y_true_tix, pred_tix))
    total_actual = float(np.sum(y_true_tix))
    wape_pct = float(np.sum(np.abs(y_true_tix - pred_tix)) / max(total_actual, 1e-9) * 100.0)
    mae_occ_pts = float(mean_absolute_error(y_true_occ, pred_occ) * 100.0)
    within_tol_pct = float(np.mean(np.abs(y_true_occ - pred_occ) <= occ_tolerance) * 100.0)

    under_err = np.maximum(0.0, y_true_tix - pred_tix)
    over_err = np.maximum(0.0, pred_tix - y_true_tix)
    total_cost = float(np.sum(under_cost * under_err + over_cost * over_err))
    mean_cost = float(total_cost / max(len(y_true_tix), 1))

    # Sold-out detection: actual sold_out = (y_true_tix >= capacity)
    y_so = (y_true_tix >= capacity).astype(int)
    # Flag predicted sold-out if point occupancy >= 0.92 (or upper bound >= 0.98 if available)
    pred_so_point = (pred_occ >= 0.92).astype(int)
    so_prec_point = float(precision_score(y_so, pred_so_point, zero_division=0))
    so_rec_point = float(recall_score(y_so, pred_so_point, zero_division=0))

    metrics: dict[str, float] = {
        "n_shows": int(len(y_true_tix)),
        "mae_tickets": round(mae_tix, 4),
        "rmse_tickets": round(rmse_tix, 4),
        "wape_pct": round(wape_pct, 4),
        "mae_occ_pts": round(mae_occ_pts, 4),
        "within_10_occ_pts_pct": round(within_tol_pct, 2),
        "clip_outside_pct": round(outside_pct, 3),
        "total_business_cost": round(total_cost, 2),
        "mean_business_cost_per_show": round(mean_cost, 2),
        "sold_out_actual_count": int(y_so.sum()),
        "sold_out_precision_point": round(so_prec_point, 4),
        "sold_out_recall_point": round(so_rec_point, 4),
    }

    if lower_occ is not None and upper_occ is not None:
        low_tix = np.clip(lower_occ * capacity, 0.0, capacity)
        high_tix = np.clip(upper_occ * capacity, 0.0, capacity)
        coverage = float(np.mean((y_true_tix >= low_tix) & (y_true_tix <= high_tix)) * 100.0)
        mean_width = float(np.mean(high_tix - low_tix))
        pred_so_upper = (upper_occ >= 0.98).astype(int)
        metrics["interval_80_coverage_pct"] = round(coverage, 2)
        metrics["interval_80_mean_width_tickets"] = round(mean_width, 2)
        metrics["sold_out_precision_upper90"] = round(
            float(precision_score(y_so, pred_so_upper, zero_division=0)), 4
        )
        metrics["sold_out_recall_upper90"] = round(
            float(recall_score(y_so, pred_so_upper, zero_division=0)), 4
        )

    return metrics


def _slice_metrics(
    df: pd.DataFrame,
    raw_pred_occ: np.ndarray,
    mask: pd.Series,
    cfg: dict[str, Any],
) -> dict[str, float]:
    sub = df.loc[mask]
    if sub.empty:
        return {"n_shows": 0, "mae_tickets": np.nan, "rmse_tickets": np.nan, "wape_pct": np.nan, "mae_occ_pts": np.nan}
    ev = cfg["evaluation"]
    return compute_core_metrics(
        y_true_tix=sub["tickets_sold"].to_numpy(dtype=float),
        raw_pred_occ=raw_pred_occ[mask.to_numpy()],
        y_true_occ=sub["occupancy_rate"].to_numpy(dtype=float),
        capacity=sub["capacity"].to_numpy(dtype=float),
        under_cost=float(ev["under_forecast_cost_per_ticket"]),
        over_cost=float(ev["over_forecast_cost_per_ticket"]),
        occ_tolerance=float(ev["occupancy_tolerance"]),
    )


def evaluate_all_slices(
    df: pd.DataFrame,
    raw_pred_occ: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compute cold-start, capacity-bucket, weekday, slot, and genre breakdowns."""
    slices: dict[str, Any] = {"cold_start": {}, "capacity_bucket": {}, "slot": {}, "genre": {}, "weekday": {}}

    # Cold-start & history-depth slices
    new_mask = df["is_new_movie"] == 1
    low_mask = (df["is_new_movie"] == 0) & (df["is_low_history"] == 1)
    mature_mask = (df["is_new_movie"] == 0) & (df["is_low_history"] == 0)
    for label, m in [("new_movie", new_mask), ("low_history", low_mask), ("mature", mature_mask)]:
        slices["cold_start"][label] = _slice_metrics(df, raw_pred_occ, m, cfg)

    # Capacity buckets
    buckets = cfg["evaluation"]["capacity_buckets"]
    cap_labels = ["small_<=120", "medium_121_220", "large_>220"]
    cap_cat = pd.cut(df["capacity"], bins=buckets, labels=cap_labels)
    for lbl in cap_labels:
        slices["capacity_bucket"][lbl] = _slice_metrics(df, raw_pred_occ, cap_cat == lbl, cfg)

    # Slot, Genre, Weekday
    for sl in ["matinee", "evening", "late"]:
        slices["slot"][sl] = _slice_metrics(df, raw_pred_occ, df["slot"] == sl, cfg)

    for g in sorted(df["genre"].unique()):
        slices["genre"][str(g)] = _slice_metrics(df, raw_pred_occ, df["genre"] == g, cfg)

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for wd in range(7):
        slices["weekday"][day_names[wd]] = _slice_metrics(df, raw_pred_occ, df["weekday"] == wd, cfg)

    return slices


def generate_evaluation_plots(
    test_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    best_model_name: str,
    pred_occ: np.ndarray,
    pred_tix: np.ndarray,
    lower_tix: np.ndarray,
    upper_tix: np.ndarray,
    slice_results: dict[str, Any],
    figures_dir: Path,
) -> None:
    """Save standard Phase 4 evaluation plots to reports/figures/."""
    sns.set_theme(style="whitegrid", palette="deep")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=comparison_df, x="model", y="mae_tickets", ax=axes[0], color="#2b5c8f")
    axes[0].set_title("Test MAE (Tickets) — Lower is Better")
    axes[0].tick_params(axis="x", rotation=15)

    sns.barplot(data=comparison_df, x="model", y="wape_pct", ax=axes[1], color="#d95f02")
    axes[1].set_title("Test WAPE (%) — Lower is Better")
    axes[1].tick_params(axis="x", rotation=15)
    plt.tight_layout()
    fig.savefig(figures_dir / "eval_01_model_comparison.png", dpi=150)
    plt.close(fig)

    # 2. Predicted vs Actual & Residuals
    residuals = test_df["tickets_sold"].to_numpy(dtype=float) - pred_tix
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].scatter(test_df["tickets_sold"], pred_tix, alpha=0.22, s=14, color="#1b9e77")
    max_v = float(max(test_df["tickets_sold"].max(), pred_tix.max()))
    axes[0].plot([0, max_v], [0, max_v], "r--", linewidth=1.5, label="Ideal y = x")
    axes[0].set_title(f"{best_model_name}: Predicted vs Actual Tickets (Test)")
    axes[0].set_xlabel("Actual Tickets Sold")
    axes[0].set_ylabel("Predicted Tickets Sold")
    axes[0].legend()

    sns.histplot(residuals, bins=40, kde=True, ax=axes[1], color="#7570b3")
    axes[1].axvline(0.0, color="red", linestyle="--")
    axes[1].set_title("Test Residuals (Actual - Predicted Tickets)")
    axes[1].set_xlabel("Residual (Tickets)")
    plt.tight_layout()
    fig.savefig(figures_dir / "eval_02_residuals_and_pred_vs_actual.png", dpi=150)
    plt.close(fig)

    # 3. Cold-start & Capacity Bucket Slices
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    cs_df = pd.DataFrame(
        [{"slice": k, "mae_tickets": v["mae_tickets"], "wape_pct": v["wape_pct"]} for k, v in slice_results["cold_start"].items()]
    )
    sns.barplot(data=cs_df, x="slice", y="mae_tickets", ax=axes[0], color="#e7298a")
    axes[0].set_title("Test MAE by Cold-Start / History Depth")

    cap_df = pd.DataFrame(
        [{"bucket": k, "mae_tickets": v["mae_tickets"], "wape_pct": v["wape_pct"]} for k, v in slice_results["capacity_bucket"].items()]
    )
    sns.barplot(data=cap_df, x="bucket", y="mae_tickets", ax=axes[1], color="#66a61e")
    axes[1].set_title("Test MAE by Screen Capacity Bucket")
    plt.tight_layout()
    fig.savefig(figures_dir / "eval_03_slice_errors.png", dpi=150)
    plt.close(fig)

    # 4. Interval calibration plot (80 sampled test shows sorted by prediction)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(test_df), size=min(80, len(test_df)), replace=False)
    order = sample_idx[np.argsort(pred_tix[sample_idx])]
    x_pos = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.fill_between(x_pos, lower_tix[order], upper_tix[order], alpha=0.28, color="#386cb0", label="80% CQR Interval")
    ax.plot(x_pos, pred_tix[order], color="#386cb0", linewidth=1.6, label="Point Forecast")
    ax.scatter(x_pos, test_df["tickets_sold"].to_numpy()[order], color="#d95f02", s=22, zorder=5, label="Actual Tickets")
    ax.set_title("80% Prediction Interval Calibration (80 Random Test Shows)")
    ax.set_xlabel("Sampled Shows (Sorted by Forecast)")
    ax.set_ylabel("Tickets Sold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "eval_04_interval_calibration.png", dpi=150)
    plt.close(fig)


def run_evaluation(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate baseline and all trained candidate models once on the untouched test set."""
    cfg = config or load_config()
    ev = cfg["evaluation"]
    proc_dir = PROJECT_ROOT / "data" / "processed"
    test_df = pd.read_parquet(proc_dir / "test.parquet")

    bundle = joblib.load(PROJECT_ROOT / cfg["paths"]["model_bundle"])
    feat_cols: list[str] = bundle["feature_columns"]
    X_test = test_df[feat_cols]
    y_test_tix = test_df["tickets_sold"].to_numpy(dtype=float)
    y_test_occ = test_df["occupancy_rate"].to_numpy(dtype=float)
    cap_test = test_df["capacity"].to_numpy(dtype=float)

    rows = []
    # 1. Baseline
    base_raw_occ = bundle["baseline"].predict_occupancy(test_df)
    base_metrics = compute_core_metrics(
        y_true_tix=y_test_tix,
        raw_pred_occ=base_raw_occ,
        y_true_occ=y_test_occ,
        capacity=cap_test,
        under_cost=float(ev["under_forecast_cost_per_ticket"]),
        over_cost=float(ev["over_forecast_cost_per_ticket"]),
        occ_tolerance=float(ev["occupancy_tolerance"]),
    )
    rows.append({"model": "Baseline (Hist Avg)", **base_metrics})

    # 2. All trained ML models (Ridge, RandomForest, HistGradientBoosting)
    model_preds: dict[str, np.ndarray] = {"Baseline (Hist Avg)": base_raw_occ}
    for name, pipe in bundle["candidate_models"].items():
        raw_occ = pipe.predict(X_test)
        model_preds[name] = raw_occ

        # Attach calibrated 80% CQR interval for the selected winning model
        if name == bundle["best_model_name"]:
            q10_raw = bundle["q10_model"].predict(X_test) - float(bundle["cqr_delta"])
            q90_raw = bundle["q90_model"].predict(X_test) + float(bundle["cqr_delta"])
            low_occ = np.clip(np.minimum(q10_raw, raw_occ), 0.0, 1.0)
            high_occ = np.clip(np.maximum(q90_raw, raw_occ), 0.0, 1.0)
        else:
            low_occ, high_occ = None, None

        m = compute_core_metrics(
            y_true_tix=y_test_tix,
            raw_pred_occ=raw_occ,
            y_true_occ=y_test_occ,
            capacity=cap_test,
            under_cost=float(ev["under_forecast_cost_per_ticket"]),
            over_cost=float(ev["over_forecast_cost_per_ticket"]),
            occ_tolerance=float(ev["occupancy_tolerance"]),
            lower_occ=low_occ,
            upper_occ=high_occ,
        )
        m["improvement_vs_baseline_mae_pct"] = round(
            (base_metrics["mae_tickets"] - m["mae_tickets"]) / base_metrics["mae_tickets"] * 100.0, 2
        )
        m["improvement_vs_baseline_wape_pct"] = round(
            (base_metrics["wape_pct"] - m["wape_pct"]) / base_metrics["wape_pct"] * 100.0, 2
        )
        rows.append({"model": name, **m})

    comp_df = pd.DataFrame(rows)
    best_name = bundle["best_model_name"]
    best_raw_occ = model_preds[best_name]
    best_occ, best_tix, _ = clip_predictions(best_raw_occ, cap_test)

    q10_raw = bundle["q10_model"].predict(X_test) - float(bundle["cqr_delta"])
    q90_raw = bundle["q90_model"].predict(X_test) + float(bundle["cqr_delta"])
    low_occ = np.clip(np.minimum(q10_raw, best_occ), 0.0, 1.0)
    high_occ = np.clip(np.maximum(q90_raw, best_occ), 0.0, 1.0)
    low_tix = np.clip(low_occ * cap_test, 0.0, cap_test)
    high_tix = np.clip(high_occ * cap_test, 0.0, cap_test)

    slice_results = evaluate_all_slices(test_df, best_raw_occ, cfg)
    baseline_slices = evaluate_all_slices(test_df, base_raw_occ, cfg)

    # Worst 20 test predictions for error analysis
    err_df = test_df[
        [
            "show_id",
            "show_date",
            "slot",
            "screen_id",
            "screen_type",
            "capacity",
            "movie_id",
            "genre",
            "movie_age_days",
            "shows_so_far_movie",
            "is_holiday",
            "is_special_event",
            "tickets_sold",
            "sold_out",
        ]
    ].copy()
    err_df["pred_tickets"] = np.round(best_tix, 1)
    err_df["abs_error"] = np.round(np.abs(err_df["tickets_sold"] - best_tix), 1)
    err_df["residual"] = np.round(err_df["tickets_sold"] - best_tix, 1)
    worst_20 = err_df.sort_values("abs_error", ascending=False).head(20).to_dict(orient="records")

    generate_evaluation_plots(
        test_df=test_df,
        comparison_df=comp_df,
        best_model_name=best_name,
        pred_occ=best_occ,
        pred_tix=best_tix,
        lower_tix=low_tix,
        upper_tix=high_tix,
        slice_results=slice_results,
        figures_dir=PROJECT_ROOT / cfg["paths"]["figures_dir"],
    )

    report_payload = {
        "best_model_name": best_name,
        "test_rows": int(len(test_df)),
        "test_date_range": [str(test_df["show_date"].min()), str(test_df["show_date"].max())],
        "model_comparison_test": rows,
        "best_model_slices_test": slice_results,
        "baseline_cold_start_slices_test": baseline_slices["cold_start"],
        "worst_20_shows": worst_20,
    }

    metrics_path = PROJECT_ROOT / cfg["paths"]["metrics_report"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(report_payload, fh, indent=2)

    return report_payload


def main() -> None:
    report = run_evaluation()
    comp_df = pd.DataFrame(report["model_comparison_test"])
    cols = [
        "model",
        "mae_tickets",
        "rmse_tickets",
        "wape_pct",
        "mae_occ_pts",
        "within_10_occ_pts_pct",
        "sold_out_recall_point",
        "mean_business_cost_per_show",
    ]
    print("=== Untouched Test Set Evaluation ===")
    print(comp_df[cols].to_string(index=False))
    print("\n=== Cold-Start & History Slices (Best Model) ===")
    print(pd.DataFrame(report["best_model_slices_test"]["cold_start"]).T[["n_shows", "mae_tickets", "rmse_tickets", "wape_pct"]].to_string())


if __name__ == "__main__":
    main()

