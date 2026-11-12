"""Global and local model explainability (Permutation Importance + SHAP).

Generates:
- reports/figures/explain_01_permutation_importance.png
- reports/figures/explain_02_shap_summary.png
- reports/explanations.json (global top features + 1 accurate show + 1 large error)

Usage:
    python -m src.explain
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
import shap
from sklearn.inspection import permutation_importance

from src.config import PROJECT_ROOT, load_config
from src.evaluate import clip_predictions


def _get_transformed_feature_names(pre) -> list[str]:
    names: list[str] = []
    for name, trans, cols in pre.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            names.extend(trans.get_feature_names_out(cols).tolist())
        elif hasattr(trans, "named_steps") and "ohe" in trans.named_steps:
            names.extend(trans.named_steps["ohe"].get_feature_names_out(cols).tolist())
        else:
            names.extend(list(cols))
    return names


def _collapse_shap_to_raw_features(
    shap_row: np.ndarray,
    transformed_names: list[str],
    raw_features: list[str],
    categorical_features: list[str],
) -> list[dict[str, Any]]:
    """Aggregate one-hot SHAP contributions back to original input column names."""
    grouped: dict[str, float] = {f: 0.0 for f in raw_features}
    for val, tname in zip(shap_row, transformed_names):
        matched = None
        for cat in categorical_features:
            prefix = f"{cat}_"
            if tname.startswith(prefix):
                matched = cat
                break
        if matched is None:
            matched = tname
        if matched in grouped:
            grouped[matched] += float(val)

    ranked = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [{"feature": k, "shap_occ_impact": round(v, 4)} for k, v in ranked[:8]]


def run_explainability(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute permutation importance, SHAP summary, and local case explanations."""
    cfg = config or load_config()
    seed = int(cfg["project"]["seed"])
    figures_dir = PROJECT_ROOT / cfg["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(PROJECT_ROOT / cfg["paths"]["model_bundle"])
    best_pipe = bundle["best_model"]
    best_name = bundle["best_model_name"]
    feat_cols: list[str] = bundle["feature_columns"]
    cat_cols: list[str] = bundle["categorical_features"]

    test_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "test.parquet")
    X_test = test_df[feat_cols]
    y_test_occ = test_df["occupancy_rate"].to_numpy(dtype=float)
    cap_test = test_df["capacity"].to_numpy(dtype=float)

    # 1. Global Permutation Importance (at original raw-feature level)
    perm = permutation_importance(
        best_pipe,
        X_test,
        y_test_occ,
        n_repeats=8,
        random_state=seed,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    perm_df = pd.DataFrame(
        {
            "feature": feat_cols,
            "importance_occ_mae": perm.importances_mean * 100.0,
            "std_occ_mae": perm.importances_std * 100.0,
        }
    ).sort_values("importance_occ_mae", ascending=False)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    top_perm = perm_df.head(15)
    sns.barplot(data=top_perm, x="importance_occ_mae", y="feature", ax=ax, color="#2b5c8f")
    ax.set_title(f"{best_name}: Top 15 Features by Permutation Importance (Test)")
    ax.set_xlabel("MAE Increase When Shuffled (Occupancy Percentage Points)")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(figures_dir / "explain_01_permutation_importance.png", dpi=150)
    plt.close(fig)

    # 2. SHAP Global & Local Explanations
    pre = best_pipe.named_steps["pre"]
    reg = best_pipe.named_steps["reg"]
    X_test_trans = pre.transform(X_test)
    trans_names = _get_transformed_feature_names(pre)

    sample_n = min(600, len(test_df))
    rng = np.random.default_rng(seed)
    shap_idx = rng.choice(len(test_df), size=sample_n, replace=False)
    X_shap_sample = X_test_trans[shap_idx]

    if best_name in ("HistGradientBoosting", "RandomForest"):
        explainer = shap.TreeExplainer(reg)
        shap_vals_sample = explainer.shap_values(X_shap_sample)
    else:
        explainer = shap.LinearExplainer(reg, X_shap_sample)
        shap_vals_sample = explainer.shap_values(X_shap_sample)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_vals_sample,
        X_shap_sample,
        feature_names=trans_names,
        max_display=15,
        show=False,
    )
    plt.title(f"{best_name}: SHAP Summary (Top 15 Encoded Inputs)")
    plt.tight_layout()
    plt.savefig(figures_dir / "explain_02_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Local explanations: 1 accurate prediction + 1 large error
    raw_occ = best_pipe.predict(X_test)
    pred_occ, pred_tix, _ = clip_predictions(raw_occ, cap_test)
    q10_raw = bundle["q10_model"].predict(X_test) - float(bundle["cqr_delta"])
    q90_raw = bundle["q90_model"].predict(X_test) + float(bundle["cqr_delta"])
    low_tix = np.clip(np.minimum(q10_raw, pred_occ) * cap_test, 0.0, cap_test)
    high_tix = np.clip(np.maximum(q90_raw, pred_occ) * cap_test, 0.0, cap_test)

    abs_err = np.abs(test_df["tickets_sold"].to_numpy(dtype=float) - pred_tix)

    # Pick 1 representative accurate evening/weekend prediction (moderate-high volume, abs_err <= 3)
    good_candidates = np.where((abs_err <= 3.0) & (test_df["tickets_sold"].to_numpy() >= 50))[0]
    good_idx = int(good_candidates[0]) if len(good_candidates) > 0 else int(np.argmin(abs_err))

    # Pick the largest error show on test
    bad_idx = int(np.argmax(abs_err))

    def _explain_single(row_idx: int, label: str) -> dict[str, Any]:
        row = test_df.iloc[row_idx]
        x_row_trans = pre.transform(X_test.iloc[[row_idx]])
        sv = explainer.shap_values(x_row_trans)[0]
        top_drivers = _collapse_shap_to_raw_features(sv, trans_names, feat_cols, cat_cols)

        cap_val = int(row["capacity"])
        p_tix = round(float(pred_tix[row_idx]), 1)
        a_tix = int(row["tickets_sold"])
        err_val = round(float(a_tix - p_tix), 1)
        lo = round(float(low_tix[row_idx]), 1)
        hi = round(float(high_tix[row_idx]), 1)

        top3_str = ", ".join(
            f"{d['feature']} ({d['shap_occ_impact']:+.3f} occ)" for d in top_drivers[:3]
        )
        if label == "accurate_prediction":
            narrative = (
                f"For {row['movie_title']} ({row['genre']}, age {int(row['movie_age_days'])}d) in "
                f"{row['screen_id']} ({row['screen_type']}, cap {cap_val}) during {row['slot']} on "
                f"{row['show_date']}, the model predicted {p_tix:.0f} tickets (80% interval: {lo:.0f}-{hi:.0f}) "
                f"and actual sales were {a_tix} (error {err_val:+.1f}). Key drivers: {top3_str}. "
                f"Limitation: point forecast assumes normal weather/external competition and T-48h pricing remains fixed."
            )
        else:
            cold_note = "new/low-history release" if int(row["is_new_movie"]) == 1 or int(row["is_low_history"]) == 1 else "established title"
            event_note = "public holiday / promo event" if int(row["is_holiday"]) == 1 or int(row["is_special_event"]) == 1 else "standard calendar day"
            narrative = (
                f"Largest error ({err_val:+.1f} tickets): {row['movie_title']} ({row['genre']}, {cold_note}, "
                f"age {int(row['movie_age_days'])}d, {int(row['shows_so_far_movie'])} prior shows) on {row['show_date']} "
                f"({event_note}, {row['slot']}) in {row['screen_id']} (cap {cap_val}). Model predicted {p_tix:.0f} "
                f"(80% interval: {lo:.0f}-{hi:.0f}) vs actual {a_tix} (sold_out={int(row['sold_out'])}). "
                f"Top drivers: {top3_str}. Why it missed: at T-48h, unobserved movie-specific buzz/random demand shock "
                f"(or sparse cold-start/holiday interaction) pushed attendance well outside the expected mean."
            )

        return {
            "case": label,
            "show_id": str(row["show_id"]),
            "show_date": str(row["show_date"]),
            "slot": str(row["slot"]),
            "screen_id": str(row["screen_id"]),
            "screen_type": str(row["screen_type"]),
            "capacity": cap_val,
            "movie_id": str(row["movie_id"]),
            "genre": str(row["genre"]),
            "movie_age_days": int(row["movie_age_days"]),
            "shows_so_far_movie": int(row["shows_so_far_movie"]),
            "is_holiday": int(row["is_holiday"]),
            "is_special_event": int(row["is_special_event"]),
            "actual_tickets": a_tix,
            "predicted_tickets": p_tix,
            "residual_tickets": err_val,
            "interval_80_tickets": [lo, hi],
            "top_drivers": top_drivers,
            "plain_language_explanation": narrative,
        }

    explanations = {
        "best_model_name": best_name,
        "global_permutation_importance": perm_df.round(4).to_dict(orient="records"),
        "local_accurate_case": _explain_single(good_idx, "accurate_prediction"),
        "local_large_error_case": _explain_single(bad_idx, "large_error"),
    }

    out_path = PROJECT_ROOT / "reports" / "explanations.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(explanations, fh, indent=2)

    return explanations


def main() -> None:
    exp = run_explainability()
    print("=== Top 10 Global Features (Permutation Importance, Test) ===")
    print(pd.DataFrame(exp["global_permutation_importance"]).head(10).to_string(index=False))
    print("\n=== Local Accurate Case ===")
    print(exp["local_accurate_case"]["plain_language_explanation"])
    print("\n=== Local Large Error Case ===")
    print(exp["local_large_error_case"]["plain_language_explanation"])


if __name__ == "__main__":
    main()

