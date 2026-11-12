# System Architecture — Cinema Ticket Demand Forecasting (`T-48h`)

## End-to-End Pipeline & Inference Flow

```mermaid
flowchart TD
    A["data/generate_data.py<br/>Seeded Generator (21,708 raw shows)"] --> B["src/data_prep.py<br/>Dedupe, Outlier Clean, Train-Only Imputation, Pandera"]
    B --> C["src/features.py<br/>30 Strict T-48h Features (< t - 48h searchsorted)"]
    C --> D["src/split.py<br/>Chronological Split (70% Train / 15% Val / 15% Test)"]

    D -->|Train + Val| E["src/train.py<br/>Baseline + Ridge + RF + HistGB + CQR Intervals"]
    E --> F["models/model_bundle.joblib + model_metadata.json"]
    E --> G["reports/experiments.csv"]

    D -->|Frozen Test| H["src/evaluate.py & src/explain.py<br/>Metrics, Cold-Start Slices, SHAP & Error Analysis"]
    F --> H
    H --> I["reports/metrics.json, explanations.json, figures/"]

    F --> J["src/predict.py<br/>Shared Feature Builder + Cold-Start Fallback + Local SHAP"]
    J --> K["api/main.py<br/>FastAPI (/health, /model-info, /predict)"]
    K --> L["app/streamlit_app.py<br/>Decision-Support UI (Direct Fallback if API Offline)"]
    J -.-> L
    L --> M["Cinema Operations Manager<br/>Human Review for Screen & Staff Allocation"]
```

## Key Architectural Guarantees
1. **Zero Training-Serving Skew:** Both FastAPI (`api/main.py`) and Streamlit (`app/streamlit_app.py`) invoke `src.predict.predict_single_show()`, loading the exact `ColumnTransformer` + `HistGradientBoostingRegressor` + CQR quantile models saved in `models/model_bundle.joblib`.
2. **Temporal Isolation:** `src/features.py` enforces strict `< show_datetime - 48h` cutoffs via `np.searchsorted(..., side="left")` before `src/split.py` partitions the timeline into non-overlapping `train`, `val`, and `test` dates.
3. **Human-in-the-Loop Control:** All outputs include uncertainty intervals, cold-start warnings, local SHAP drivers, and an explicit decision-support disclaimer.

