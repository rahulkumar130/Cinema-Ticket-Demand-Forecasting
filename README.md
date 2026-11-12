# Cinema Ticket Demand Forecasting (`T-48h`)

> **Internship Assessment — Algions AI Labs (AI & ML Role)**  
> **Candidate:** `[YOUR NAME]` (`[REG NO]`, `[EMAIL]`) | **OS:** macOS | **Python:** `3.11.15`

Decision-support ML system that forecasts **show-level ticket sales (`tickets_sold`) and seat occupancy (`occupancy_rate`) 48 hours before showtime (`T-48h`)** to help cinema operations teams size halls and schedule staff. Predictions support human review and **never automate high-stakes scheduling decisions**.

---

## Key Results (Untouched Chronological Test Set: 3,234 Shows)

| Model | MAE (Tickets) | RMSE (Tickets) | WAPE (%) | MAE (Occ Pts) | ±10 Occ Pts (%) | Mean Cost / Show |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** (Capacity-Adjusted Hist Avg) | 26.18 | 37.16 | 60.86% | 15.14 | 39.27% | 139.47 |
| **Ridge** (`alpha=25.0`) | 12.66 | 17.95 | 29.43% | 7.36 | 73.41% | 68.91 |
| **Random Forest** (`max_depth=10, leaf=5`) | 10.08 | 14.36 | 23.43% | 5.86 | 83.64% | 59.88 |
| **HistGradientBoosting** *(Selected)* | **9.93** | **13.79** | **23.09%** | **5.82** | 83.52% | **59.60** |

- **Improvement over baseline:** **62.05%** reduction in both test MAE and WAPE.
- **80% CQR Uncertainty Interval:** Empirical test coverage = **82.59%** (mean width `32.00` tickets).
- **Synthetic-data caveat:** Trained and evaluated on a seeded simulation (`21,568` clean shows over `540` days). Results prove pipeline correctness, leakage prevention, and reproducible workflow—not identical accuracy on real multiplex POS data.

---

## Quickstart & Reproducibility

### 1. Environment Setup
```bash
make setup
```
Creates `.venv` using `Python 3.11` and installs pinned dependencies from [requirements.txt](requirements.txt).

### 2. Reproduce Data, Cleaning, Features & Chronological Split
```bash
make data
```
Runs `data.generate_data` (`seed=42`, `540` days, `65` movies, `8` screens) -> `src.data_prep` (dedupe, outlier removal, train-only imputation, Pandera validation) -> `src.features` (`30` strict `T-48h` features).

### 3. Train Models & Calibrate Uncertainty Intervals
```bash
make train
```
Splits chronologically (`70/15/15` via `src.split` or `make train` on existing splits), runs validation tuning (`Ridge`, `RandomForest`, `HistGradientBoosting`), fits 10th/90th quantile models with CQR calibration, logs to [reports/experiments.csv](reports/experiments.csv), and saves [models/model_bundle.joblib](models/model_bundle.joblib) + [models/model_metadata.json](models/model_metadata.json).

### 4. Evaluate on Untouched Test Set & Generate Explainability
```bash
make evaluate
```
Outputs [reports/metrics.json](reports/metrics.json), [reports/explanations.json](reports/explanations.json), and 6 diagnostic charts under `reports/figures/`.

### 5. Run Automated Tests
```bash
make test
```
Runs all `pytest` checks (schema validation, strict `T-48h` anti-leakage perturbation invariance, chronological split, saved artifact integrity, and FastAPI endpoint tests).

### 6. Launch Inference API & Streamlit UI
```bash
# Terminal 1: FastAPI backend (http://127.0.0.1:8000/docs)
make api

# Terminal 2: Streamlit decision-support UI (http://localhost:8501)
make app
```

---

## Repository Structure

```text
cinema-demand-forecasting/
├── config.yaml                  # Seed, paths, T-48h horizon, resolved split dates, cost weights
├── Makefile                     # Single-command targets: setup, data, train, evaluate, test, api, app
├── requirements.txt             # Pinned dependencies (scipy==1.14.1, scikit-learn==1.6.1, etc.)
├── .env.example                 # Placeholder runtime env variables (no secrets)
├── data/
│   ├── generate_data.py         # Seeded synthetic cinema generator (21,708 raw shows)
│   ├── raw/                     # Generated raw CSV
│   └── processed/               # Cleaned parquet, engineered features, train/val/test splits
├── src/
│   ├── config.py                # YAML loader + deterministic seed helper
│   ├── data_prep.py             # Dedupe, outlier cleaning, train-only imputation, Pandera check
│   ├── features.py              # 30 strict T-48h features (< t - 48h searchsorted + prefix sums)
│   ├── split.py                 # Chronological 70/15/15 splitter + TimeSeriesSplit helper
│   ├── baseline.py              # Capacity-adjusted hierarchical historical average baseline
│   ├── train.py                 # Pipelines, val tuning, CQR 80% interval calibration, artifact save
│   ├── evaluate.py              # Test metrics, slice evaluations, business cost, diagnostic plots
│   ├── explain.py               # Permutation importance, SHAP summary, local accurate/error cases
│   └── predict.py               # Shared live inference + cold-start fallback + local SHAP drivers
├── notebooks/
│   ├── 01_eda.ipynb             # Executed EDA with 5 saved charts
│   ├── 02_leakage_audit.ipynb   # Feature availability table, perturbation proof, sanity checks
│   └── 03_modeling_and_evaluation.ipynb  # Tuning summary, test comparison & error slices
├── api/main.py                  # FastAPI (/health, /model-info, /predict) with Pydantic validation
├── app/streamlit_app.py         # Interactive decision-support UI with cold-start & SHAP panels
├── models/                      # Saved joblib bundle + JSON metadata
├── reports/                     # technical_report.md, metrics.json, explanations.json, figures/
├── tests/                       # 5 pytest test modules (11 tests total)
└── docs/                        # data_card.md, model_card.md, architecture.md, video_script.md, viva_prep.md, submission_form_pack.md
```

---

## Known Gaps & Next Steps
- **LightGBM macOS `libomp` dependency:** We use scikit-learn's `HistGradientBoostingRegressor` (mathematically equivalent histogram GBDT with native quantile loss) so reviewers on any OS can run `make setup && make train` without installing Homebrew `libomp`.
- **Sold-out censoring & rare-event recall:** Because only `1.34%` of shows sell out and observed `tickets_sold` is right-censored at `capacity`, point sold-out recall is `0.318` (`0.364` using the 90th-percentile upper interval). Adding pre-sale booking velocity at `T-48h` in a real cinema POS would substantially improve peak/sold-out detection.
- **No real cinema data:** Replace `data/generate_data.py` with historical POS exports + weather/competitor release calendars before production deployment.
