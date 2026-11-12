# Submission Form Pack (Draft Answers)

> Fill in the bracketed personal/GitHub fields before final submission.

---

## 1. Candidate & Project Details
- **Candidate Name:** `[YOUR NAME]`
- **Registration No:** `[REG NO]`
- **University Email:** `[EMAIL]`
- **Assessment Role:** AI & ML Intern — Algions AI Labs
- **Project Title:** Cinema Ticket Demand Forecasting (`T-48h`)
- **Assignment Confirmation:** Completed the single assigned project (`Cinema Ticket Demand Forecasting`) end-to-end with reproducible data generation, `T-48h` leakage-audited features, chronological train/val/test evaluation, FastAPI + Streamlit inference interface, tests, and documentation.
- **Repository URL:** `[GITHUB_REPO_URL]`
- **Branch & Final Commit ID:** `[main]` / `[FINAL_COMMIT_HASH]`
- **Demo Video Link (3–5 min):** `[VIDEO_URL]`
- **Live / Local Demo URLs:** FastAPI (`http://127.0.0.1:8000/docs`), Streamlit (`http://localhost:8501`)

## 2. Role-Specific Fields
- **Dataset Source & Licence:** Self-generated synthetic show-level cinema dataset (`data/generate_data.py`, `seed=42`, `21,708` raw / `21,568` clean shows over `540` days, `65` movies, `8` screens); Licence: Project-internal synthetic (`MIT`), documented in [docs/data_card.md](data_card.md). Zero PII.
- **Notebooks & Report Links:**
  - EDA: [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb)
  - Leakage Audit: [notebooks/02_leakage_audit.ipynb](../notebooks/02_leakage_audit.ipynb)
  - Modeling & Evaluation: [notebooks/03_modeling_and_evaluation.ipynb](../notebooks/03_modeling_and_evaluation.ipynb)
  - Technical Report: [reports/technical_report.md](../reports/technical_report.md)
  - Model Card & Architecture: [docs/model_card.md](model_card.md), [docs/architecture.md](architecture.md)
- **Model Artifact Location:** `models/model_bundle.joblib` and `models/model_metadata.json`
- **Untouched Test Set Metrics (`3,234` shows, `2025-04-04` to `2025-06-23`):**
  - **Baseline (Capacity-Adjusted Hist Avg):** `MAE = 26.18` tickets | `RMSE = 37.16` | `WAPE = 60.86%` | `MAE_occ = 15.14` pts
  - **Selected Model (`HistGradientBoosting`):** `MAE = 9.93` tickets (`-62.05%` vs baseline) | `RMSE = 13.79` | `WAPE = 23.09%` | `MAE_occ = 5.82` pts | `±10 occ pts = 83.52%` | `80% CQR coverage = 82.59%`
  - **Cold-Start (`new_movie`, `n=226`):** `MAE = 16.11` tickets | `RMSE = 21.34` | `WAPE = 15.16%`
- **Inference API & App Entrypoints:** `api/main.py` (`make api`) and `app/streamlit_app.py` (`make app`)

## 3. Technology Stack & Setup Notes
- **Stack:** Python `3.11.15`, `numpy`, `pandas`, `pyarrow`, `scikit-learn==1.6.1`, `scipy==1.14.1`, `pandera`, `shap`, `joblib`, `matplotlib`, `seaborn`, `fastapi`, `uvicorn`, `pydantic`, `streamlit`, `pytest`.
- **Setup Notes:** Run `make setup && make data && make train && make evaluate && make test` from repo root. Pinning `scipy==1.14.1` resolves a macOS ARM64 `dlopen` bug in `scipy 1.15.3`; using `HistGradientBoostingRegressor` avoids Homebrew `libomp` requirements for `lightgbm`.

## 4. Known Incomplete Features / Assumptions & AI Tool Disclosure
- **Assumptions & Limitations:** Synthetic-data caveat (models reflect simulator rules, not real cinema POS dynamics); right-censored `tickets_sold` hides overflow demand on `sold_out` shows (`1.34%` of clean data; point sold-out recall `0.318`, upper-90th recall `0.364`); no advance pre-sale velocity feature (`tickets_booked_at_T48h`).
- **AI Tools Used & Purpose:** Used Antigravity AI coding assistant for structured scaffolding, boilerplate generation, debugging macOS wheel compatibility, and drafting documentation; all code, splits, leakage tests, and metric outputs were executed, verified, and reviewed by the candidate for full live walkthrough readiness.
- **Dataset & Library Attribution:** Self-generated synthetic data (`data/generate_data.py`); open-source Python ecosystem (`scikit-learn`, `pandas`, `pandera`, `shap`, `fastapi`, `streamlit`, `pytest`).

## 5. Declarations
- `[x]` **Originality:** All implementation, experiments, and documentation were produced specifically for this assessment.
- `[x]` **Code Walkthrough Readiness:** I understand and can explain or modify every module in `data/`, `src/`, `api/`, `app/`, and `tests/`.
- `[x]` **Privacy & Secrets Hygiene:** No real personal data, API keys, or secrets are committed (`.env.example` contains placeholders only).
- `[x]` **Working Links & Reproducibility:** All file paths, notebooks, figures, API/app commands, and tests have been verified end-to-end.

