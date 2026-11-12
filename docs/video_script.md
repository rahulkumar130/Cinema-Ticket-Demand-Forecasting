# Demo Video Script & Shot List (3–5 Minutes)

**Target Runtime:** 4 minutes 15 seconds  
**Goal:** Walk a technical reviewer through problem framing, `T-48h` anti-leakage guarantees, EDA, model comparison, cold-start behavior, and a **live real prediction** from the saved artifact.

---

## Shot List & Script

### Shot 1 — Problem Framing & Architecture (0:00 – 0:45)
- **Visual:** Show [README.md](../README.md) and [docs/architecture.md](architecture.md) Mermaid diagram.
- **Narration:**  
  *"Hi, I'm `[YOUR NAME]`. This project is a `T-48h` decision-support ML system for Cinema Ticket Demand Forecasting. Two days before every screening, operations managers decide hall sizing and staffing. Our prediction unit is a single show (`movie × screen × date × showtime`), predicting `occupancy_rate` and converting to capacity-clipped `tickets_sold`. Every output supports human review—no automated scheduling."*

### Shot 2 — Synthetic Data & EDA Highlights (0:45 – 1:30)
- **Visual:** Open [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb) and flash [eda_01_distributions.png](../reports/figures/eda_01_distributions.png) and [eda_02_demand_drivers.png](../reports/figures/eda_02_demand_drivers.png).
- **Narration:**  
  *"Using a seeded generator (`21,708` raw shows over 18 months, `65` movies, `8` screens), we injected realistic missingness, duplicates, and POS glitches, then cleaned them with train-only imputation and Pandera validation down to `21,568` clean shows. In EDA, we see clear demand drivers: weekend peaks, Tuesday discount bumps, evening dominance, a steep release-age decay from 60% opening occupancy to 11% after 30 days, and right-censoring at capacity (`1.34%` sold-out shows)."*

### Shot 3 — `T-48h` Leakage Audit (1:30 – 2:15)
- **Visual:** Open [src/features.py](../src/features.py) (`_compute_group_cutoff_stats`) and [notebooks/02_leakage_audit.ipynb](../notebooks/02_leakage_audit.ipynb) perturbation cell output.
- **Narration:**  
  *"At `T-48h`, using `.shift(1)` is a classic bug because yesterday's `T-24h` shows haven't happened yet. Instead, we use `np.searchsorted(..., side='left')` with `cutoff = show_datetime - 48h` so only strictly earlier shows enter any rolling average. In our leakage audit, corrupting all `1,081` shows after `T-48h` produced zero change in target features, and shuffling targets doubled validation MAE from `14.56` to `29.92` tickets."*

### Shot 4 — Model Comparison, CQR Intervals & Cold Start (2:15 – 3:15)
- **Visual:** Show [eval_01_model_comparison.png](../reports/figures/eval_01_model_comparison.png), [eval_03_slice_errors.png](../reports/figures/eval_03_slice_errors.png), and [explain_01_permutation_importance.png](../reports/figures/explain_01_permutation_importance.png).
- **Narration:**  
  *"On the untouched chronological test set (`3,234` shows), `HistGradientBoosting` beats our capacity-adjusted baseline by `62.05%`—cutting MAE from `26.18` down to `9.93` tickets (`23.09%` WAPE) and halving asymmetric operational cost. Notice `Ridge` overshoots `[0, capacity]` on `6.12%` of shows before clipping, whereas tree ensembles stay inside bounds. Our Conformalized Quantile Regression 80% intervals achieve `82.59%` empirical test coverage. On brand-new cold-start movies (`n=226`), MAE is `16.11` tickets vs `59.62` for the baseline."*

### Shot 5 — Live Prediction Demo (3:15 – 4:15)
- **Visual:** Run `make api` and `make app` in terminal, open Streamlit (`http://localhost:8501`), submit **both**:
  1. A **brand-new unknown Action release** (`shows_so_far_movie=0`, `SCR-08` IMAX `302` seats, Saturday evening) -> highlight the **Cold-Start Warning**, **High Sold-Out Risk**, **80% interval**, and **"Why This Forecast?" SHAP bar chart**.
  2. A **mature midweek matinee** -> show lower predicted occupancy, absence of cold-start warning, and negative `movie_age_days` SHAP driver.
- **Visual (Closing):** Run `make test` showing all `11` pytest checks passing.
- **Narration:**  
  *"Everything runs reproducibly from `make data && make train && make evaluate && make test` with fixed seed `42`, and serves real predictions from `models/model_bundle.joblib`. Thanks for watching!"*

