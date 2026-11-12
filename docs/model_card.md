# Model Card — `HistGradientBoosting` T-48h Occupancy Forecaster

## 1. Model Details
- **Model Name / Architecture:** `HistGradientBoostingRegressor` inside a `scikit-learn` `Pipeline` (`ColumnTransformer` with median/mode imputation, standard scaling, and `OneHotEncoder(handle_unknown="ignore")`).
- **Uncertainty Model:** Dual `HistGradientBoostingRegressor(loss="quantile")` models at `q=0.10` and `q=0.90`, calibrated on the validation split via Conformalized Quantile Regression (`cqr_delta = +0.0143` occupancy).
- **Target & Conversion:** Trained on `occupancy_rate = tickets_sold / capacity`; converted at inference via `clip(pred_occ * capacity, 0, capacity)`.
- **Forecast Horizon:** `T-48h` (48 hours prior to scheduled `show_datetime`).
- **Artifact Location:** `models/model_bundle.joblib` and `models/model_metadata.json` (`seed=42`).

## 2. Intended Use & Scope
- **Intended Use:** Operational **decision support** for cinema managers reviewing hall sizing (screen swaps) and floor/concession staffing 2 days before showtime.
- **Out-of-Scope / Prohibited Use:** Automated unattended screen cancellations, dynamic surge pricing without human approval, or direct deployment to real multiplexes without retraining on actual POS history.

## 3. Training & Evaluation Summary
- **Train:** `15,097` shows (`2024-01-01` .. `2025-01-12`)
- **Validation (tuning & CQR calibration):** `3,237` shows (`2025-01-13` .. `2025-04-03`)
- **Test (frozen, evaluated once):** `3,234` shows (`2025-04-04` .. `2025-06-23`)
- **Test Performance:**
  - `MAE = 9.93` tickets (`-62.05%` vs baseline `26.18`)
  - `RMSE = 13.79` tickets | `WAPE = 23.09%` | `MAE (occupancy) = 5.82` percentage points
  - `±10 occupancy points accuracy = 83.52%`
  - `80% CQR interval coverage = 82.59%` (mean width `32.00` tickets)
  - `New-movie slice (n=226): MAE = 16.11` tickets, `WAPE = 15.16%`

## 4. Responsible Use: Representativeness, Bias, Privacy & Failure Modes
1. **Representativeness:** Data is 100% synthetic. It captures weekday/holiday cycles, release decay, slot×genre interactions, price elasticity, and noise, but omits weather shocks, trailer/social buzz, school exam periods, and competing nearby theaters.
2. **Bias & Capacity Censoring:**
   - *Right-censoring bias:* `tickets_sold` is capped at `capacity` (`1.34%` sold-out shows overall, `2.37%` on IMAX in clean data). The model learns *capped attendance*, not *unconstrained willingness to buy*. For screen-upgrade decisions, managers must inspect the **90th-percentile upper interval** (`sold_out_risk`) alongside point forecasts.
   - *Popularity / blockbuster skew:* Blockbusters (~12% of catalog) disproportionately drive sold-out events, while niche genres/late slots have sparse high-occupancy examples.
3. **Privacy:** Zero PII, zero customer IDs, zero payment/demographic attributes. All records are aggregate show-level counts.
4. **Failure Modes:**
   - **Cold-start releases (`shows_so_far_movie == 0`):** Error rises (`MAE 16.11` vs `9.45` for mature shows) because latent quality/buzz is unobserved at `T-48h`.
   - **Rare promo/holiday × mature-film combinations:** Largest test error (`+85.7` tickets) occurred on a 65-day-old movie during a special event in a 302-seat IMAX hall.
   - **Distribution drift:** Post-launch changes in ticket pricing tiers, competitor openings, or theater renovations will degrade calibration.
5. **Monitoring & Retraining Recommendations:**
   - Track weekly rolling **WAPE**, **80% interval empirical coverage** (alert if `< 72%` or `> 88%`), and **feature PSI/drift** (`ticket_price`, `movie_age_days`, `sold_out` rate).
   - Retrain monthly on an expanding or rolling 12-month window and re-calibrate `cqr_delta` on the most recent 60 days.
   - **Governance:** A human operations manager makes every final screen and staffing decision.

