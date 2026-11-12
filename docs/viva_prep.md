# Viva & Technical Walkthrough Prep (30 Questions + 3 Live Exercises)

## Part A — 30 Likely Interview Questions & Concise Answers

1. **What exact business decision does this model support?**  
   Assigning hall capacity (which screen) and floor/concession staffing per scheduled show 48 hours in advance, strictly as human decision support.

2. **What is the prediction unit?**  
   One scheduled show (`movie × screen × date × showtime`), uniquely identified by `show_id` and `(screen_id, show_datetime)`.

3. **Why model `occupancy_rate` instead of raw `tickets_sold` directly?**  
   Capacities vary from `85` to `302` seats. `occupancy_rate = tickets_sold / capacity` normalizes demand onto `[0, 1]`, and multiplying by `capacity` + clipping enforces physical seat limits automatically.

4. **Why is the forecast horizon `T-48h` instead of `T-2h`?**  
   `T-48h` gives operations managers enough lead time to finalize staff rosters and swap screens before showtime, while still using recent movie performance up to 2 days out.

5. **Why did you use a chronological 70/15/15 split instead of random K-Fold?**  
   Random splitting leaks future demand trends, holiday shocks, and future movie runs into training. Chronological splitting (`src/split.py`) mirrors production: train on the past (`2024-01-01`..`2025-01-12`), tune on validation (`2025-01-13`..`2025-04-03`), test once on the future (`2025-04-04`..`2025-06-23`).

6. **Why split by calendar date boundary instead of exact row index?**  
   Splitting by `show_date` guarantees shows on the same day never straddle `train`/`val` or `val`/`test`.

7. **Why WAPE instead of MAPE?**  
   On low-attendance matinees (`2–10` tickets), MAPE explodes toward infinity when actuals are small. `WAPE = sum(|y - y_hat|) / sum(y)` weights errors by ticket volume and stays well-defined.

8. **Why does mature WAPE (`24.80%`) look higher than new-movie WAPE (`15.16%`) even though mature MAE (`9.45`) is lower than new-movie MAE (`16.11`)?**  
   Mature shows (30+ days old) have much smaller average ticket counts in the denominator of WAPE, so a smaller absolute error (`9.45` tickets) produces a larger percentage ratio. Always inspect MAE alongside WAPE.

9. **How did you prevent target/future leakage in lag features?**  
   In `src/features.py` (`_compute_group_cutoff_stats`), every row uses `cutoff = show_datetime - 48h` and `np.searchsorted(timestamps, cutoff, side="left")`, ensuring strictly earlier shows (`< t - 48h`) enter rolling/expanding averages.

10. **Why isn't `.shift(1)` safe here?**  
    With 5 shows per day across 8 screens, `.shift(1)` only skips 1 show—still leaking same-day and `T-24h` shows that have not happened yet at `T-48h`.

11. **How did you verify zero leakage?**  
    Three ways: (1) exact equivalence test against a brute-force `< t - 48h` filter, (2) mutating all targets `>= t - 48h` to `0.99` and asserting `0.0` change in features at `t` (`tests/test_features_no_leakage.py`), and (3) shuffled-target sanity check in `02_leakage_audit.ipynb` (`14.56 → 29.92` MAE).

12. **How did you prevent preprocessing/imputation leakage?**  
    In `src/data_prep.py`, `ImputationStats` medians/modes are fitted strictly on `show_date <= train_end_date`. In `src/train.py`, `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` sit inside a `sklearn.pipeline.Pipeline` fitted only on `train`.

13. **Why did you drop `tickets_sold > capacity` rows instead of clipping them?**  
    During error analysis, clipping `16` injected over-capacity POS glitches to `capacity` created fake `sold_out=1` labels on stale weekday shows (`134.9` ticket error). Dropping corrupted target rows (`< 0` or `> capacity`) aligns with real POS void/error handling.

14. **How does the baseline work?**  
    `CapacityAdjustedHistoricalBaseline` (`src/baseline.py`) builds a hierarchical training-set occupancy lookup: `(movie, weekday, slot) → (movie, slot) → (genre, slot) → slot → global mean`, then multiplies by `capacity`.

15. **Why did `HistGradientBoosting` beat `Ridge` by ~2.7 tickets MAE?**  
    Demand drivers interact nonlinearly (e.g., `Animation × matinee × weekend`, `Horror × late × Fri/Sat`, exponential `movie_age_days` decay), which tree splits capture naturally while linear `Ridge` cannot—and `Ridge` also overshoots `[0, capacity]` on `6.12%` of shows before clipping.

16. **Why did you use `HistGradientBoostingRegressor` instead of `LightGBM`?**  
    `HistGradientBoostingRegressor` is scikit-learn's native LightGBM-inspired histogram GBDT, supports `loss="quantile"` natively, and avoids macOS Homebrew `libomp` binary issues so reviewers can run `make setup && make train` reliably.

17. **How are the 80% uncertainty intervals computed and calibrated?**  
    We train two quantile `HistGradientBoosting` models (`q=0.10`, `q=0.90`) on `train`, then apply **Conformalized Quantile Regression (CQR)** on `val` (`cqr_delta = +0.0143` occ), achieving `82.59%` empirical coverage on the untouched test set.

18. **What is sold-out censoring and why does it matter operationally?**  
    `tickets_sold` is capped at `capacity` (`1.34%` sold-out overall). The model learns *seated attendance*, which works for staffing, but underestimates *unconstrained overflow* when deciding whether to upgrade a movie to a bigger hall.

19. **Why is sold-out recall (`0.318` point / `0.364` at upper90) relatively low?**  
    Only `22` shows (`0.68%`) sold out in the test set, and random show-level noise/unobserved buzz drives unexpected sell-outs. Using `upper90 >= 0.98` trades precision (`0.195`) for higher recall (`0.364`) to warn managers.

20. **How did you define asymmetric business cost?**  
    `8.0` per under-forecasted ticket (lost margin + understaffing friction) vs `3.0` per over-forecasted ticket (wasted staff/hall slack). `HistGradientBoosting` cuts cost from `139.47` (baseline) to `59.60` per show.

21. **How does the pipeline handle cold-start / brand-new movies?**  
    When `shows_so_far_movie == 0` or `movie_id` is unseen, lag features are `NaN`, `is_new_movie = 1`, `movie_effective_occ_prior` falls back to causal `(genre, slot)` / `(slot, weekday)` priors, and `src/predict.py` raises `cold_start_warning=True`.

22. **Why did you also report `TimeSeriesSplit` CV in `train.py`?**  
    To verify hyperparameter stability across multiple expanding chronological windows inside `train` (`10.32` CV MAE vs `10.92` Val MAE for the winning config) rather than relying on a single validation slice.

23. **How are global and local explanations computed?**  
    Global: `permutation_importance` on the test Pipeline + `shap.TreeExplainer` summary plot (`src/explain.py`). Local: `shap.TreeExplainer` aggregated back from one-hot columns to raw features (`src/predict.py`).

24. **What was the most important feature globally, and why?**  
    `movie_age_days` (`+6.69` occupancy points MAE when shuffled), because demand decays steeply after opening weekend (`60%` at `0–3d` down to `11%` at `31+d`).

25. **Why was the largest test error (`+85.7` tickets) on a 65-day-old movie?**  
    `Comedy Feature #18` on `2025-04-14` had `is_special_event=1` in the `302`-seat IMAX hall; the steep 65-day age decay dominated the tree prediction (`20` tickets) relative to the rare special-event boost (`106` actual).

26. **How does the FastAPI app load and validate predictions?**  
    `api/main.py` loads `models/model_bundle.joblib` once in `lifespan`, validates requests via Pydantic (`ShowForecastRequest`, returning HTTP 422 on invalid capacity/price/dates), and calls `src.predict.predict_single_show`.

27. **How does the Streamlit app work if FastAPI isn't running?**  
    It posts to `API_URL/predict` first (`2s` timeout) and cleanly falls back to `src.predict.predict_single_show()` using the exact same `model_bundle.joblib`—zero hard-coded outputs.

28. **Is there any privacy risk in this system?**  
    No—data is synthetic and aggregated strictly at show level (`movie × screen × datetime`), with zero customer IDs or PII.

29. **What is the biggest caveat of these results?**  
    Because data is synthetic (`data/generate_data.py`), metrics prove engineering rigor and leakage control, not real-world cinema accuracy.

30. **What would you change first with real cinema POS data?**  
    (1) Add `T-48h` advance pre-sale count (`tickets_already_booked_at_T48h`)—the strongest real-world predictor, (2) integrate external weather, school holidays, and competing release schedules, and (3) switch to Tobit/censored regression or uncensoring for sold-out screen-upgrade decisions.

---

## Part B — 3 "Live Change" Walkthrough Exercises

### Exercise 1: Add a New `T-48h`-Safe Feature (e.g. `competing_shows_same_day`)
1. **Verify availability:** Total scheduled shows across all screens on `show_date` is known when the schedule is published (`<= T-48h`).
2. **Edit [src/features.py](file:///Users/rahulkumar/Downloads/Cinema%20Ticket%20Demand%20Forecasting/src/features.py):**
   - Register `"competing_shows_same_day": "Schedule creation (daily show count)"` in `FEATURE_AVAILABILITY`.
   - Append `"competing_shows_same_day"` to `NUMERIC_FEATURES`.
   - In `build_features()`, compute `df["competing_shows_same_day"] = df.groupby("show_date")["show_id"].transform("count") - 1`.
3. **Edit [src/predict.py](file:///Users/rahulkumar/Downloads/Cinema%20Ticket%20Demand%20Forecasting/src/predict.py) & [api/main.py](file:///Users/rahulkumar/Downloads/Cinema%20Ticket%20Demand%20Forecasting/api/main.py):** Add optional `competing_shows_same_day: int = 39` to `ShowForecastRequest` and include it in `_build_feature_row`.
4. **Re-run & verify:** `make data && make train && make evaluate && make test`.

### Exercise 2: Change the Forecast Horizon (e.g. `T-48h` → `T-24h` or `T-72h`)
1. **Edit [config.yaml](file:///Users/rahulkumar/Downloads/Cinema%20Ticket Demand%20Forecasting/config.yaml):** Update `project.horizon_hours: 24` (or `72`).
2. **Check [src/features.py](file:///Users/rahulkumar/Downloads/Cinema%20Ticket%20Demand%20Forecasting/src/features.py):** `run_feature_pipeline()` already reads `cfg["project"]["horizon_hours"]` dynamically and passes it to `_compute_group_cutoff_stats(..., horizon_hours=...)`!
3. **Update leakage test [tests/test_features_no_leakage.py](file:///Users/rahulkumar/Downloads/Cinema%20Ticket%20Demand%20Forecasting/tests/test_features_no_leakage.py):** Read `horizon_hours = load_config()["project"]["horizon_hours"]` instead of hardcoded `48`.
4. **Re-run:** `make data && make train && make evaluate && make test` to observe how MAE improves at `T-24h` (more recent lags) vs widens at `T-72h`.

### Exercise 3: Swap or Add a Model Family (e.g. `PoissonRegressor` or `XGBoost`)
1. **Install dependency** (if external, e.g., `xgboost` in `requirements.txt`) or import `sklearn.linear_model.PoissonRegressor`.
2. **Edit [src/train.py](file:///Users/rahulkumar/Downloads/Cinema%20Ticket%20Demand%20Forecasting/src/train.py):**
   - Note: `PoissonRegressor` predicts non-negative counts (`tickets_sold`) or positive rates; if modeling `occupancy_rate`, add `TweedieRegressor(power=1.5)` or `ExtraTreesRegressor` inside `search_space` with a small validation grid.
   - Add `"Poisson"` / `"ExtraTrees"` to `search_space`.
3. **Edit [src/explain.py](file:///Users/rahulkumar/Downloads/Cinema%20Ticket%20Demand%20Forecasting/src/explain.py) & [src/predict.py](file:///Users/rahulkumar/Downloads/Cinema%20Ticket%20Demand%20Forecasting/src/predict.py):** Ensure SHAP explainer handles the new model type (e.g., `TreeExplainer` for `ExtraTreesRegressor`, `LinearExplainer` for GLM/Ridge).
4. **Re-run:** `make train && make evaluate && make test` and inspect `reports/experiments.csv` + `reports/metrics.json`.

