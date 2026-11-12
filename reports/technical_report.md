# Technical Report — Cinema Ticket Demand Forecasting (`T-48h`)

**Author:** `[YOUR NAME]` (`[REG NO]`, `[EMAIL]`)  
**Assessment:** Algions AI Labs — AI & ML Internship  
**Horizon:** 48 hours before showtime (`T-48h`) | **Dataset:** Seeded Synthetic Cinema Simulation (`seed=42`)

---

## 1. Problem Framing, Prediction Unit, Target & Horizon
- **Business Decision:** Cinema operations managers must assign **which screen (hall capacity)** and **how many floor/concession staff** to allocate to each scheduled screening.
- **Prediction Unit:** One scheduled show (`movie × screen × date × showtime`).
- **Target Formulation:** Model `occupancy_rate = tickets_sold / capacity` $\in [0, 1]$ so the model normalizes across small (`85`-seat) and large (`302`-seat IMAX) screens, then convert predictions via $\hat{y}_i = \text{clip}(\hat{r}_i \cdot c_i, 0, c_i)$. Both occupancy points and ticket counts are reported.
- **Forecast Horizon & Cutoff:** Predictions are made **exactly 48 hours prior to `show_datetime` (`T-48h`)**. Only schedule/movie metadata, pricing, calendar flags, and completed show sales strictly prior to `show_datetime - 48h` are permitted.
- **Human-in-the-Loop Policy:** Outputs are strictly decision support (accompanied by 80% prediction intervals and cold-start warnings) and never automate screen cancellations or staff rosters.

---

## 2. Data Source, Generator Assumptions & Data Quality Handling
- **Source & Licence:** Self-generated synthetic dataset produced by [data/generate_data.py](../data/generate_data.py) (`licence: self-generated / MIT`, documented in [docs/data_card.md](../docs/data_card.md)). Contains zero personal or customer-level data.
- **Scale:** `21,708` raw shows across `540` days (`2024-01-01` to `2025-06-23`), `65` movies, and `8` screens (`Standard`, `Dolby`, `IMAX`, `VIP`; capacities `85–302`).
- **Encoded Demand Drivers:** Hidden per-movie popularity (with ~12% blockbusters, **not** exposed to models), release-age decay, weekday/Tuesday-discount/weekend seasonality, public holidays, occasional special promo events, slot × genre interactions, screen format, price elasticity, random Gaussian noise, and hard capacity capping (`sold_out`).
- **Cleaning & Train-Only Imputation ([src/data_prep.py](../src/data_prep.py)):**
  - Removed `108` duplicate records (`show_id` / `(screen_id, show_datetime)`).
  - Dropped `16` corrupted `tickets_sold < 0` rows and `16` corrupted `tickets_sold > capacity` POS glitches (preventing false `sold_out` labels; initially clipping `> capacity` to `capacity` caused 3 spurious worst-case test errors, which error analysis caught and fixed).
  - Nullified `16` impossible `ticket_price` (`<4` or `>60`) and `16` impossible `runtime_min` (`<60` or `>240`) outliers.
  - Fitted medians (`ticket_price` by `screen_type`, `runtime_min` by `genre`) and modes (`rating` by `genre`, `language`) **strictly on the training cutoff (`<= 2025-01-12`)**, reducing missingness from `538` total null cells (`~0.6%` per nullable column) to `0`.
  - Validated all types, ranges, and `tickets_sold <= capacity` with Pandera (`21,568` clean shows).

---

## 3. Exploratory Data Analysis (EDA) Highlights
Executed in [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb) with charts saved under `reports/figures/`:
- **Target Distributions (`eda_01_distributions.png`):** Clean `tickets_sold` has mean `47.23` (median `33`, max `302`); `occupancy_rate` has mean `0.27` (median `0.21`), with `288` sold-out shows (`1.34%` in clean data).
- **Demand Drivers (`eda_02_demand_drivers.png`):** Strong weekend surge (Fri–Sun, peaking Saturday), Tuesday discount bump over Mon/Wed, evening slot dominance over matinee/late, and steep post-release decay from `0–3d` (~60% mean occupancy) down to `31+d` (~11%).
- **Capacity & Censoring (`eda_03_capacity_and_censoring.png`):** Larger premium screens (`SCR-06` Dolby, `SCR-08` IMAX) exhibit higher mean occupancy and higher sold-out rates because blockbusters are preferentially scheduled in larger halls.
- **Time Series (`eda_05_timeseries.png`):** Clear weekly seasonality plus sharp single-day holiday/special-event spikes across the 18-month horizon.

---

## 4. `T-48h` Feature Engineering & Leakage Audit
### Feature Inventory (30 Features in [src/features.py](../src/features.py))
| Group | Features | Availability Time |
| :--- | :--- | :--- |
| **Calendar & Schedule (8)** | `weekday`, `month`, `is_weekend`, `is_tuesday`, `is_holiday`, `is_special_event`, `slot`, `show_hour` | Schedule publication (weeks ahead) |
| **Screen & Pricing (4)** | `screen_id`, `screen_type`, `capacity`, `ticket_price` | At or before `T-48h` |
| **Movie Metadata (5)** | `genre`, `language`, `rating`, `runtime_min`, `movie_age_days` | Pre-release / schedule creation |
| **History Depth (3)** | `shows_so_far_movie`, `is_new_movie`, `is_low_history` | Strictly `< show_datetime - 48h` |
| **Movie Causal Lags (6)** | `movie_occ_mean_all`, `movie_occ_mean_7d`, `movie_occ_mean_14d`, `movie_sold_out_rate_all`, `movie_slot_occ_mean`, `movie_weekday_occ_mean` | Strictly `< show_datetime - 48h` via `np.searchsorted(..., side="left")` |
| **Cold-Start Priors (4)** | `genre_slot_occ_prior`, `genre_occ_prior`, `slot_weekday_occ_prior`, `movie_effective_occ_prior` | Strictly `< show_datetime - 48h` |

### Leakage Audit ([notebooks/02_leakage_audit.ipynb](../notebooks/02_leakage_audit.ipynb) & `tests/test_features_no_leakage.py`)
1. **Brute-force equivalence:** Vectorized prefix-sum + `searchsorted` outputs match an explicit `df[df["show_datetime"] < t - 48h]` filter within `1e-9`.
2. **Future-target perturbation test:** Corrupting `occupancy_rate = 0.99` and `sold_out = 1` on all `1,081` shows at `>= 2024-02-18 11:00:00` produced **0.0 difference** in all 30 features at `2024-02-20 11:00:00`.
3. **Sanity checks (Validation Ridge):** Normal `T-48h` features achieved `14.56` tickets MAE; shuffling targets degraded MAE to `29.92` tickets (confirming genuine temporal alignment); dropping movie lag features degraded MAE to `15.08` tickets (confirming lags help without single-feature target leakage).
4. **Train-only Preprocessing:** `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` reside inside `sklearn.pipeline.Pipeline` fitted strictly on `train`.

---

## 5. Chronological Split, Baseline & Model Comparison
- **Chronological Split ([src/split.py](../src/split.py)):**
  - **Train (`70.0%`):** `15,097` shows (`2024-01-01` .. `2025-01-12`)
  - **Validation (`15.0%`):** `3,237` shows (`2025-01-13` .. `2025-04-03`) — used exclusively for hyperparameter selection and CQR calibration
  - **Test (`15.0%`, frozen):** `3,234` shows (`2025-04-04` .. `2025-06-23`) — evaluated once
- **Baseline ([src/baseline.py](../src/baseline.py)):** Capacity-adjusted hierarchical historical mean occupancy `(movie, weekday, slot) → (movie, slot) → (genre, slot) → slot → global mean` fitted on train.
- **Tuning (`reports/experiments.csv`):** Grid search on validation across Ridge (`alpha`), RandomForest (`max_depth`, `min_samples_leaf`), and HistGradientBoosting (`max_depth`, `learning_rate`, `min_samples_leaf`), plus 4-fold `TimeSeriesSplit` CV on train. Selected `HistGradientBoosting(max_iter=250, max_depth=4, learning_rate=0.08, min_samples_leaf=40)` (`Val MAE = 10.92`, `CV MAE = 10.32`).

### Untouched Test Set Results (`3,234` Shows)
| Model | MAE (Tickets) | RMSE (Tickets) | WAPE (%) | MAE (Occ Pts) | ±10 Occ Pts (%) | Clip Outside (%) | Sold-Out Prec / Rec | Mean Business Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Hist Avg)** | 26.18 | 37.16 | 60.86% | 15.14 | 39.27% | 0.00% | 0.00 / 0.00 | 139.47 |
| **Ridge** (`alpha=25`) | 12.66 | 17.95 | 29.43% | 7.36 | 73.41% | 6.12% | 0.00 / 0.00 | 68.91 |
| **RandomForest** | 10.08 | 14.36 | 23.43% | 5.86 | 83.64% | 0.00% | 0.29 / 0.36 | 59.88 |
| **HistGradientBoosting** | **9.93** | **13.79** | **23.09%** | **5.82** | 83.52% | 0.00% | 0.37 / 0.32 *(0.20 / 0.36 @ q90)* | **59.60** |

- **Business Cost Assumptions:** Under-forecasting costs `8.0` units/ticket (lost revenue + understaffing); over-forecasting costs `3.0` units/ticket (excess staffing/hall slack). `HistGradientBoosting` reduces per-show operational cost by **57.3%** (`139.47 → 59.60`) and MAE/WAPE by **62.05%**.
- **80% CQR Prediction Intervals:** Empirical test coverage = **82.59%** (`cqr_delta = +0.0143` occ), mean interval width = **32.00 tickets**.

---

## 6. Cold-Start, Capacity Buckets, Error Analysis & Explainability
### Cold-Start & Capacity Slices (Test Set)
- **Cold-start (`new_movie`, `n=226`):** `MAE = 16.11` tickets, `RMSE = 21.34`, `WAPE = 15.16%` (vs Baseline `MAE = 59.62`, `WAPE = 56.12%`).
- **Low-history (`1–5` prior shows, `n=5`):** `MAE = 18.05` tickets, `RMSE = 21.26`, `WAPE = 15.37%` (tiny `n=5` slice; interpreted cautiously).
- **Mature (`>5` prior shows, `n=3,003`):** `MAE = 9.45` tickets, `RMSE = 13.03`, `WAPE = 24.80%` (higher percentage WAPE reflects lower mean sales after 30+ days of decay).
- **Capacity Buckets:** Small (`<=120`): `MAE 5.90` (`24.44%` WAPE); Medium (`121–220`): `MAE 9.97` (`23.30%` WAPE); Large (`>220`): `MAE 15.91` (`22.23%` WAPE). Occupancy MAE stays remarkably uniform (`5.74–5.89` pts) across all hall sizes.

### Explainability ([src/explain.py](../src/explain.py))
- **Global Top Drivers (Permutation Importance):** `movie_age_days` (`+6.69` occ pts MAE when shuffled), `slot_weekday_occ_prior` (`+1.89`), `weekday` (`+0.88`), `movie_slot_occ_mean` (`+0.54`), `movie_occ_mean_7d` (`+0.43`).
- **Local Accurate Case:** *Comedy Feature #35* (`age 3d`, `SCR-01` cap `85`, `evening`, `2025-04-04`) — predicted `60` tickets (`80%` interval `51–76`) vs actual `57` (`-2.9` error), driven by opening age (`+0.216` occ), evening slot (`+0.139`), and `T-48h` slot history (`+0.043`).
- **Local Large Error Case:** *Comedy Feature #18* (`age 65d`, `SCR-08` IMAX cap `302`, `late`, `is_special_event=1`, `2025-04-14`) — predicted `20` tickets (`7–49`) vs actual `106` (`+85.7` error). The 65-day age decay (`-0.099` occ) outweighed the rare special-event boost (`+0.039`) inside a 302-seat hall.

---

## 7. Responsible Use, Limitations & Future Work
- **Synthetic-Data Caveat:** Models learn only generator-encoded signals; real-cinema deployment requires retraining on real POS transactions.
- **Sold-Out Censoring:** Only `22` actual sold-out shows occurred in the test set (`31.8%` point recall, `36.4%` upper-90th recall). Right-censoring hides unconstrained blockbuster overflow; managers must combine upper-bound alerts with pre-sale velocity.
- **Privacy & Governance:** Zero personal data; human operations managers hold final authority over screen and staffing decisions.
- **Future Work:** Add `T-48h` advance-booking curve features (tickets already sold by `T-48h`), censored Tobit/survival regression for unconstrained demand estimation, and monthly automated drift/retraining pipelines.

