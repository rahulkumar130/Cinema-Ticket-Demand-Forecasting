# Data Card — Synthetic Cinema Show-Level Demand Dataset

## 1. Overview & Provenance
- **Dataset Name:** `shows_raw.csv` / `shows_clean.parquet`
- **Source & Licence:** **Self-generated synthetic data** produced by [`data/generate_data.py`](../data/generate_data.py) (`licence: MIT / project-internal synthetic`). Contains **zero real customer, POS, or personally identifiable information (PII)**.
- **Generation Parameters (`config.yaml` defaults):**
  - Random seed: `42`
  - Horizon: `540` days (~18 months starting `2024-01-01`)
  - Screens: `8` screens (capacities ~85–305 across `Standard`, `Dolby`, `IMAX`, `VIP`)
  - Movies: `65` movies across 8 genres (`Action`, `Comedy`, `Drama`, `Horror`, `Sci-Fi`, `Animation`, `Romance`, `Thriller`), 4 languages, and 4 MPAA-style ratings
  - Daily slots: `5` daily showtimes (`11:00`, `14:30` -> `matinee`; `18:00`, `20:30` -> `evening`; `22:45` -> `late`)
  - Expected volume: ~21,600 raw shows

## 2. Prediction Unit & Schema
Each row represents **one scheduled show** (`movie × screen × date × showtime`):
- **Identifiers & schedule:** `show_id`, `show_date`, `show_time`, `show_datetime`, `slot`, `screen_id`
- **Screen metadata:** `screen_type`, `capacity`
- **Movie metadata:** `movie_id`, `movie_title`, `genre`, `language`, `rating`, `runtime_min`, `release_date`
- **Pricing & calendar:** `ticket_price`, `is_holiday`, `is_special_event`
- **Target & censoring:** `tickets_sold`, `occupancy_rate` (`tickets_sold / capacity`), `sold_out` (`1` if `tickets_sold == capacity`)

## 3. Generator Assumptions & Encoded Drivers
The simulation computes a latent show occupancy rate using:
1. **Hidden movie popularity:** Drawn per movie (not exposed in the dataset), with ~12% designated as high-popularity blockbusters.
2. **Release age decay:** Peak demand during opening days (`age_days <= 2`), moderate during week 1 (`3–6`), followed by exponential decay over ~6–8 weeks.
3. **Calendar & holidays:** Weekend surges (Fri–Sun), Tuesday discount bump, public holiday boost (`+24%`), and occasional special promo events (`+22%`).
4. **Slot × genre interactions:** Evening slots outperform matinees; weekend Animation matinees and Fri/Sat late Horror shows receive explicit boosts.
5. **Screen format & price elasticity:** `IMAX` and `Dolby` attract higher occupancy; higher `ticket_price` relative to format reference dampens demand.
6. **Capacity censoring:** Latent demand is multiplied by `capacity`, rounded, and **clipped to `[0, capacity]`**, creating realistic right-censored `sold_out` shows.
7. **Injected data-quality anomalies:** ~1.5% missing values (`ticket_price`, `runtime_min`, `rating`, `language`), ~0.5% duplicate records, and ~0.3% corrupted POS outliers (negative/extreme prices, impossible runtimes, negative or over-capacity ticket counts).

## 4. Known Limitations & Warnings
> **Synthetic-Data Caveat:** A machine learning model trained on this dataset can **only learn patterns explicitly encoded by the generator** plus random noise. Metrics achieved here validate pipeline correctness, leakage prevention, and modeling workflow—**they do not guarantee identical accuracy on real cinema POS data**, which exhibits unmodeled factors such as competitor releases, weather shocks, marketing spend, franchise loyalty, and dynamic pricing.

