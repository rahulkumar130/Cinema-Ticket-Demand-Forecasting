"""Streamlit decision-support UI for T-48h Cinema Ticket Demand Forecasting.

Calls the FastAPI service when running (API_URL env var, default http://127.0.0.1:8000)
and falls back to loading `models/model_bundle.joblib` directly via `src.predict`
so the app always serves live model predictions (zero hard-coded outputs).

Run locally:
    streamlit run app/streamlit_app.py --server.port 8501
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_artifacts, predict_single_show

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Cinema Ticket Demand Forecaster (T-48h)",
    page_icon="🎬",
    layout="wide",
)


def _call_forecast_backend(payload: dict) -> tuple[dict, str]:
    """Try FastAPI /predict first; fall back cleanly to local joblib inference."""
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=2.0)
        if resp.status_code == 200:
            return resp.json(), f"FastAPI ({API_URL}/predict)"
    except requests.RequestException:
        pass
    return predict_single_show(payload), "Local Saved Artifact (models/model_bundle.joblib)"


def main() -> None:
    bundle, metadata, _, _ = load_artifacts()
    priors = bundle["reference_priors"]
    known_movies = priors.get("movies", {})
    known_screens = priors.get("screens", {})

    st.title("🎬 Cinema Ticket Demand Forecaster (T-48h)")
    st.caption(
        f"Decision-support forecast for hall sizing & staffing 48 hours before showtime | "
        f"Active Model: **{metadata['best_model_name']}** | Seed: `{metadata['seed']}`"
    )

    with st.form("show_forecast_form"):
        st.subheader("Scheduled Show Details (Known at T-48h)")
        col1, col2, col3 = st.columns(3)

        movie_options = ["Custom / Unknown New Release"] + sorted(known_movies.keys())
        with col1:
            selected_movie = st.selectbox("Movie Selection", options=movie_options, index=0)
            if selected_movie == "Custom / Unknown New Release":
                movie_id = st.text_input("Custom Movie ID", value="NEW-2025-001")
                default_genre, default_lang, default_rating, default_rt = "Action", "English", "PG-13", 125
                default_hist = 0
            else:
                info = known_movies[selected_movie]
                movie_id = selected_movie
                default_genre = info["genre"]
                default_lang = info["language"]
                default_rating = info["rating"]
                default_rt = int(round(info["runtime_min"]))
                default_hist = int(info["total_train_shows"])
                st.caption(f"Catalog Title: {info['movie_title']}")

            genres = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Animation", "Romance", "Thriller"]
            genre = st.selectbox("Genre", options=genres, index=genres.index(default_genre) if default_genre in genres else 0)
            languages = ["English", "Hindi", "Spanish", "Regional"]
            language = st.selectbox("Language", options=languages, index=languages.index(default_lang) if default_lang in languages else 0)

        with col2:
            ratings = ["G", "PG", "PG-13", "R"]
            rating = st.selectbox("Rating", options=ratings, index=ratings.index(default_rating) if default_rating in ratings else 2)
            runtime_min = st.number_input("Runtime (min)", min_value=60, max_value=240, value=default_rt, step=5)
            default_show = date(2025, 6, 28)
            default_rel = default_show - timedelta(days=1 if selected_movie == "Custom / Unknown New Release" else 10)
            release_date = st.date_input("Release Date", value=default_rel)
            show_date = st.date_input("Scheduled Show Date", value=default_show)
            slot = st.selectbox("Showtime Slot", options=["matinee", "evening", "late"], index=1)

        with col3:
            screen_options = sorted(known_screens.keys()) + ["Custom Screen"]
            selected_screen = st.selectbox("Screen", options=screen_options, index=3 if len(screen_options) > 3 else 0)
            if selected_screen in known_screens:
                def_stype = known_screens[selected_screen]["screen_type"]
                def_cap = int(known_screens[selected_screen]["capacity"])
            else:
                def_stype, def_cap = "Standard", 150
            stypes = ["Standard", "Dolby", "IMAX", "VIP"]
            screen_type = st.selectbox("Screen Format", options=stypes, index=stypes.index(def_stype) if def_stype in stypes else 0)
            capacity = st.number_input("Hall Capacity (Seats)", min_value=20, max_value=600, value=def_cap, step=5)
            ticket_price = st.number_input("Ticket Price ($)", min_value=4.0, max_value=60.0, value=15.0, step=0.5)
            shows_so_far_movie = st.number_input(
                "T-48h Movie History Depth (Prior Shows)",
                min_value=0,
                max_value=5000,
                value=default_hist,
                step=1,
            )

        c_flag1, c_flag2, c_flag3 = st.columns(3)
        with c_flag1:
            is_holiday = st.checkbox("Public Holiday", value=False)
        with c_flag2:
            is_special_event = st.checkbox("Special Promo / Event", value=False)
        with c_flag3:
            use_override = st.checkbox("Provide T-48h Observed Occupancy Prior", value=False)
            prior_occ = st.slider("Observed Prior Occupancy", 0.0, 1.0, 0.40, 0.02, disabled=not use_override)

        submitted = st.form_submit_button("Generate T-48h Forecast", type="primary")

    if submitted:
        payload = {
            "movie_id": movie_id,
            "genre": genre,
            "language": language,
            "rating": rating,
            "runtime_min": float(runtime_min),
            "release_date": release_date.strftime("%Y-%m-%d"),
            "show_date": show_date.strftime("%Y-%m-%d"),
            "slot": slot,
            "screen_id": selected_screen if selected_screen != "Custom Screen" else "SCR-CUSTOM",
            "screen_type": screen_type,
            "capacity": int(capacity),
            "ticket_price": float(ticket_price),
            "is_holiday": is_holiday,
            "is_special_event": is_special_event,
            "shows_so_far_movie": int(shows_so_far_movie),
            "prior_movie_occupancy": float(prior_occ) if use_override else None,
        }

        result, source_label = _call_forecast_backend(payload)

        st.divider()
        st.caption(f"Served by: `{source_label}`")

        if result["cold_start_warning"]:
            st.warning(f"⚠️ **Cold-Start / Low-History Warning:** {result['warning_reason']}")

        if result["sold_out_risk"] == "High":
            st.error("🔥 **High Sold-Out Risk:** Upper 80% interval approaches hall capacity — consider upgrading to a larger screen.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Forecasted Tickets", f"{result['predicted_tickets_rounded']} / {result['capacity']}")
        m2.metric("Predicted Occupancy", f"{result['predicted_occupancy_pct']:.1f}%")
        lo_t, hi_t = result["interval_80_tickets"]
        m3.metric("80% Uncertainty Interval", f"{lo_t:.0f} – {hi_t:.0f} tickets")
        m4.metric("Sold-Out Risk", result["sold_out_risk"])

        st.write("**Seat Occupancy Gauge**")
        st.progress(min(1.0, max(0.0, result["predicted_occupancy_pct"] / 100.0)))
        lo_o, hi_o = result["interval_80_occupancy_pct"]
        st.caption(f"80% occupancy interval: **{lo_o:.1f}% – {hi_o:.1f}%** (CQR-calibrated)")

        st.subheader("Why This Forecast? (Top SHAP Drivers)")
        factors = result.get("top_factors", [])
        if factors:
            fdf = pd.DataFrame(factors)
            fdf["impact_label"] = fdf["impact_occ_pts"].apply(lambda x: f"{x:+.2f} occ pts")
            st.bar_chart(fdf.set_index("feature")["impact_occ_pts"])
            st.dataframe(
                fdf[["feature", "impact_occ_pts", "impact_tickets", "direction"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Local SHAP driver breakdown is unavailable for this model family.")

        st.info(f"ℹ️ **Human-in-the-Loop Policy:** {result['disclaimer']}")


if __name__ == "__main__":
    main()

