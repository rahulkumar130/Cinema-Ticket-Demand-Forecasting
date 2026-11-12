"""FastAPI inference service for Cinema Ticket Demand Forecasting (T-48h).

Endpoints:
    GET  /health
    GET  /model-info
    POST /predict

Run locally:
    uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.predict import load_artifacts, predict_single_show


class ShowForecastRequest(BaseModel):
    """Validated input schema for a scheduled cinema show at T-48h."""

    movie_id: str = Field(default="NEW-MOVIE", description="Catalog ID or unknown title ID")
    genre: Literal["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Animation", "Romance", "Thriller"] = "Action"
    language: str = Field(default="English", min_length=1, max_length=40)
    rating: Literal["G", "PG", "PG-13", "R"] = "PG-13"
    runtime_min: float = Field(default=125.0, ge=60.0, le=240.0)
    release_date: str = Field(..., description="YYYY-MM-DD")
    show_date: str = Field(..., description="YYYY-MM-DD")
    slot: Literal["matinee", "evening", "late"] = "evening"
    screen_id: str = Field(default="SCR-04", min_length=1, max_length=32)
    screen_type: Literal["Standard", "Dolby", "IMAX", "VIP"] = "Dolby"
    capacity: int = Field(..., ge=20, le=600)
    ticket_price: float = Field(..., ge=4.0, le=60.0)
    is_holiday: bool = False
    is_special_event: bool = False
    shows_so_far_movie: int = Field(default=0, ge=0, le=5000)
    prior_movie_occupancy: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("release_date", "show_date")
    @classmethod
    def _validate_iso_date(cls, value: str) -> str:
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("Date must be in YYYY-MM-DD format") from exc
        return value


class FactorImpact(BaseModel):
    feature: str
    impact_occ_pts: float
    impact_tickets: float
    direction: Literal["up", "down"]


class ShowForecastResponse(BaseModel):
    model_name: str
    predicted_tickets: float
    predicted_tickets_rounded: int
    predicted_occupancy_pct: float
    interval_80_tickets: list[float]
    interval_80_occupancy_pct: list[float]
    capacity: int
    cold_start_warning: bool
    warning_reason: str | None
    sold_out_risk: Literal["Low", "Moderate", "High"]
    top_factors: list[FactorImpact]
    model_version_created_at: str | None
    disclaimer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load saved joblib artifact and metadata at startup."""
    bundle, metadata, _, _ = load_artifacts()
    app.state.bundle = bundle
    app.state.metadata = metadata
    yield


app = FastAPI(
    title="Cinema Ticket Demand Forecasting API",
    description="T-48h show-level ticket demand & occupancy decision-support service.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, object]:
    bundle = getattr(app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle not loaded.")
    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": bundle["best_model_name"],
    }


@app.get("/model-info")
def model_info() -> dict[str, object]:
    bundle = getattr(app.state, "bundle", None)
    metadata = getattr(app.state, "metadata", None)
    if bundle is None or metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not loaded.")
    priors = bundle["reference_priors"]
    return {
        "metadata": metadata,
        "known_movie_count": len(priors.get("movies", {})),
        "known_screens": priors.get("screens", {}),
    }


@app.post("/predict", response_model=ShowForecastResponse)
def predict(request: ShowForecastRequest) -> ShowForecastResponse:
    if not hasattr(app.state, "bundle"):
        load_artifacts()
    try:
        result = predict_single_show(request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction error: {exc}") from exc
    return ShowForecastResponse(**result)

