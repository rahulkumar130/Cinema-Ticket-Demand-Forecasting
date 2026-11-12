"""Integration tests for FastAPI /health, /model-info, and /predict endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def test_health_and_model_info() -> None:
    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200
        h_json = h.json()
        assert h_json["status"] == "ok"
        assert h_json["model_loaded"] is True

        info = client.get("/model-info")
        assert info.status_code == 200
        assert info.json()["known_movie_count"] > 0


def test_predict_known_and_unknown_cold_start_movies() -> None:
    with TestClient(app) as client:
        # 1. Cold-start unknown movie
        cold_payload = {
            "movie_id": "UNKNOWN-NEW-TITLE",
            "genre": "Action",
            "language": "English",
            "rating": "PG-13",
            "runtime_min": 130,
            "release_date": "2025-06-27",
            "show_date": "2025-06-28",
            "slot": "evening",
            "screen_id": "SCR-08",
            "screen_type": "IMAX",
            "capacity": 302,
            "ticket_price": 19.5,
            "is_holiday": False,
            "is_special_event": False,
            "shows_so_far_movie": 0,
        }
        r_cold = client.post("/predict", json=cold_payload)
        assert r_cold.status_code == 200
        body = r_cold.json()
        assert body["cold_start_warning"] is True
        assert 0.0 <= body["predicted_tickets"] <= 302.0
        lo, hi = body["interval_80_tickets"]
        assert 0.0 <= lo <= body["predicted_tickets"] <= hi <= 302.0
        assert len(body["top_factors"]) > 0
        assert "human" in body["disclaimer"].lower()

        # 2. Known mature movie with prior history
        warm_payload = {
            **cold_payload,
            "movie_id": "MOV-001",
            "release_date": "2025-05-01",
            "shows_so_far_movie": 45,
        }
        r_warm = client.post("/predict", json=warm_payload)
        assert r_warm.status_code == 200
        warm_body = r_warm.json()
        assert warm_body["cold_start_warning"] is False
        assert 0.0 <= warm_body["predicted_tickets"] <= 302.0


def test_predict_rejects_invalid_input_gracefully() -> None:
    with TestClient(app) as client:
        bad_payload = {
            "movie_id": "MOV-001",
            "genre": "Action",
            "release_date": "2025-06-01",
            "show_date": "2025-06-10",
            "slot": "evening",
            "screen_type": "Standard",
            "capacity": -50,
            "ticket_price": -10.0,
        }
        r = client.post("/predict", json=bad_payload)
        assert r.status_code == 422

