"""Exercise deserialization and the Flask routes without starting a server."""

from __future__ import annotations

import json
from pathlib import Path

from app import app

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    samples = json.loads(
        (BASE_DIR / "artifacts" / "test_samples.json").read_text(encoding="utf-8")
    )
    with app.test_client() as client:
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.get_json()["model_loaded"] is True

        prediction_response = client.post("/predict", json=samples[0]["features"])
        assert prediction_response.status_code == 200
        result = prediction_response.get_json()
        assert result["prediction"] in {"RECOVERED", "NOT RECOVERED"}
        assert 0.0 <= result["recovery_probability"] <= 1.0

        bad_response = client.post("/predict", json={"OCC_YEAR": "not-a-number"})
        assert bad_response.status_code == 400

    print("Smoke test passed: pickle loading, /health, and /predict are working.")


if __name__ == "__main__":
    main()
