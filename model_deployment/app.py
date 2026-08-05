"""Flask web application and JSON API for bicycle recovery predictions."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.environ.get(
        "BICYCLE_MODEL_PATH",
        BASE_DIR / "artifacts" / "bicycle_recovery_model.pkl",
    )
)


def load_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Serialized model not found at {path}. Run train_models.py first."
        )
    # Pickle files can execute code while loading. Only load the trusted artifact
    # produced locally by train_models.py.
    with path.open("rb") as file:
        return pickle.load(file)


artifact = load_artifact(MODEL_PATH)
pipeline = artifact["pipeline"]
features = artifact["features"]
numerical_features = set(artifact["numerical_features"])
metadata = artifact["metadata"]
decision_threshold = float(artifact["decision_threshold"])

app = Flask(__name__)


def prepare_record(payload: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    missing_features = [feature for feature in features if feature not in payload]
    if missing_features:
        raise ValueError(
            "Missing required feature(s): " + ", ".join(missing_features)
        )

    record: dict[str, Any] = {}
    null_features: list[str] = []
    for feature in features:
        value = payload.get(feature)
        if value is None or (isinstance(value, str) and not value.strip()):
            record[feature] = np.nan
            null_features.append(feature)
        elif feature in numerical_features:
            try:
                record[feature] = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{feature} must be a number.") from error
        else:
            record[feature] = str(value).strip()

    return pd.DataFrame([record], columns=features), null_features


def make_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    model_input, imputed_features = prepare_record(payload)
    recovery_probability = float(pipeline.predict_proba(model_input)[0, 1])
    recovered = recovery_probability >= decision_threshold
    return {
        "prediction": "RECOVERED" if recovered else "NOT RECOVERED",
        "recovered": recovered,
        "recovery_probability": round(recovery_probability, 6),
        "not_recovered_probability": round(1 - recovery_probability, 6),
        "decision_threshold": decision_threshold,
        "model": artifact["selected_model"],
        "imputed_features": imputed_features,
        "disclaimer": metadata["important_note"],
    }


@app.get("/health")
def health() -> tuple[Any, int]:
    return (
        jsonify(
            {
                "status": "ok",
                "model_loaded": True,
                "model": artifact["selected_model"],
            }
        ),
        200,
    )


@app.get("/api/schema")
def api_schema() -> tuple[Any, int]:
    return jsonify(metadata), 200


@app.post("/predict")
def predict() -> tuple[Any, int]:
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON."}), 415
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request JSON must be an object."}), 400
    try:
        return jsonify(make_prediction(payload)), 200
    except ValueError as error:
        return jsonify({"error": str(error), "expected_features": features}), 400


@app.route("/", methods=["GET", "POST"])
def home() -> tuple[str, int] | str:
    values = dict(metadata["example_input"])
    prediction_result = None
    error_message = None

    if request.method == "POST":
        values.update(request.form.to_dict())
        try:
            prediction_result = make_prediction(values)
        except ValueError as error:
            error_message = str(error)

    return render_template(
        "index.html",
        schema=metadata,
        values=values,
        prediction=prediction_result,
        error=error_message,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
