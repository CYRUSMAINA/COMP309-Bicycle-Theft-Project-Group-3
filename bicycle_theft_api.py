from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)


# ============================================================
# LOAD SAVED ML OBJECTS
# ============================================================

num_imputer = joblib.load("num_imputer.pkl")
cat_imputer = joblib.load("cat_imputer.pkl")
encoder = joblib.load("encoder.pkl")
standard_scaler = joblib.load("standard_scaler.pkl")
selector = joblib.load("selector.pkl")
logistic_model = joblib.load("logistic_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# ============================================================
# FEATURES USED DURING MODEL TRAINING
# ============================================================

numerical_features = [
    "OCC_YEAR",
    "OCC_DAY",
    "OCC_DOY",
    "OCC_HOUR",
    "REPORT_YEAR",
    "REPORT_DAY",
    "REPORT_DOY",
    "REPORT_HOUR",
    "BIKE_SPEED",
    "BIKE_COST",
    "LONG_WGS84",
    "LAT_WGS84"
]

categorical_features = [
    "OCC_MONTH",
    "OCC_DOW",
    "REPORT_MONTH",
    "REPORT_DOW",
    "DIVISION",
    "LOCATION_TYPE",
    "PREMISES_TYPE",
    "BIKE_MAKE",
    "BIKE_TYPE",
    "BIKE_COLOUR",
    "PRIMARY_OFFENCE",
    "NEIGHBOURHOOD_158",
    "NEIGHBOURHOOD_140"
]


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/")
def home():
    return "Bicycle Theft Prediction API is running!"


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # 1. GET JSON DATA
        # ----------------------------------------------------

        json_data = request.get_json()

        if not json_data:
            return jsonify({
                "error": "No JSON data received"
            }), 400


        # ----------------------------------------------------
        # 2. CONVERT JSON INTO DATAFRAME
        # ----------------------------------------------------

        query = pd.DataFrame([json_data])


        # ----------------------------------------------------
        # 3. CHECK REQUIRED FEATURES
        # ----------------------------------------------------

        required_features = numerical_features + categorical_features

        missing_features = [
            feature for feature in required_features
            if feature not in query.columns
        ]

        if missing_features:

            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features
            }), 400


        # ----------------------------------------------------
        # 4. SELECT NUMERICAL AND CATEGORICAL DATA
        # ----------------------------------------------------

        X_numerical = query[numerical_features].copy()

        X_categorical = query[categorical_features].copy()


        # ----------------------------------------------------
        # 5. IMPUTE MISSING NUMERICAL VALUES
        # ----------------------------------------------------

        X_numerical = num_imputer.transform(X_numerical)


        # ----------------------------------------------------
        # 6. IMPUTE MISSING CATEGORICAL VALUES
        # ----------------------------------------------------

        X_categorical = cat_imputer.transform(X_categorical)


        # ----------------------------------------------------
        # 7. ONE-HOT ENCODE CATEGORICAL FEATURES
        # ----------------------------------------------------

        X_categorical = encoder.transform(X_categorical)


        # Convert sparse matrix to normal array if necessary
        if hasattr(X_categorical, "toarray"):
            X_categorical = X_categorical.toarray()


        # ----------------------------------------------------
        # 8. COMBINE NUMERICAL + CATEGORICAL FEATURES
        # ----------------------------------------------------

        X_processed = np.hstack([
            X_numerical,
            X_categorical
        ])


        # ----------------------------------------------------
        # 9. STANDARDIZE FEATURES
        # ----------------------------------------------------

        X_processed = standard_scaler.transform(X_processed)


        # ----------------------------------------------------
        # 10. SELECT THE 100 FEATURES
        # ----------------------------------------------------

        X_processed = selector.transform(X_processed)


        # ----------------------------------------------------
        # 11. MAKE PREDICTION
        # ----------------------------------------------------

        prediction_encoded = logistic_model.predict(X_processed)


        # ----------------------------------------------------
        # 12. CONVERT NUMBER BACK TO STATUS
        # ----------------------------------------------------

        prediction = label_encoder.inverse_transform(
            prediction_encoded
        )


        # ----------------------------------------------------
        # 13. RETURN RESULT
        # ----------------------------------------------------

        return jsonify({
            "prediction": prediction[0]
        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)