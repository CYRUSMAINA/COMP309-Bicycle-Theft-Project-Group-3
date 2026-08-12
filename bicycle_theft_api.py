from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)


# Load saved ML objects
num_imputer = joblib.load("num_imputer.pkl")
cat_imputer = joblib.load("cat_imputer.pkl")
encoder = joblib.load("encoder.pkl")
standard_scaler = joblib.load("standard_scaler.pkl")
selector = joblib.load("selector.pkl")
logistic_model = joblib.load("logistic_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# Home route
@app.route("/")
def home():
    return "Bicycle Theft Prediction API is running!"


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    # Get JSON data
    json_data = request.get_json()

    # Convert JSON into DataFrame
    query = pd.DataFrame(json_data)

    # Temporary test response
    return jsonify({
        "message": "Prediction endpoint received the data",
        "received_data": query.to_dict(orient="records")
    })


# Start Flask
if __name__ == "__main__":
    app.run(debug=True)