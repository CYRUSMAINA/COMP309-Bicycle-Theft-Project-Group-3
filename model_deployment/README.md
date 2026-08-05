# Predictive Models, Evaluation, and Flask Deployment

This folder implements Brian's assigned COMP309 project section:

- logistic regression and decision tree classifiers using scikit-learn;
- accuracy, balanced accuracy, precision, recall, F1, confusion matrices, and ROC-AUC;
- automatic recommendation of the highest-ROC-AUC model;
- pickle serialization/deserialization of the complete preprocessing and model pipeline;
- a Flask JSON API and Jinja web client;
- a Python API test client and an importable Postman collection;
- test examples selected only from the untouched test split.

## Modeling assumptions

The binary target is `RECOVERED = 1`; `STOLEN` and `UNKNOWN` are grouped as `NOT RECOVERED = 0`. The positive class is about 1% of the dataset, so both classifiers use `class_weight="balanced"`. Accuracy is shown because the assignment asks for scores, but ROC-AUC, balanced accuracy, recall, precision, and F1 are also shown because ordinary accuracy is misleading for this imbalanced target.

`PRIMARY_OFFENCE` is excluded. Some recovered records contain values such as `PROPERTY - FOUND`, which reveal the outcome and would create target leakage. Identifier fields, free-text bike model, report dates, and map-projection coordinates are also excluded from the model API.

## 1. Install dependencies

From the repository root:

```powershell
cd model_deployment
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate` and use `python` instead of `py`.

## 2. Train and evaluate

```powershell
py train_models.py
```

The script reads `../Bicycle_Thefts.csv`, fits both required models, evaluates them on the same stratified 20% test set, chooses the model with the highest ROC-AUC, and writes:

- `artifacts/bicycle_recovery_model.pkl` - serialized winning pipeline;
- `artifacts/model_metadata.json` - schema, examples, and model details;
- `artifacts/test_samples.json` - records never used for training;
- `evaluation/model_scores.csv` and `model_metrics.json` - numeric results;
- `evaluation/confusion_matrix_*.png` - confusion matrices;
- `evaluation/roc_curve_comparison.png` - both ROC curves;
- `postman/Bicycle_Theft_Model_API.postman_collection.json` - Postman tests.

The completed results and report-ready interpretation are in [`MODEL_RESULTS.md`](MODEL_RESULTS.md).

## 3. Run the Flask API and web client

```powershell
py app.py
```

Open `http://127.0.0.1:5000` for the Jinja client. API routes:

- `GET /health`
- `GET /api/schema`
- `POST /predict`

Example JSON body (the generated Postman collection contains a real held-out row):

```json
{
  "OCC_YEAR": 2022,
  "OCC_DAY": 15,
  "OCC_HOUR": 18,
  "BIKE_SPEED": 21,
  "BIKE_COST": 1200,
  "LONG_WGS84": -79.3956,
  "LAT_WGS84": 43.6400,
  "OCC_MONTH": "July",
  "OCC_DOW": "Friday",
  "DIVISION": "D14",
  "LOCATION_TYPE": "Apartment (Rooming House, Condo)",
  "PREMISES_TYPE": "Apartment",
  "BIKE_MAKE": "TREK",
  "BIKE_TYPE": "RG",
  "BIKE_COLOUR": "BLK",
  "NEIGHBOURHOOD_158": "Harbourfront-CityPlace (165)"
}
```

## 4. Test the API

Keep `app.py` running, open a second terminal in this folder, then run:

```powershell
py test_api_client.py
```

For Postman, import `postman/Bicycle_Theft_Model_API.postman_collection.json`, start Flask, and run the collection. The collection performs a health check and sends a held-out test record to `/predict`.

To test model deserialization and Flask routes without starting a server:

```powershell
py smoke_test.py
```

> Security note: only deserialize the pickle file created by this project. Never load an untrusted pickle file.
