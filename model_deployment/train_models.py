"""Train, compare, evaluate, and serialize bicycle recovery classifiers.

This script trains the two algorithms required by the assignment:
logistic regression and a decision tree. It evaluates both on the same
untouched stratified test set, recommends the model with the highest ROC-AUC,
and serializes the complete winning preprocessing/model pipeline with pickle.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from model_config import (
    CATEGORICAL_FEATURES,
    DECISION_THRESHOLD,
    FEATURES,
    FEATURE_DESCRIPTIONS,
    NUMERICAL_FEATURES,
    POSITIVE_STATUS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR.parent / "Bicycle_Thefts.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
EVALUATION_DIR = BASE_DIR / "evaluation"
POSTMAN_DIR = BASE_DIR / "postman"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to Bicycle_Thefts.csv",
    )
    return parser.parse_args()


def make_preprocessor() -> ColumnTransformer:
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def make_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                        solver="liblinear",
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        class_weight="balanced",
                        max_depth=12,
                        min_samples_leaf=20,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def load_data(dataset_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = pd.read_csv(dataset_path, encoding="utf-8-sig", low_memory=False)
    required_columns = set(FEATURES + [TARGET_COLUMN])
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    data = data.dropna(subset=[TARGET_COLUMN]).copy()
    X = data[FEATURES].copy()
    # Binary project target: returned/recovered versus not returned.
    y = data[TARGET_COLUMN].eq(POSITIVE_STATUS).astype(int)
    return X, y


def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> tuple[dict[str, Any], np.ndarray]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1_score": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "confusion_matrix": matrix.tolist(),
        "true_negative": int(matrix[0, 0]),
        "false_positive": int(matrix[0, 1]),
        "false_negative": int(matrix[1, 0]),
        "true_positive": int(matrix[1, 1]),
    }
    return metrics, predictions


def save_confusion_matrix(
    model_name: str,
    y_true: pd.Series,
    predictions: np.ndarray,
) -> None:
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=[0, 1],
        display_labels=["Not recovered", "Recovered"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title(f"{model_name} - Test Confusion Matrix")
    output_name = model_name.lower().replace(" ", "_")
    plt.tight_layout()
    plt.savefig(EVALUATION_DIR / f"confusion_matrix_{output_name}.png", dpi=180)
    plt.close()


def json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def build_test_samples(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_count: int = 10,
) -> list[dict[str, Any]]:
    # Include both classes so the API demonstration is useful even though only
    # about 1% of the full dataset belongs to the recovered class.
    positive_indices = y_test[y_test == 1].sample(
        n=min(sample_count // 2, int((y_test == 1).sum())),
        random_state=RANDOM_STATE,
    ).index.tolist()
    negative_count = min(sample_count - len(positive_indices), int((y_test == 0).sum()))
    negative_indices = y_test[y_test == 0].sample(
        n=negative_count,
        random_state=RANDOM_STATE,
    ).index.tolist()
    sample_indices = positive_indices + negative_indices
    samples: list[dict[str, Any]] = []
    for index in sample_indices:
        row = X_test.loc[index]
        samples.append(
            {
                "features": {
                    feature: json_safe_value(row[feature]) for feature in FEATURES
                },
                "expected_class": "RECOVERED" if int(y_test.loc[index]) else "NOT RECOVERED",
                "expected_recovered": bool(y_test.loc[index]),
            }
        )
    return samples


def build_metadata(
    X_train: pd.DataFrame,
    example_input: dict[str, Any],
    selected_model: str,
    metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    schema = []
    for feature in FEATURES:
        feature_type = "number" if feature in NUMERICAL_FEATURES else "string"
        item: dict[str, Any] = {
            "name": feature,
            "type": feature_type,
            "description": FEATURE_DESCRIPTIONS[feature],
            "required": True,
            "example": example_input[feature],
        }
        if feature in CATEGORICAL_FEATURES:
            item["suggested_values"] = (
                X_train[feature]
                .dropna()
                .astype(str)
                .value_counts()
                .head(30)
                .index.tolist()
            )
        schema.append(item)

    return {
        "project": "COMP309 Bicycle Theft Recovery Prediction",
        "selected_model": selected_model,
        "selection_rule": "Highest ROC-AUC on the untouched test set; F1 breaks a tie.",
        "decision_threshold": DECISION_THRESHOLD,
        "positive_class": POSITIVE_STATUS,
        "negative_class": "NOT RECOVERED (STATUS is STOLEN or UNKNOWN)",
        "feature_count": len(FEATURES),
        "features": schema,
        "example_input": example_input,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        "important_note": (
            "This is an educational model trained on historical police data. "
            "Its probability is not a guarantee that a bicycle will be recovered."
        ),
    }


def write_postman_collection(example_input: dict[str, Any]) -> None:
    collection = {
        "info": {
            "name": "COMP309 Bicycle Theft Model API",
            "description": (
                "Tests the Flask model service with one record held out from training."
            ),
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "variable": [{"key": "base_url", "value": "http://127.0.0.1:5000"}],
        "item": [
            {
                "name": "Health check",
                "request": {
                    "method": "GET",
                    "url": {
                        "raw": "{{base_url}}/health",
                        "host": ["{{base_url}}"],
                        "path": ["health"],
                    },
                },
            },
            {
                "name": "Predict bicycle recovery",
                "request": {
                    "method": "POST",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": {
                        "mode": "raw",
                        "raw": json.dumps(example_input, indent=2),
                        "options": {"raw": {"language": "json"}},
                    },
                    "url": {
                        "raw": "{{base_url}}/predict",
                        "host": ["{{base_url}}"],
                        "path": ["predict"],
                    },
                },
            },
        ],
    }
    path = POSTMAN_DIR / "Bicycle_Theft_Model_API.postman_collection.json"
    path.write_text(json.dumps(collection, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    POSTMAN_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Dataset: {len(X):,} rows")
    print(f"Recovered records: {int(y.sum()):,} ({y.mean():.2%})")
    print(f"Training rows: {len(X_train):,}")
    print(f"Untouched test rows: {len(X_test):,}\n")

    models = make_models()
    all_metrics: dict[str, dict[str, Any]] = {}
    roc_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        model_metrics, predictions = calculate_metrics(y_test, probabilities)
        all_metrics[model_name] = model_metrics
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
        roc_data[model_name] = (false_positive_rate, true_positive_rate)
        save_confusion_matrix(model_name, y_test, predictions)
        print(
            f"  ROC-AUC={model_metrics['roc_auc']:.4f}, "
            f"F1={model_metrics['f1_score']:.4f}, "
            f"balanced accuracy={model_metrics['balanced_accuracy']:.4f}"
        )

    selected_model_name = max(
        all_metrics,
        key=lambda name: (
            all_metrics[name]["roc_auc"],
            all_metrics[name]["f1_score"],
        ),
    )
    selected_pipeline = models[selected_model_name]

    plt.figure(figsize=(8, 6))
    for model_name, (false_positive_rate, true_positive_rate) in roc_data.items():
        auc_value = all_metrics[model_name]["roc_auc"]
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"{model_name} (AUC = {auc_value:.3f})",
        )
    plt.plot([0, 1], [0, 1], "k--", label="Random classifier (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison on Untouched Test Data")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(EVALUATION_DIR / "roc_curve_comparison.png", dpi=180)
    plt.close()

    test_samples = build_test_samples(X_test, y_test)
    example_input = test_samples[0]["features"]
    metadata = build_metadata(
        X_train,
        example_input,
        selected_model_name,
        all_metrics,
    )
    artifact = {
        "pipeline": selected_pipeline,
        "features": FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "selected_model": selected_model_name,
        "decision_threshold": DECISION_THRESHOLD,
        "metadata": metadata,
    }

    with (ARTIFACT_DIR / "bicycle_recovery_model.pkl").open("wb") as file:
        pickle.dump(artifact, file, protocol=pickle.HIGHEST_PROTOCOL)

    (ARTIFACT_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (ARTIFACT_DIR / "test_samples.json").write_text(
        json.dumps(test_samples, indent=2), encoding="utf-8"
    )
    (EVALUATION_DIR / "model_metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )

    score_rows = []
    for model_name, model_metrics in all_metrics.items():
        score_rows.append(
            {
                "model": model_name,
                **{
                    key: value
                    for key, value in model_metrics.items()
                    if key not in {"confusion_matrix"}
                },
                "recommended": model_name == selected_model_name,
            }
        )
    pd.DataFrame(score_rows).to_csv(
        EVALUATION_DIR / "model_scores.csv", index=False
    )
    write_postman_collection(example_input)

    print(f"\nRecommended model: {selected_model_name}")
    print(f"Serialized model: {ARTIFACT_DIR / 'bicycle_recovery_model.pkl'}")
    print(f"Evaluation files: {EVALUATION_DIR}")


if __name__ == "__main__":
    main()
