"""Train a Random Forest model for student mathematics-risk prediction.

Target definition from the laboratory:
    G3 < 10  -> 1 (AT RISK)
    G3 >= 10 -> 0 (NOT AT RISK)

G3 is deliberately excluded from all predictor features.
"""
from pathlib import Path
import json

import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "student_data.csv"
MODEL_PATH = ROOT / "student_model.pkl"
METRICS_PATH = ROOT / "evaluation_results.json"

def main():
    print("Downloading UCI Student Performance dataset (ID 320)...")
    student = fetch_ucirepo(id=320)
    df = pd.concat(
        [student.data.features, student.data.targets],
        axis=1
    )
    df = df.loc[:, ~df.columns.duplicated()]
    df.to_csv(DATA_PATH, index=False)

    if "G3" not in df.columns:
        raise ValueError("The downloaded dataset does not contain the required G3 target.")

    df = df.drop_duplicates().copy()
    y = (pd.to_numeric(df["G3"], errors="coerce") < 10).astype(int)
    X = df.drop(columns=["G3"]).copy()  # Prevent target leakage.

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision_at_risk": round(float(precision_score(y_test, predictions, pos_label=1, zero_division=0)), 4),
        "recall_at_risk": round(float(recall_score(y_test, predictions, pos_label=1, zero_division=0)), 4),
        "f1_at_risk": round(float(f1_score(y_test, predictions, pos_label=1, zero_division=0)), 4),
        "test_records": int(len(y_test)),
        "training_records": int(len(y_train)),
        "confusion_matrix_labels": ["Not at risk", "At risk"],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, predictions,
            labels=[0, 1],
            target_names=["Not at risk", "At risk"],
            output_dict=True,
            zero_division=0,
        ),
        "positive_class": "At risk (G3 < 10)",
        "target_rule": "G3 < 10 = AT RISK; G3 >= 10 = NOT AT RISK",
        "random_state": 42,
        "test_size": 0.20,
    }

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nMODEL EVALUATION (positive class: AT RISK)")
    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"Precision: {results['precision_at_risk']:.4f}")
    print(f"Recall   : {results['recall_at_risk']:.4f}")
    print(f"F1-score : {results['f1_at_risk']:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions,
        target_names=["Not at risk", "At risk"],
        zero_division=0,
    ))
    print(f"\nSaved: {DATA_PATH.name}, {MODEL_PATH.name}, {METRICS_PATH.name}")

if __name__ == "__main__":
    main()
