"""Streamlit dashboard for the Student Performance Prediction System."""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "student_model.pkl"
DATA_PATH = ROOT / "student_data.csv"
METRICS_PATH = ROOT / "evaluation_results.json"

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

st.title("🎓 Student Performance Prediction System")
st.caption("Classroom prototype for exploring academic risk. Predictions are not diagnoses or final decisions.")

if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("Model or dataset file is missing. Run `python train_model.py` first.")
    st.stop()

model = load_model()
data = load_data()

c1, c2, c3 = st.columns(3)
c1.metric("Historical student records", f"{len(data):,}")
c2.metric("Average final Mathematics grade (G3)", f"{data['G3'].mean():.2f}")
c3.metric("Historical at-risk records", f"{int((data['G3'] < 10).sum()):,}")

st.subheader("Dataset preview — observed historical data")
st.dataframe(data.head(10), use_container_width=True)

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    st.subheader("Model evaluation — held-out test data")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Precision (at risk)", f"{metrics['precision_at_risk']:.3f}")
    m3.metric("Recall (at risk)", f"{metrics['recall_at_risk']:.3f}")
    m4.metric("F1-score (at risk)", f"{metrics['f1_at_risk']:.3f}")
    st.caption("These are measured evaluation results from the test split, not predictions for an individual student.")

st.divider()
st.subheader("Predict uploaded student records")
st.write("Upload a CSV containing the same predictor columns used during training. Do not include G3.")
feature_columns = list(model.feature_names_in_)
template = data.drop(columns=["G3"]).loc[:, feature_columns].head(5)
st.download_button(
    "Download sample input CSV",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="sample_prediction_input.csv",
    mime="text/csv",
)

uploaded = st.file_uploader("Choose student CSV", type=["csv"])
if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        missing = [column for column in feature_columns if column not in new_data.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            predictions = model.predict(new_data.loc[:, feature_columns])
            results = new_data.copy()
            results["Prediction"] = [
                "AT RISK" if int(value) == 1 else "NOT AT RISK"
                for value in predictions
            ]
            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)
            st.metric("Predicted at-risk records", int((predictions == 1).sum()))
            st.download_button(
                "Download prediction results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="student_predictions.csv",
                mime="text/csv",
            )
            st.warning(
                "Use this output only as an early-support signal. A faculty member should review "
                "the student's circumstances and other evidence before taking action."
            )
    except Exception as exc:
        st.error(f"Could not process the uploaded CSV: {exc}")
