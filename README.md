# Student Performance Prediction System

A classroom machine-learning prototype that estimates whether a student is at risk of failing the final Mathematics grade.

## Problem and intended user
Academic difficulties may only become visible after poor results. This prototype is intended for faculty members or academic analysts to explore historical indicators and identify students who may benefit from additional support.

## Dataset
- Source: [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
- Dataset ID: 320
- Subject file used: Mathematics records
- Target: `G3 < 10` means **AT RISK**; `G3 >= 10` means **NOT AT RISK**.
- `G3` is the target only and is excluded from predictor features to prevent target leakage.

## Technologies
Python, pandas, scikit-learn, ucimlrepo, joblib, and Streamlit.

## Installation
Use Python 3.10 or later.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Train and evaluate the model
Run from this project directory:

```bash
python train_model.py
```

The script downloads the dataset, saves `student_data.csv`, trains a preprocessing + Random Forest pipeline, evaluates it on a stratified 20% test split, and saves:
- `student_model.pkl`
- `evaluation_results.json`

Record the actual accuracy, at-risk precision, at-risk recall, and at-risk F1-score printed by the script. Do not copy or invent metric values.

## Run the Streamlit dashboard

```bash
streamlit run app.py
```

Open the local URL printed in the terminal (usually http://localhost:8501). Download the sample input CSV from the app, upload it, inspect predictions, and download the result CSV.

## SAD mini-analysis
- **Problem:** Academic risk may be noticed only after poor results; historical indicators may help identify learners who need support earlier.
- **User:** Faculty member or academic analyst.
- **Inputs:** Student attributes and academic indicators, including study time, absences, past failures, G1, and G2.
- **Process:** Validate records, preprocess missing/numeric/categorical data, apply the trained classifier, and classify risk.
- **Outputs:** AT RISK / NOT AT RISK label, prediction-results CSV, and model evaluation summary.
- **Limitation:** The model is trained on one Portuguese secondary-school dataset. It may not generalize to another school, country, or student population.

## Model evaluation
The script reports accuracy and the precision, recall, and F1-score for the **At risk** class (positive class). The actual values are generated when you run `python train_model.py` and saved in `evaluation_results.json`. Results are not pre-filled because they must come from the executed model.

## Responsible use
This is a classroom demonstration using public historical data. Predictions are not diagnoses, do not prove that a student will fail, and must not be used as the sole basis for real student decisions. Faculty should review context and offer appropriate support.

## Repository
After creating a public GitHub repository named `student-performance-prediction`, add and push the source files. Replace the placeholder below with your real repository URL:

`https://github.com/YOUR-USERNAME/student-performance-prediction`
