# Final Submission Checklist

- [ ] 1. SAD mini-analysis: problem, user, inputs, process, outputs, and limitation (`SAD_mini_analysis_and_reflection.md`)
- [ ] 2. `train_model.py` and `app.py`
- [ ] 3. `requirements.txt` and `README.md`
- [ ] 4. Model evaluation results: accuracy, precision, recall, and F1-score. Run the training script and record actual values from `evaluation_results.json`.
- [ ] 5. Screenshot of the Streamlit dashboard and screenshot of prediction results. Capture these after running the app; do not use mock screenshots as proof.
- [ ] 6. GitHub repository URL. Create/push the repository and replace the placeholder URL in `README.md`.

## Commands

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Screenshot evidence to capture
1. Dashboard screenshot: show the title, dataset metrics, preview, and evaluation metrics.
2. Prediction screenshot: upload `sample_prediction_input.csv` and show the prediction results table and predicted at-risk count.
