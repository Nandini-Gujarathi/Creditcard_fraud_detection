# Project Documentation — Credit Card Fraud Detection

Aim
To develop and demonstrate an end-to-end credit card fraud detection pipeline that trains a robust classifier on a real-world dataset, addresses the severe class imbalance, and exposes model predictions and explanations through an interactive dashboard.

Objective
- Load and preprocess a Kaggle-style credit card transactions dataset and engineer a stable `Amount_log` feature.  
- Handle class imbalance using SMOTE and reduce feature dimensionality with PCA for efficient, stable training.  
- Train an XGBoost binary classifier and perform light hyperparameter tuning (GridSearchCV) optimizing ROC AUC.  
- Evaluate performance with ROC AUC, average precision (PR AUC), confusion matrix and precision–recall curves; save plots for reference.  
- Provide an interactive Streamlit dashboard for data exploration, quick model checks, single-transaction prediction, and SHAP explainability.

Introduction
Credit card fraud detection is a high-impact binary classification problem where fraudulent transactions are rare compared to legitimate ones. This project uses a Kaggle-style dataset containing anonymized PCA features (V1..V28), `Amount`, `Time`, and `Class`. The pipeline demonstrates practical steps to prepare data, handle imbalance with SMOTE, compress features via PCA, train an XGBoost classifier, and present results in a Streamlit app with SHAP-based explanations for transparency.

Method
- Data ingestion: Read CSV and validate required columns (`Amount`, `Class`).  
- Feature engineering: Create a derived `Amount_log = log1p(Amount)` to reduce skew and drop raw `Amount` and `Time`.  
- Preprocessing: Standard scale features using `StandardScaler`.  
- Class balancing: Use SMOTE on the training set to synthetically oversample the minority (fraud) class.  
- Dimensionality reduction: Apply PCA (for example n_components=10 in training) on the scaled and oversampled training set to reduce dimensionality and speed up model training.  
- Model training: Train an XGBoost classifier (binary:logistic) with a constrained GridSearchCV over `max_depth`, `n_estimators`, and `learning_rate` to keep compute practical while optimizing ROC AUC.  
- Evaluation: Assess model on held-out test set using ROC AUC and average precision (PR AUC). Save confusion matrix and precision–recall curve plots to `models/`.  
- Explainability & UI: Load saved artifacts (`model.pkl`, `scaler.pkl`, `pca.pkl`) in Streamlit; for single predictions compute SHAP values via `TreeExplainer` and render summary plots for feature contributions.

Material
- Dataset: `data/creditcard.csv` — Kaggle Credit Card Fraud dataset format (columns: `Time`, `V1`..`V28`, `Amount`, `Class`).  
- Scripts: `train_model.py` (training pipeline), `app.py` (Streamlit dashboard).  
- Output artifacts: `models/model.pkl`, `models/scaler.pkl`, `models/pca.pkl`, and evaluation plots `models/confusion_matrix.png`, `models/precision_recall_curve.png`.  
- Libraries: (from `requirements.txt`) numpy, pandas, scikit-learn, imbalanced-learn, xgboost, shap, streamlit, plotly, matplotlib, seaborn, joblib.  
- Hardware/software: Python 3.10+ recommended. Training is CPU-friendly but may be slower on large datasets; a machine with >=8GB RAM is recommended when using SMOTE on full dataset.

Procedure
1. Environment setup (PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
2. Place the dataset at `data/creditcard.csv` (download from Kaggle if necessary).  
3. Train the model:
   ```powershell
   python train_model.py
   ```
   - The script will: load CSV, create `Amount_log`, drop `Time`, split into train/test (80/20 stratified), scale features, apply SMOTE to the training set, fit PCA, run GridSearchCV on an XGBoost classifier, evaluate on the test set, and save artifacts and figures to `models/`.
4. Inspect results: confirm `models/model.pkl`, `models/scaler.pkl`, `models/pca.pkl`, and plot files are present. Review printed ROC AUC, average precision, classification report, and confusion matrix in the training run output.  
5. Run the dashboard:
   ```powershell
   streamlit run app.py
   ```
   - Upload a CSV or use the sample dataset. The dashboard displays KPIs, histograms/boxplots, ROC/PR plots (sample), PCA scatter, and a single-transaction prediction with SHAP explanation (if SHAP works in the environment).

Result
- Artifacts: The training procedure produces `models/model.pkl`, `models/scaler.pkl`, and `models/pca.pkl`. These are required by the Streamlit app for predictions.  
- Evaluation plots: `models/confusion_matrix.png` and `models/precision_recall_curve.png` will be saved for result inspection.  
- Numeric metrics: Training prints ROC AUC and average precision (PR AUC) on the held-out test set. The classification report and confusion matrix (threshold 0.5 by default) are also printed.  
- Dashboard outputs: The Streamlit app provides interactive visualizations, a sampled ROC AUC metric, PCA projection, and single-transaction prediction with SHAP-based feature contributions for interpretability.  

Interpretation guidance
- ROC AUC (0.5 = random, 1.0 = perfect) measures ranking ability across thresholds. Use it for general model quality.  
- Average precision / PR AUC is better suited to heavily imbalanced data — it summarizes precision/recall trade-offs.  
- Confusion matrix and classification report at threshold 0.5 provide precision and recall; in fraud detection you may prefer tuning threshold to increase recall (catch more frauds) at the expense of precision.  
- SHAP values help explain individual predictions; interpret positive SHAP contributions as increasing predicted fraud probability and negative as decreasing it.

Limitations and notes
- SMOTE creates synthetic minority samples and can introduce noise; consider alternative strategies (class-weighting, ensemble methods, or more conservative resampling) depending on business risk tolerance.  
- PCA reduces interpretability of original features — if feature-level interpretability is required, consider skipping PCA for explainability workflows or keep a mapping/feature importances before PCA.  
- SHAP with PCA-transformed features explains the transformed space; feature-level interpretability is less direct when PCA is used.

Next steps (optional enhancements)
- Add unit tests for data loading and preprocessing and a smoke test that trains on a small sampled CSV.  
- Add CI with GitHub Actions to run linting and tests.  
- Create a Dockerfile for reproducible environment and easier deployment.  
- Provide an API (FastAPI/Flask) to serve predictions programmatically alongside the Streamlit UI.

---

If you want, I can also copy these sections into `README.md` or commit `DOCUMENTATION.md` (already added) and push to your remote. Tell me if you'd like a condensed one-page summary or a printable PDF version.
