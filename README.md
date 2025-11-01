# Credit Card Fraud Detection

An end-to-end credit card fraud detection pipeline with training scripts and an interactive Streamlit dashboard for exploration, prediction, and SHAP-based explainability.

This repository trains an XGBoost classifier on a Kaggle-style credit card transactions dataset, addresses class imbalance with SMOTE, uses PCA for dimensionality reduction, and exposes predictions and explanations via a Streamlit app.

## Quick links
- Training script: `train_model.py`
- Streamlit app: `app.py`
- Requirements: `requirements.txt`
- Trained artifacts (output): `models/` (contains `model.pkl`, `scaler.pkl`, `pca.pkl`)

## Requirements
- Python 3.10+ recommended (should work with 3.8+ but 3.10+ is preferred)
- Windows (PowerShell) instructions are provided below

Dependencies are listed in `requirements.txt`. The main libraries used are:
- numpy, pandas
- scikit-learn, imbalanced-learn
- xgboost
- streamlit, plotly, matplotlib, seaborn
- shap, joblib

## Setup (Windows PowerShell)
Open PowerShell, navigate to the project root, and run the following commands to create a virtual environment and install dependencies:

```powershell
cd "C:\Users\nandi\OneDrive\Desktop\Creditcard_fraud_detection"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notes:
- If you cannot run `Activate.ps1` because of an execution policy, run PowerShell as Administrator and allow script execution for the current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## Dataset
Place the dataset CSV at `data/creditcard.csv`. This project expects the Kaggle-style dataset with columns including `Time`, `V1`..`V28`, `Amount`, and `Class` (0 = legitimate, 1 = fraud).

If you don't have the dataset in the repo, download it from Kaggle and save it to `data/creditcard.csv`.

## Train the model
Run the training pipeline which will:
- Load the dataset
- Create `Amount_log = log1p(Amount)` and drop `Time`/raw `Amount`
- Split dataset (80/20 stratified)
- Scale features, apply SMOTE to training set, fit PCA, and run a small GridSearchCV for XGBoost
- Save artifacts and evaluation plots to `models/`

To train, run:

```powershell
python train_model.py
```

After successful training, confirm the following files exist in `models/`:
- `model.pkl` — trained XGBoost model
- `scaler.pkl` — fitted StandardScaler
- `pca.pkl` — fitted PCA transformer
- `confusion_matrix.png` and `precision_recall_curve.png` — evaluation plots

## Run the Streamlit app (dashboard)
The dashboard loads saved artifacts from `models/` and provides interactive exploration and single-transaction predictions with SHAP explanations.

Start the app with:

```powershell
streamlit run app.py
```

App behavior notes:
- If no model artifacts are found, the app will ask you to train and place the files in `models/`.
- The app can optionally load `data/creditcard.csv` as a sample dataset or accept an uploaded CSV via the sidebar.

## File / Folder overview
- `app.py` — Streamlit dashboard for exploration and prediction
- `train_model.py` — training pipeline; produces `models/*.pkl` and evaluation plots
- `requirements.txt` — Python dependencies
- `DOCUMENTATION.md` — detailed project documentation and methodology
- `models/` — output directory for saved artifacts and plots (created by `train_model.py`)
- `assets/` — static assets such as `logo.png` used by the app
- `data/` — (not included) expected location for `creditcard.csv`

## Troubleshooting & tips
- Execution policy errors when activating the venv:
  - Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` in an elevated PowerShell or follow your org policy.
- Missing dataset error in `train_model.py` or `app.py`:
  - Ensure `data/creditcard.csv` exists and has the expected columns (`Time`, `V1..V28`, `Amount`, `Class`).
- Memory issues when applying SMOTE on the full dataset:
  - SMOTE can be memory-intensive. Try training on a sampled subset first (modify `train_model.py` to sample rows) or increase available RAM.
- SHAP plotting errors in the app:
  - SHAP may fail in some environments; the app catches and reports SHAP errors. If SHAP is required, ensure `shap` is installed and that `matplotlib` works in your environment.

## Development & next steps (optional)
- Add a small unit test that runs the preprocessing on a tiny CSV to guarantee the pipeline does not crash.
- Add a Dockerfile for reproducible deployment.
- Add a `requirements-dev.txt` and a GitHub Actions workflow for linting & tests.

## Contact
If you'd like help extending the README (e.g., adding Docker instructions, CI, or a condensed quickstart), tell me what you'd prefer and I can update it.

---
Short validated run commands summary (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
