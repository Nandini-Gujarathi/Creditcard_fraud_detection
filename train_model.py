# train_model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             average_precision_score)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns 

sns.set(style="whitegrid")

DATA_PATH = "data/creditcard.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def preprocess_split(df):
    # Assumes dataset like Kaggle's creditcard.csv with 'Class' and 'Amount', 'Time'
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    # Optional: create derived features
    # log transform Amount to reduce skew
    X["Amount_log"] = np.log1p(X["Amount"])
    X = X.drop(columns=["Amount"])  # use Amount_log instead

    # Drop Time or scale it
    X = X.drop(columns=["Time"])
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def build_and_train(X_train, y_train):
    # Scaling then SMOTE then model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("Original training class distribution:", np.bincount(y_train))
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
    print("After SMOTE distribution:", np.bincount(y_res))

    # Option: dimensionality reduction for faster training (not required)
    pca = PCA(n_components=10, random_state=42)
    X_res_pca = pca.fit_transform(X_res)

    # Use XGBoost
    xgb = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="auc",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42,
        max_depth=6,
        n_estimators=200,
        learning_rate=0.05
    )

    # Quick hyperparam tuning (small grid - adjust to compute budget)
    param_grid = {
        "max_depth": [4, 6],
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1]
    }

    cv = GridSearchCV(xgb, param_grid, scoring="roc_auc", cv=3, verbose=1, n_jobs=-1)
    cv.fit(X_res_pca, y_res)
    print("Best params:", cv.best_params_)
    best = cv.best_estimator_

    # Save scaler and pca and model (but the app will transform raw features accordingly)
    return scaler, pca, best

def evaluate_model(scaler, pca, model, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_pca)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    avg_prec = average_precision_score(y_test, y_pred_proba)
    print("ROC AUC:", auc)
    print("Average Precision (PR AUC):", avg_prec)
    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # Save some evaluation plots
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion matrix")
    plt.savefig("models/confusion_matrix.png", bbox_inches='tight')
    plt.close()

    # Precision-Recall curve
    from sklearn.metrics import precision_recall_curve
    p, r, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6,4))
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.savefig("models/precision_recall_curve.png", bbox_inches='tight')
    plt.close()

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_split(df)
    scaler, pca, model = build_and_train(X_train, y_train)
    evaluate_model(scaler, pca, model, X_test, y_test)

    # Save artifacts
    joblib.dump(model, os.path.join(MODELS_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(pca, os.path.join(MODELS_DIR, "pca.pkl"))
    print("Saved model, scaler, pca to", MODELS_DIR)

if __name__ == "__main__":
    main()
