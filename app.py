# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt  # <-- ensure matplotlib imported

st.set_page_config(page_title="🛡️ Credit Card Fraud Detection", layout="wide", initial_sidebar_state="expanded")

# --- CSS to make it aesthetic ---
def local_css():
    st.markdown(
        """
        <style>
        .reportview-container {
            background: linear-gradient(180deg, #0f172a 0%, #0f172a 50%, #071032 100%);
            color: #E6EEF8;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #071032, #0b1b2b);
            color: white;
        }
        .stButton>button {
            border-radius: 10px;
        }
        .big-font { font-size:24px !important; color:#fff; font-weight:600; }
        .kpi { background: rgba(255,255,255,0.03); padding: 12px; border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True
    )
local_css()

# Paths
MODEL_DIR = "models"
MODEL_FPATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_FPATH = os.path.join(MODEL_DIR, "scaler.pkl")
PCA_FPATH = os.path.join(MODEL_DIR, "pca.pkl")
DATA_FPATH = "data/creditcard.csv"

# Header
col1, col2 = st.columns([0.9, 0.1])
with col1:
    st.title("🛡️ Credit Card Fraud Detection — Interactive Dashboard")
with col2:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=60)

# Sidebar controls
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload your creditcard.csv (optional)", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample dataset from data/creditcard.csv", value=not bool(uploaded))
predict_single = st.sidebar.expander("Single transaction prediction (manual input)")

# Load data
@st.cache_data
def load_data(path=DATA_FPATH):
    return pd.read_csv(path)

if uploaded:
    df = pd.read_csv(uploaded)
elif use_sample and os.path.exists(DATA_FPATH):
    df = load_data(DATA_FPATH)
else:
    st.warning("Please upload a dataset or put 'data/creditcard.csv' in the right folder.")
    st.stop()

# Quick preprocessing for UI visuals
if "Class" not in df.columns:
    st.error("Dataset must have a 'Class' column with 0=legit, 1=fraud.")
    st.stop()

df["Amount_log"] = np.log1p(df["Amount"])
display_df = df.sample(min(5000, len(df)), random_state=42)  # small sample for faster charts

# Top KPIs
total_trans = len(df)
frauds = int(df["Class"].sum())
fraud_rate = frauds / total_trans * 100
col1, col2, col3 = st.columns([1,1,1])
col1.markdown(f"<div class='kpi'><div class='big-font'>{total_trans:,}</div><div style='color:#9AA9BF'>Total transactions</div></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='kpi'><div class='big-font'>{frauds}</div><div style='color:#9AA9BF'>Frauds</div></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='kpi'><div class='big-font'>{fraud_rate:.4f}%</div><div style='color:#9AA9BF'>Fraud rate</div></div>", unsafe_allow_html=True)

st.markdown("---")

# Interactive graphs row
c1, c2 = st.columns([1,1])
with c1:
    st.subheader("Transaction Amount Distribution")
    fig1 = px.histogram(display_df, x="Amount_log", nbins=80, title="Log(Amount) distribution (sample)")
    st.plotly_chart(fig1, use_container_width=True)
with c2:
    st.subheader("Fraud vs Legit: Amount by Class")
    fig2 = px.box(display_df, x="Class", y="Amount_log", points="outliers",
                  labels={"Class":"Class (0 legit, 1 fraud)", "Amount_log":"Log(Amount)"},
                  title="Amount (log) by class")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Load model artifacts if available
model_ready = os.path.exists(MODEL_FPATH) and os.path.exists(SCALER_FPATH) and os.path.exists(PCA_FPATH)
if not model_ready:
    st.info("Model not found in 'models/'. Train a model using train_model.py and place model.pkl, scaler.pkl, pca.pkl in models/.")
else:
    model = joblib.load(MODEL_FPATH)
    scaler = joblib.load(SCALER_FPATH)
    pca = joblib.load(PCA_FPATH)

    # Evaluate model on a subset for UI - compute ROC AUC quickly
    st.subheader("Model performance (quick stats on sample)")
    sample = df.sample(min(5000, len(df)), random_state=1)
    X = sample.drop(columns=["Class", "Amount", "Time"], errors="ignore")
    X["Amount_log"] = np.log1p(sample["Amount"])
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    y = sample["Class"].values
    y_proba = model.predict_proba(X_pca)[:,1]
    auc = roc_auc_score(y, y_proba)
    st.metric("ROC AUC (sample)", f"{auc:.4f}")

    # ROC (approx via binned thresholds) and PR curve
    st.markdown("### ROC-like and PR plots")
    from sklearn.metrics import precision_recall_curve, roc_curve
    fpr, tpr, _ = roc_curve(y, y_proba)
    pr_precision, pr_recall, _ = precision_recall_curve(y, y_proba)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    roc_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    roc_fig.update_layout(xaxis_title="FPR", yaxis_title="TPR", height=350)
    col1, col2 = st.columns(2)
    col1.plotly_chart(roc_fig, use_container_width=True)

    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=pr_recall, y=pr_precision, mode="lines", name="PR"))
    pr_fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=350)
    col2.plotly_chart(pr_fig, use_container_width=True)

    st.markdown("---")

    # PCA scatter (2D) for visualization
    st.subheader("PCA scatter (2 components)")
    pca2 = PCA(n_components=2, random_state=42)
    X2 = scaler.transform(df.drop(columns=["Class", "Amount", "Time"], errors="ignore").assign(Amount_log=np.log1p(df["Amount"])))
    pca2_proj = pca2.fit_transform(X2)
    pca_df = pd.DataFrame(pca2_proj, columns=["PC1","PC2"])
    pca_df["Class"] = df["Class"].values
    fig_pca = px.scatter(pca_df.sample(min(3000,len(pca_df)), random_state=42), x="PC1", y="PC2", color="Class",
                         title="PCA 2D projection (sample)", opacity=0.7)
    st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("---")

    # Single transaction prediction UI + SHAP
    st.subheader("Single transaction prediction")
    st.write("Fill in feature values to get a prediction and SHAP explanation.")
    input_cols = [c for c in X.columns]
    # build simple form
    with st.form("single_pred"):
        user_inputs = {}
        for feat in input_cols:
            user_inputs[feat] = st.number_input(feat, value=float(X[feat].median()), format="%.6f")
        submitted = st.form_submit_button("Predict")

    if submitted:
        inp_df = pd.DataFrame([user_inputs])
        inp_scaled = scaler.transform(inp_df)
        inp_pca = pca.transform(inp_scaled)
        proba = model.predict_proba(inp_pca)[0,1]
        st.metric("Fraud probability", f"{proba*100:.2f}%")

        # SHAP explainability (fixed)
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(inp_pca)
            st.subheader("SHAP values (feature contributions)")

            shap.initjs()

            # Create a Matplotlib figure safely
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.summary_plot(shap_vals, inp_pca, plot_type="bar", show=False)
            st.pyplot(fig)  # ✅ Explicit figure passed
            plt.close(fig)  # ✅ prevent memory leak
        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")

st.markdown("---")
st.caption("Built with ❤️. Train model with `python train_model.py`, then run UI with `streamlit run app.py`.")
