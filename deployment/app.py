from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="wide")

# Path relative to this file, so the app works from any working directory
MODEL_PATH = Path(__file__).parent / "best_credit_risk_XGBoost_model.pkl"

# Feature name -> (label, min, default, step, help)
FEATURES = {
    "rev_util":    ("Revolving utilization (ratio)", 0.0, 0.30, 0.01, "Credit card balance ÷ credit limit"),
    "age":         ("Age", 18, 45, 1, "Borrower age in years"),
    "late_30_59":  ("Times 30–59 days late", 0, 0, 1, "Past 2 years"),
    "debt_ratio":  ("Debt ratio", 0.0, 0.35, 0.01, "Monthly debt payments ÷ monthly income"),
    "monthly_inc": ("Monthly income", 0.0, 5000.0, 100.0, "Gross monthly income"),
    "open_credit": ("Open credit lines & loans", 0, 8, 1, ""),
    "late_90":     ("Times 90+ days late", 0, 0, 1, ""),
    "real_estate": ("Real-estate loans / lines", 0, 1, 1, ""),
    "late_60_89":  ("Times 60–89 days late", 0, 0, 1, "Past 2 years"),
    "dependents":  ("Number of dependents", 0, 0, 1, ""),
}


@st.cache_resource
def load_bundle():
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["model"]
    explainer = shap.TreeExplainer(pipe.named_steps["classifier"])
    return bundle, pipe, explainer


bundle, pipe, explainer = load_bundle()

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01,
                              help="Probability above which the applicant is flagged high risk")
st.sidebar.markdown("**Model info**")
st.sidebar.write(f"CV {bundle['metric']}: {bundle['best_cv_score']:.3f}")
st.sidebar.json(bundle["best_params"], expanded=False)

st.title("💳 Credit Risk Predictor")
tab_single, tab_batch = st.tabs(["Single applicant", "Batch (CSV)"])

# ---------------- Single prediction ----------------
with tab_single:
    with st.form("applicant"):
        cols = st.columns(2)
        values = {}
        for i, (name, (label, mn, default, step, hlp)) in enumerate(FEATURES.items()):
            with cols[i % 2]:
                values[name] = st.number_input(label, min_value=mn, value=default,
                                               step=step, help=hlp or None)
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        X = pd.DataFrame([values])[list(pipe.feature_names_in_)]
        proba = float(pipe.predict_proba(X)[0, 1])
        high_risk = proba >= threshold

        c1, c2 = st.columns(2)
        c1.metric("Probability of default", f"{proba:.1%}")
        if high_risk:
            c2.error("HIGH RISK")
        else:
            c2.success("LOW RISK")

        # SHAP explanation
        st.subheader("Why this prediction?")
        X_t = pipe.named_steps["preprocessor"].transform(X)
        sv = explainer(X_t)
        sv.feature_names = list(pipe.feature_names_in_)
        sv.data = X.values  # show raw input values on the plot
        fig = plt.figure()
        shap.plots.waterfall(sv[0], show=False)
        st.pyplot(fig, clear_figure=True)
        st.caption("Red pushes risk up, blue pushes it down (log-odds scale).")

# ---------------- Batch prediction ----------------
with tab_batch:
    st.write("Upload a CSV with these columns:")
    st.code(", ".join(pipe.feature_names_in_))
    file = st.file_uploader("CSV file", type="csv")
    if file:
        df = pd.read_csv(file)
        missing = set(pipe.feature_names_in_) - set(df.columns)
        if missing:
            st.error(f"Missing columns: {', '.join(sorted(missing))}")
        else:
            X = df[list(pipe.feature_names_in_)]
            df["default_probability"] = pipe.predict_proba(X)[:, 1]
            df["prediction"] = np.where(df["default_probability"] >= threshold, "High risk", "Low risk")
            st.dataframe(df, width="stretch")
            st.download_button("Download results", df.to_csv(index=False),
                               "predictions.csv", "text/csv")
