# main.py — Streamlit UI (polished result + probability bar)
import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np

# ---------------- Page config ----------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("Telecom Customer Churn Predictor 🔮")
st.write("Quick, real-world demo: same preprocessing as training → consistent predictions.")

# ---------------- Paths (adjust if needed) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSOR_PKL = os.path.join(BASE_DIR, "Model", "preprocessor.pkl")
MODEL_PKL = os.path.join(BASE_DIR, "Model", "model.pkl")
FEATURE_COLS_PKL = os.path.join(BASE_DIR, "Model", "feature_columns.pkl")  # optional

# ---------------- Load artifacts ----------------
if not os.path.exists(PREPROCESSOR_PKL) or not os.path.exists(MODEL_PKL):
    st.error("Missing model artifacts. Put preprocessor.pkl and model.pkl inside the Model/ folder.")
    st.stop()

with open(PREPROCESSOR_PKL, "rb") as f:
    preprocessor = pickle.load(f)
with open(MODEL_PKL, "rb") as f:
    model = pickle.load(f)

# ---------------- Sidebar inputs ----------------
st.sidebar.header("Customer Input")
def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    partner = st.sidebar.selectbox("Partner", ("Yes", "No"))
    dependents = st.sidebar.selectbox("Dependents", ("Yes", "No"))
    phone_service = st.sidebar.selectbox("Phone Service", ("Yes", "No"))
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ("Yes", "No", "No phone service"))
    internet_service = st.sidebar.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
    online_security = st.sidebar.selectbox("Online Security", ("Yes", "No", "No internet service"))
    online_backup = st.sidebar.selectbox("Online Backup", ("Yes", "No", "No internet service"))
    device_protection = st.sidebar.selectbox("Device Protection", ("Yes", "No", "No internet service"))
    tech_support = st.sidebar.selectbox("Tech Support", ("Yes", "No", "No internet service"))
    streaming_tv = st.sidebar.selectbox("Streaming TV", ("Yes", "No", "No internet service"))
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ("Yes", "No", "No internet service"))
    contract = st.sidebar.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ("Yes", "No"))
    payment_method = st.sidebar.selectbox("Payment Method", (
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ))
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 0.0, 150.0, 70.0)
    total_charges_default = float(monthly_charges * tenure)
    total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 10000.0, total_charges_default)
    senior = st.sidebar.selectbox("Senior Citizen", ("No", "Yes"))

    data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ---------------- Preprocess (match training) ----------------
def apply_training_style_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Contract -> numeric months (match notebook)
    contract_map = {"Month-to-month": 1, "One year": 12, "Two year": 24}
    if "Contract" in df.columns:
        df["Contract"] = df["Contract"].map(contract_map).astype(float)
    # gender label-encode used in training: Female=0, Male=1
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Female": 0, "Male": 1}).astype(float)
    # ensure numeric
    for col in ["MonthlyCharges", "TotalCharges", "tenure", "SeniorCitizen", "Contract"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    # engineered features (must match training)
    if "CLV" not in df.columns:
        df["CLV"] = df["tenure"] * df["MonthlyCharges"]
    if "Tenure_Contract" not in df.columns:
        df["Tenure_Contract"] = df["tenure"] * df["Contract"]
    return df

# ---------------- Small helper to render big colored banner --------------
def show_result_banner(is_churn: bool, prob: float | None, threshold: float):
    if is_churn:
        st.markdown(f"<h2 style='color:#b00020;'>❌ Customer likely to CHURN — risk ≥ {threshold:.2f}</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color:#0b6623;'>✅ Customer likely to STAY — risk < {threshold:.2f}</h2>", unsafe_allow_html=True)
    if prob is not None:
        # big numeric metric
        st.metric(label="Churn probability", value=f"{prob:.2%}")
        # probability bar (visual)
        bar_df = pd.DataFrame({"probability":[prob]})
        st.bar_chart(bar_df.T)  # single-column bar chart (simple visual)
    else:
        st.info("Model does not provide probability (predict_proba unavailable).")

# ---------------- Main prediction flow ----------------
st.subheader("Customer Input Summary")
st.write(input_df)  # user sees the raw inputs first

threshold = st.slider("Decision threshold for classifying 'Churn'", 0.0, 1.0, 0.5, 0.01)

if st.button("Predict Churn"):
    # 1) Apply notebook-style preprocessing
    model_inputs = apply_training_style_preprocessing(input_df)
    st.write("After training-style preprocessing:")
    st.write(model_inputs)

    # 2) Transform via preprocessor
    try:
        X_trans = preprocessor.transform(model_inputs)
        X_arr = np.asarray(X_trans)
        st.write("preprocessor.transform output shape:", X_arr.shape, "dtype:", X_arr.dtype)
    except Exception as e:
        st.error("preprocessor.transform() failed. Check preprocessor/model alignment.")
        st.exception(e)
        st.stop()

    # 3) Ensure numeric array
    try:
        if not np.issubdtype(X_arr.dtype, np.number):
            X_arr = X_arr.astype(float)
    except Exception as e:
        st.error("Transformed output cannot be converted to numeric array.")
        st.exception(e)
        st.stop()

    # 4) Predict
    try:
        pred = model.predict(X_arr)
        proba = model.predict_proba(X_arr)[:, 1] if hasattr(model, "predict_proba") else None
    except Exception as e:
        st.error("Model prediction failed.")
        st.exception(e)
        st.stop()

    # 5) Determine churn flag using threshold
    if proba is not None:
        churn_prob = float(proba[0])
        is_churn = 1 if churn_prob >= threshold else 0
    else:
        churn_prob = None
        try:
            is_churn = int(pred[0])
        except Exception:
            is_churn = 1 if str(pred[0]).lower().startswith("y") else 0

    # 6) Append result into same table and show
    display_df = input_df.copy()
    display_df["Predicted_Churn"] = ["Yes" if is_churn == 1 else "No"]
    if churn_prob is not None:
        display_df["Churn_Prob"] = [f"{churn_prob:.2%}"]
    st.subheader("Input + Prediction")
    st.write(display_df)

    # 7) Friendly banner + probability visual
    show_result_banner(is_churn == 1, churn_prob, threshold)

    # 8) Optional debug block (collapsed)
    with st.expander("Debug: model internals (show/hide)"):
        st.write("Transformed numeric vector (first row):")
        st.write(X_arr[0][:100])  # show first 100 values
        if hasattr(preprocessor, "get_feature_names_out"):
            try:
                fn = preprocessor.get_feature_names_out()
                feat_df = pd.DataFrame([X_arr[0]], columns=fn)
                st.write("Top features (after transform) — first 60:")
                st.write(feat_df.T.head(60))
            except Exception as e:
                st.write("Can't get feature names:", e)
        # show model coefficients / importances if available
        if hasattr(model, "coef_"):
            st.write("Model coef_ (first 60):")
            st.write(model.coef_.ravel()[:60])
        elif hasattr(model, "feature_importances_"):
            st.write("Model feature_importances_ (first 60):")
            st.write(model.feature_importances_[:60])

# Footer
st.markdown("---")
st.caption("UX: single-row input -> predict -> appended results, with a probability bar and threshold control. Model behavior depends on training quality; if prob looks extreme, retrain/calibrate the model.")
