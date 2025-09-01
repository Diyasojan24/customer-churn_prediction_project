# main.py — Streamlit UI (polished result + probability bar)
import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ---------------- Page config ----------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("Telecom Customer Churn Predictor 🔮")
st.write(
    "This interactive web app predicts customer churn based on their information. Please input the customer's details on the left sidebar to get a prediction.")

# ---------------- Paths (adjust if needed) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSOR_PKL = os.path.join(BASE_DIR, "Model", "preprocessor.pkl")
MODEL_PKL = os.path.join(BASE_DIR, "Model", "model.pkl")

# ---------------- Load artifacts ----------------
try:
    with open(PREPROCESSOR_PKL, "rb") as f:
        preprocessor = pickle.load(f)
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Missing model artifacts. Please ensure 'preprocessor.pkl' and 'model.pkl' are in the 'Model/' folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

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
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ("No", "Yes"))

    data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
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


# ---------------- Small helper to render big colored banner --------------
def show_result_banner(is_churn: bool, prob: float, threshold: float):
    if is_churn:
        st.markdown(f"""
            <div style='background-color: #ffcccc; padding: 10px; border-radius: 5px;'>
                <h2 style='color:#b00020;'>❌ Customer is likely to CHURN (Risk ≥ {threshold:.0%})</h2>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='background-color: #ccffcc; padding: 10px; border-radius: 5px;'>
                <h2 style='color:#0b6623;'>✅ Customer is likely to STAY (Risk < {threshold:.0%})</h2>
            </div>
            """, unsafe_allow_html=True)

    # Big numeric metric for churn probability
    st.metric(label="Churn Probability", value=f"{prob:.2%}")

    # Probability bar (visual) using st.progress
    st.progress(prob)


# ---------------- Main prediction flow ----------------
st.subheader("Customer Input Summary")
st.write(input_df)

threshold = st.slider("Decision threshold for classifying 'Churn'", 0.0, 1.0, 0.5, 0.01)

if st.button("Predict Churn"):
    # Create a copy to avoid modifying the displayed summary dataframe
    df_to_process = input_df.copy()

    # --- Replicate Feature Engineering from the Notebook ---
    # The preprocessor expects these columns to exist before it runs.

    # 1. Map 'Contract' to numeric values. This REPLACES the text column.
    contract_map = {"Month-to-month": 1, "One year": 12, "Two year": 24}
    df_to_process['Contract'] = df_to_process['Contract'].map(contract_map)

    # 2. Map 'gender' to numeric.
    gender_map = {"Male": 1, "Female": 0}
    df_to_process['gender'] = df_to_process['gender'].map(gender_map)

    # 3. Create the engineered features using the now-numeric columns.
    df_to_process['Tenure_Contract'] = df_to_process['tenure'] * df_to_process['Contract']
    df_to_process['CLV'] = df_to_process['tenure'] * df_to_process['MonthlyCharges']

    # 4) Transform via preprocessor
    try:
        X_trans = preprocessor.transform(df_to_process)
    except Exception as e:
        st.error(
            "There was an error during the data preprocessing step. Please check the input values and ensure they are consistent with the model's training data.")
        st.exception(e)
        st.stop()

    # 5) Predict
    try:
        proba = model.predict_proba(X_trans)[:, 1]
    except Exception as e:
        st.error("The model failed to make a prediction. Please check the model file and input data.")
        st.exception(e)
        st.stop()

    # 6) Determine churn flag using threshold
    churn_prob = float(proba[0])
    is_churn = churn_prob >= threshold

    # 7) Append result into same table and show
    display_df = input_df.copy()
    display_df["Predicted_Churn"] = "Yes" if is_churn else "No"
    display_df["Churn_Prob"] = f"{churn_prob:.2%}"
    st.subheader("Input + Prediction")
    st.write(display_df)

    # 8) Friendly banner + probability visual
    show_result_banner(is_churn, churn_prob, threshold)

    # 9) Optional debug block (collapsed)
    with st.expander("Debug: Model Internals (Show/Hide)"):
        st.write("Data after feature engineering (before scaling/encoding):")
        st.write(df_to_process)
        st.write("Transformed data (first 20 features):")

        # --- CORRECTED DEBUG CODE ---
        # Check if the transformed data is a sparse matrix or a dense numpy array
        if hasattr(X_trans, "toarray"):
            # If it's sparse, convert it to a dense array to display
            st.write(X_trans.toarray()[0, :20])
        else:
            # If it's already a dense array, display it directly
            st.write(X_trans[0, :20])

        st.write("Model Coefficients (first 20):")
        if hasattr(model, "coef_"):
            st.write(model.coef_[0, :20])
        elif hasattr(model, "feature_importances_"):
            st.write("Model Feature Importances (first 20):")
            st.write(model.feature_importances_[:20])

# Footer
st.markdown("---")
st.caption(
    "This app uses a machine learning model to predict customer churn. The model's behavior depends on the quality of the training data; if the probability seems extreme, the model may need to be retrained or calibrated.")

