import streamlit as st
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔍",
    layout="wide",
)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model   = joblib.load("fraud_detection_model.pkl")
    scaler  = joblib.load("fraud_scaler.pkl")
    features = joblib.load("feature_columns.pkl")
    return model, scaler, features

model, scaler, feature_columns = load_artifacts()

# Scaler was fitted on [Amount, Time] – indices match scaler.mean_
# feature_columns order: ['Time', 'V1'…'V28', 'Amount']

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Credit Card Fraud Detection")
st.markdown(
    "Enter the transaction details below. The model will predict whether the "
    "transaction is **legitimate** or **fraudulent**."
)
st.divider()

# ── Sample transactions for quick testing ─────────────────────────────────────
SAMPLES = {
    "Normal Transaction": {
        "Time": 406.0, "Amount": 149.62,
        "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
        "V5": -0.3383, "V6": 0.4624, "V7": 0.2396, "V8": 0.0987,
        "V9": 0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
        "V13": -0.9913, "V14": -0.3112, "V15": 1.4682, "V16": -0.4704,
        "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
        "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
        "V25": 0.1285, "V26": -0.1891, "V27": 0.1336, "V28": -0.0211,
    },
    "Fraudulent Transaction": {
        "Time": 406.0, "Amount": 2.69,
        "V1": -2.3122, "V2": 1.9519, "V3": -1.6098, "V4": 3.9979,
        "V5": -0.5220, "V6": -1.4265, "V7": -2.5374, "V8": 1.3914,
        "V9": -2.7700, "V10": -2.7722, "V11": 3.2020, "V12": -2.8992,
        "V13": -0.5955, "V14": -4.2895, "V15": 0.3898, "V16": -1.1407,
        "V17": -2.8300, "V18": -0.0168, "V19": 0.4163, "V20": 0.3269,
        "V21": 0.7101, "V22": -0.5732, "V23": -0.5714, "V24": -0.1001,
        "V25": -0.1371, "V26": -0.0550, "V27": -0.0597, "V28": -0.0526,
    },
}

col_sample, _ = st.columns([2, 5])
with col_sample:
    selected_sample = st.selectbox("Load a sample transaction", ["(none)"] + list(SAMPLES.keys()))

defaults = SAMPLES.get(selected_sample, {})

def val(key, fallback=0.0):
    return float(defaults.get(key, fallback))

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Transaction Details")

col1, col2 = st.columns(2)
with col1:
    time_val   = st.number_input("Time (seconds elapsed since first transaction)", value=val("Time", 0.0), format="%.2f")
with col2:
    amount_val = st.number_input("Amount ($)", value=val("Amount", 0.0), min_value=0.0, format="%.2f")

st.subheader("Anonymised PCA Features (V1 – V28)")
st.caption("These are the PCA-transformed features from the original transaction data.")

v_cols = st.columns(4)
v_values = {}
for i, col in enumerate(feature_columns):
    if col.startswith("V"):
        idx = (int(col[1:]) - 1) % 4
        with v_cols[idx]:
            v_values[col] = st.number_input(col, value=val(col), format="%.4f", key=col)

# ── Predict ───────────────────────────────────────────────────────────────────
st.divider()
if st.button("Predict", type="primary", use_container_width=True):
    # Scale Amount and Time (scaler fitted on [Amount, Time])
    scaled = scaler.transform([[amount_val, time_val]])[0]
    scaled_amount, scaled_time = scaled[0], scaled[1]

    # Build feature array in original column order
    row = []
    for col in feature_columns:
        if col == "Time":
            row.append(scaled_time)
        elif col == "Amount":
            row.append(scaled_amount)
        else:
            row.append(v_values[col])

    X = np.array(row).reshape(1, -1)
    prediction  = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    fraud_prob  = probability[1] * 100
    legit_prob  = probability[0] * 100

    st.subheader("Prediction Result")
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        if prediction == 1:
            st.error("⚠️ FRAUDULENT TRANSACTION")
        else:
            st.success("✅ LEGITIMATE TRANSACTION")

    with res_col2:
        st.metric("Fraud Probability",    f"{fraud_prob:.2f}%")

    with res_col3:
        st.metric("Legitimate Probability", f"{legit_prob:.2f}%")

    # Confidence bar
    st.progress(fraud_prob / 100, text=f"Fraud confidence: {fraud_prob:.1f}%")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: Random Forest Classifier  |  Dataset: Credit Card Fraud Detection (Kaggle)")
