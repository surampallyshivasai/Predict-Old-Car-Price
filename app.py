"""
app.py
------
Streamlit web application for Old Car Price Prediction.
Run with: streamlit run app.py
"""

import os
import sys
import json
import numpy as np
import streamlit as st
import joblib

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚗 Old Car Price Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(BASE_DIR, "models")
MODEL_PATH    = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH   = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")
META_PATH     = os.path.join(MODEL_DIR, "model_metadata.json")

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global font & background ───────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero banner ────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.hero-banner h1 {
    color: #e0e6ff;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 0.4rem;
    letter-spacing: -0.5px;
}
.hero-banner p {
    color: #a0b4d0;
    font-size: 1rem;
    margin: 0;
}

/* ── Section headers ────────────────────────────────────────────── */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #4a90d9;
    border-left: 4px solid #4a90d9;
    padding-left: 10px;
    margin: 1.5rem 0 0.8rem;
}

/* ── Prediction result card ─────────────────────────────────────── */
.result-card {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    border: 1px solid #4a90d9;
    box-shadow: 0 4px 20px rgba(74,144,217,0.25);
    margin-top: 1.2rem;
}
.result-card .price-label {
    color: #a0b4d0;
    font-size: 0.95rem;
    margin-bottom: 0.4rem;
    font-weight: 500;
}
.result-card .price-value {
    color: #4dd0e1;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -1px;
}
.result-card .price-unit {
    color: #a0b4d0;
    font-size: 1rem;
}

/* ── Metric tiles ───────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: #16213e;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    border: 1px solid #2a3a5e;
}

/* ── Input field labels ─────────────────────────────────────────── */
label { font-weight: 500; }

/* ── Sidebar ────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0d1b2a !important;
}
section[data-testid="stSidebar"] * { color: #ccd6f6 !important; }

/* ── Divider ────────────────────────────────────────────────────── */
hr { border-color: #2a3a5e; }
</style>
""", unsafe_allow_html=True)


# ─── Load artefacts (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    """Load and cache ML model, scaler, and encoders."""
    missing = []
    for p, n in [(MODEL_PATH, "best_model.pkl"),
                 (SCALER_PATH, "scaler.pkl"),
                 (ENCODERS_PATH, "label_encoders.pkl")]:
        if not os.path.exists(p):
            missing.append(n)
    if missing:
        return None, None, None, None

    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    meta     = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}
    return model, scaler, encoders, meta


# ─── Hero banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🚗 Old Car Price Predictor</h1>
    <p>AI-powered used car valuation · Trained on real market data</p>
</div>
""", unsafe_allow_html=True)

# ─── Load model ───────────────────────────────────────────────────────────────
model, scaler, encoders, meta = load_artifacts()

if model is None:
    st.error("⚠️  Model files not found. Please run `python src/train.py` first to train the model.")
    st.info("**Steps to train:**\n1. `pip install -r requirements.txt`\n2. `python src/train.py`\n3. `streamlit run app.py`")
    st.stop()

# ─── Sidebar: Model info ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Model Information")
    if meta:
        st.metric("Best Model",  meta.get("best_model", "N/A"))
        st.metric("R² Score",    f"{meta.get('r2', 0):.4f}")
        st.metric("MAE (₹ L)",   f"{meta.get('mae', 0):.4f}")
        st.metric("RMSE (₹ L)",  f"{meta.get('rmse', 0):.4f}")
    st.markdown("---")
    st.markdown("### 📋 How to use")
    st.markdown(
        "1. Fill in the car details on the right  \n"
        "2. Click **Predict Price**  \n"
        "3. Get an instant AI-powered estimate"
    )
    st.markdown("---")
    st.markdown("### 🔧 Fuel Types")
    st.markdown("- **Petrol** – Standard ICE  \n- **Diesel** – Better mileage  \n- **CNG** – Eco-friendly  \n- **Electric** – Zero emission")

# ─── Main form ────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">🔍 Enter Car Details</p>', unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input(
            "📅 Manufacturing Year",
            min_value=1995, max_value=2024, value=2017, step=1,
            help="Year in which the car was manufactured"
        )
        present_price = st.number_input(
            "💵 Present Price (₹ Lakhs)",
            min_value=0.5, max_value=100.0, value=9.85, step=0.1,
            help="Current ex-showroom price of the car model"
        )
        kms_driven = st.number_input(
            "🛣️ Kilometres Driven",
            min_value=500, max_value=500000, value=22000, step=500,
            help="Total kilometres the car has been driven"
        )
        owner = st.selectbox(
            "👤 Number of Previous Owners",
            options=[0, 1, 2, 3],
            format_func=lambda x: ["0 – First Owner", "1 – Second Owner",
                                    "2 – Third Owner", "3 – Fourth+ Owner"][x],
            help="How many owners has this car had before you?"
        )

    with col2:
        # Extract known classes from encoders for dropdowns
        fuel_classes   = list(encoders["Fuel_Type"].classes_)
        seller_classes = list(encoders["Seller_Type"].classes_)
        trans_classes  = list(encoders["Transmission"].classes_)

        fuel_type = st.selectbox(
            "⛽ Fuel Type",
            options=fuel_classes,
            help="Type of fuel the car runs on"
        )
        seller_type = st.selectbox(
            "🏢 Seller Type",
            options=seller_classes,
            help="Is the seller a Dealer or an Individual?"
        )
        transmission = st.selectbox(
            "⚙️ Transmission",
            options=trans_classes,
            help="Manual or Automatic gearbox"
        )

        # Derived display info
        car_age = 2026 - year
        st.info(f"📆 **Car Age:** {car_age} year{'s' if car_age != 1 else ''}")

    # ── Submit button ──────────────────────────────────────────────────────
    predict_btn = st.form_submit_button(
        "🔮 Predict Selling Price",
        use_container_width=True,
        type="primary"
    )

# ─── Prediction logic ─────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Calculating estimated price…"):
        try:
            # Encode categoricals
            fuel_enc   = encoders["Fuel_Type"].transform([fuel_type])[0]
            seller_enc = encoders["Seller_Type"].transform([seller_type])[0]
            trans_enc  = encoders["Transmission"].transform([transmission])[0]

            # Build feature vector (must match training column order)
            car_age = 2026 - year
            features = np.array([[present_price, kms_driven, fuel_enc,
                                   seller_enc, trans_enc, owner, car_age]])

            # Scale & predict
            features_scaled = scaler.transform(features)
            predicted_price = model.predict(features_scaled)[0]
            predicted_price = max(predicted_price, 0.0)

            # ── Display result ────────────────────────────────────────────
            st.markdown(f"""
            <div class="result-card">
                <div class="price-label">🎯 Estimated Selling Price</div>
                <div class="price-value">₹ {predicted_price:.2f}</div>
                <div class="price-unit">Lakhs</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Quick insight metrics ─────────────────────────────────────
            st.markdown('<br>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            depreciation = ((present_price - predicted_price) / present_price * 100) if present_price > 0 else 0
            m1.metric("Predicted Price", f"₹{predicted_price:.2f}L")
            m2.metric("Present Price",   f"₹{present_price:.2f}L")
            m3.metric("Depreciation",    f"{depreciation:.1f}%",
                      delta=f"-₹{present_price - predicted_price:.2f}L",
                      delta_color="inverse")
            m4.metric("Car Age",         f"{car_age} yrs")

            # ── Confidence note ───────────────────────────────────────────
            st.success(
                f"✅ Prediction generated using **{meta.get('best_model', 'ML Model')}** "
                f"(R² = {meta.get('r2', 0):.3f})"
            )

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#607080; font-size:0.82rem;'>"
    "Old Car Price Predictor · Built with Python & Streamlit · "
    "For educational purposes only</div>",
    unsafe_allow_html=True
)
