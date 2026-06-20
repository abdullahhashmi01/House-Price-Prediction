import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4ecf7 100%);
    }

    .main-header {
        background: linear-gradient(135deg, #6a3fd8 0%, #a64bd8 50%, #ff6f91 100%);
        padding: 36px 40px;
        border-radius: 18px;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px rgba(106, 63, 216, 0.25);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 36px;
        font-weight: 800;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin-top: 6px;
        font-size: 16px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2b1055 0%, #6a3fd8 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    .section-card {
        background: white;
        border-radius: 16px;
        padding: 24px 28px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        margin-bottom: 20px;
        border: 1px solid #eef0f6;
    }
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #2b1055;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #ff6f91, #ff9a6c);
        color: white;
        font-weight: 700;
        font-size: 17px;
        padding: 12px 0;
        border-radius: 12px;
        border: none;
        box-shadow: 0 6px 16px rgba(255, 111, 145, 0.4);
        transition: transform 0.15s ease;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-2px);
    }

    div[data-testid="stMetricValue"] {
        color: #1e7a4a;
        font-size: 42px !important;
        font-weight: 800;
    }

    .result-card {
        background: linear-gradient(135deg, #d8f3dc, #b7e4c7);
        border-radius: 18px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 8px 22px rgba(30, 122, 74, 0.2);
        margin-top: 10px;
    }
    .footer-note {
        text-align: center;
        color: #888;
        font-size: 13px;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load model & preprocessor
# ---------------------------------------------------------
MODEL_PATH = os.path.join('artifacts', 'model.pkl')
PREPROCESSOR_PATH = os.path.join('artifacts', 'preprocessor.pkl')


@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    return model, preprocessor


try:
    model, preprocessor = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    load_error = str(e)

NUM_FEATURES = ['area_sqft', 'BHK', 'floor_num', 'total_floors',
                'Bathroom', 'Balcony', 'car_parking_num']
CAT_FEATURES = ['location_grp', 'Status', 'Transaction', 'Furnishing', 'facing']

LOCATIONS = [
    'Other', 'Sector 1', 'Sector 2', 'Sector 3', 'Andheri', 'Bandra',
    'Whitefield', 'Koramangala', 'Gachibowli', 'Dwarka', 'Noida Sector 62'
]
STATUS_OPTIONS = ['Ready to Move', 'Under Construction']
TRANSACTION_OPTIONS = ['New Property', 'Resale']
FURNISHING_OPTIONS = ['Furnished', 'Semi-Furnished', 'Unfurnished']
FACING_OPTIONS = ['East', 'West', 'North', 'South',
                   'North-East', 'North-West', 'South-East', 'South-West']

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏠 About")
    st.write(
        "This app estimates house prices using a **LightGBM** regression model "
        "trained on cleaned property listing data."
    )
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.markdown("- **R²:** 0.929\n- **MAE:** ~0.11 (log scale)\n- **RMSE:** ~0.21 (log scale)")
    st.markdown("---")
    st.markdown("### 🛠 Tech Stack")
    st.markdown("`Python` · `LightGBM` · `Scikit-learn` · `Streamlit`")

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🏠 House Price Predictor</h1>
    <p>Get an instant price estimate for any property based on its details</p>
</div>
""", unsafe_allow_html=True)

if not artifacts_loaded:
    st.error(f"⚠️ Could not load model/preprocessor: {load_error}")
    st.stop()

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
with st.form("prediction_form"):

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📐 Property Size & Layout</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        area_sqft = st.number_input("Area (sqft)", min_value=100.0, max_value=10000.0, value=1200.0, step=50.0)
    with col2:
        bhk = st.number_input("BHK", min_value=1, max_value=10, value=3, step=1)
    with col3:
        floor_num = st.number_input("Floor Number", min_value=0, max_value=100, value=4, step=1)
    with col4:
        total_floors = st.number_input("Total Floors", min_value=1, max_value=100, value=12, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🚿 Amenities</div>', unsafe_allow_html=True)
    col5, col6, col7 = st.columns(3)
    with col5:
        bathroom = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
    with col6:
        balcony = st.number_input("Balconies", min_value=0, max_value=10, value=1, step=1)
    with col7:
        car_parking = st.number_input("Car Parking Slots", min_value=0, max_value=10, value=1, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📍 Location & Property Info</div>', unsafe_allow_html=True)
    col8, col9, col10 = st.columns(3)
    with col8:
        location_grp = st.selectbox("Location", LOCATIONS)
        status = st.selectbox("Status", STATUS_OPTIONS)
    with col9:
        transaction = st.selectbox("Transaction", TRANSACTION_OPTIONS)
        furnishing = st.selectbox("Furnishing", FURNISHING_OPTIONS)
    with col10:
        facing = st.selectbox("Facing", FACING_OPTIONS)
    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("✨ Predict Price", use_container_width=True)

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:
    try:
        input_dict = {
            'area_sqft': area_sqft,
            'BHK': bhk,
            'floor_num': floor_num,
            'total_floors': total_floors,
            'Bathroom': bathroom,
            'Balcony': balcony,
            'car_parking_num': car_parking,
            'location_grp': location_grp,
            'Status': status,
            'Transaction': transaction,
            'Furnishing': furnishing,
            'facing': facing,
        }

        input_df = pd.DataFrame([input_dict])[NUM_FEATURES + CAT_FEATURES]

        transformed = preprocessor.transform(input_df)
        log_price_pred = model.predict(transformed)[0]
        price_pred = np.expm1(log_price_pred)

        st.markdown(f"""
        <div class="result-card">
            <h3 style="color:#1e7a4a; margin-bottom:6px;">🎉 Estimated Price</h3>
            <div style="font-size:44px; font-weight:800; color:#1e7a4a;">₹ {price_pred:,.0f}</div>
            <p style="color:#3a5a40; margin-top:8px;">Based on {bhk} BHK · {area_sqft:.0f} sqft · {location_grp}</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 View Input Details"):
            st.json(input_dict)

    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.markdown('<p class="footer-note">Built with ❤️ using Streamlit · LightGBM model</p>', unsafe_allow_html=True)