# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib # pyright: ignore[reportMissingImports]
from PIL import Image

# --- Page Config ---
st.set_page_config(
    page_title="Corrosion Rate Prediction App",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Beauty ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 2rem;
    }
    h1, h2, h3 {
        color: #0f4c75;
    }
    .stButton>button {
        color: white;
        background-color: #1b262c;
        border-radius: 12px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🧪 Corrosion Prediction App")
st.markdown("### Predict the **Thickness Loss (mm)** due to corrosion using your trained ML model.")

# --- Load Model ---

import sklearn.compose._column_transformer as ct

class _RemainderColsList(list):
    pass

ct._RemainderColsList = _RemainderColsList

import joblib
model = joblib.load(r"C:\Users\HP\streamlit_app\tuned_elasticnet_model.pkl")





@st.cache_resource
def load_model():
    model = joblib.load(r"C:\Users\HP\streamlit_app\tuned_elasticnet_model.pkl")
    return model

model = load_model()



# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a **machine learning algorithm model** trained to predict **thickness loss (mm)** "
    "in pipelines or metallic structures. Developed by **Ogunmakinju Thomas**, Corrosion & Data Engineer."
)
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/3a/Corrosion_pipes.jpg", use_container_width=True)

# --- Input Section ---
st.header("Enter Corrosion Parameters")

# Example input fields (you can adjust names and ranges as in your dataset)
Thickness_mm = st.number_input("Thickness (mm)", min_value=3.00, max_value=30.00, value=20.00)
Temperature_C = st.number_input("Temperature_C", min_value=-30.00, max_value=120.00, value=50.00)
Grade = st.selectbox("Material Type", ["A", "B", "C", "D","E"])
Condition = st.selectbox("Condition", ['Normal', 'Critical', 'Moderate'])
Material_Loss_Percent = st.number_input("Material_Loss_Percent (percent)", min_value=0.0, max_value=100.0, value=50.0)
Pipe_Size_mm = st.number_input("Pipe_Size_mm", min_value=53.00, max_value=1499.00, value=60.00)
Strength_MPa = st.number_input("Strength_MPa", min_value=201.16, max_value=799.52, value=250.00)
Diameter_mm = st.number_input("Diameter_mm", min_value=104.00, max_value=1999.00, value=1013.07200)
Corrosion_Impact_Percent = st.number_input("Corrosion_Impact_Percent", min_value=0.01, max_value=19.98, value=10.00)
Time_Years = st.number_input("Time_Years", min_value=1.00, max_value=24.00, value=10.00)
Max_Pressure_Bar = st.number_input("Max_Pressure_Bar", min_value=10.48, max_value=199.91, value=105.00)
material_type = st.selectbox("Material Type", ['Copper', 'Cast Iron', 'Aluminum', 'Steel', 'PVC'])

# --- Prediction Button ---
if st.button("🔍 Predict Corrosion Thickness Loss"):
    # Convert input into dataframe
    input_data = pd.DataFrame({
        "Temperature_C": [Temperature_C],
        "Grade": [Grade],
        "Condition": [Condition],
        "Material_Loss_Percent": [Material_Loss_Percent],
        "Thickness_mm": [Thickness_mm],
        "Pipe_Size_mm": [Pipe_Size_mm],
        "Strength_MPa": [Strength_MPa],
        "Diameter_mm": [Diameter_mm],
        "Corrosion_Impact_Percent": [Corrosion_Impact_Percent],
        "Time_Years": [Time_Years],
        "Max_Pressure_Bar": [Max_Pressure_Bar],
        "Material": [material_type]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.success(f"### 🧠 Predicted Thickness Loss: **{prediction:.3f} mm**")

    # Optional insights
    if prediction > 5:
        st.error("⚠️ High corrosion detected — consider inhibitors or protective coatings.")
    elif prediction > 2:
        st.warning("⚠️ Moderate corrosion — inspect periodically.")
    else:
        st.info("✅ Low corrosion risk — system appears stable.")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<center>Developed by <b>Ogunmakinju Thomas</b> | Materials & Corrosion Engineer | AI/ML Researcher</center>",
    unsafe_allow_html=True
)

