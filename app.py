import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Smart Rainfall Prediction", page_icon="🌧️")

st.title("🌧️ Smart Rainfall Prediction System")
st.write("Predict whether it will rain tomorrow based on weather metrics using Machine Learning models.")

# Load Scaler and Feature Names
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Load Models
models = {
    "Member 1: KNN": joblib.load("models/knn_model.pkl"),
    "Member 2: SVM": joblib.load("models/svm_model.pkl"),
    "Member 3: ANN": joblib.load("models/ann_model.pkl"),
}

selected_model_name = st.selectbox("Select Machine Learning Model", list(models.keys()))
model = models[selected_model_name]

st.sidebar.header("Input Weather Parameters")

# Sidebar Inputs
location = st.sidebar.selectbox("Location", ["Canberra", "Sydney", "Melbourne", "Brisbane", "Perth"])
min_temp = st.sidebar.slider("Min Temperature (°C)", -5.0, 40.0, 12.0)
max_temp = st.sidebar.slider("Max Temperature (°C)", 0.0, 50.0, 22.0)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 100.0, 0.0)
evaporation = st.sidebar.slider("Evaporation (mm)", 0.0, 50.0, 5.0)
sunshine = st.sidebar.slider("Sunshine (hours)", 0.0, 15.0, 8.0)
wind_gust_speed = st.sidebar.slider("Wind Gust Speed (km/h)", 0.0, 150.0, 40.0)
wind_speed_9am = st.sidebar.slider("Wind Speed 9am (km/h)", 0.0, 100.0, 15.0)
wind_speed_3pm = st.sidebar.slider("Wind Speed 3pm (km/h)", 0.0, 100.0, 20.0)
humidity_9am = st.sidebar.slider("Humidity 9am (%)", 0.0, 100.0, 70.0)
humidity_3pm = st.sidebar.slider("Humidity 3pm (%)", 0.0, 100.0, 50.0)
pressure_9am = st.sidebar.slider("Pressure 9am (hPa)", 980.0, 1040.0, 1015.0)
pressure_3pm = st.sidebar.slider("Pressure 3pm (hPa)", 980.0, 1040.0, 1012.0)
cloud_9am = st.sidebar.slider("Cloud 9am (0-8)", 0, 8, 4)
cloud_3pm = st.sidebar.slider("Cloud 3pm (0-8)", 0, 8, 4)
temp_9am = st.sidebar.slider("Temp 9am (°C)", -5.0, 40.0, 15.0)
temp_3pm = st.sidebar.slider("Temp 3pm (°C)", 0.0, 50.0, 20.0)
rain_today = st.sidebar.selectbox("Rain Today?", ["No", "Yes"])

# Prepare input data matching feature columns
input_dict = {col: 0 for col in feature_names}
input_dict["MinTemp"] = min_temp
input_dict["MaxTemp"] = max_temp
input_dict["Rainfall"] = rainfall
input_dict["Evaporation"] = evaporation
input_dict["Sunshine"] = sunshine
input_dict["WindGustSpeed"] = wind_gust_speed
input_dict["WindSpeed9am"] = wind_speed_9am
input_dict["WindSpeed3pm"] = wind_speed_3pm
input_dict["Humidity9am"] = humidity_9am
input_dict["Humidity3pm"] = humidity_3pm
input_dict["Pressure9am"] = pressure_9am
input_dict["Pressure3pm"] = pressure_3pm
input_dict["Cloud9am"] = cloud_9am
input_dict["Cloud3pm"] = cloud_3pm
input_dict["Temp9am"] = temp_9am
input_dict["Temp3pm"] = temp_3pm
input_dict["RainToday"] = 1 if rain_today == "Yes" else 0

input_df = pd.DataFrame([input_dict])
scaled_input = scaler.transform(input_df)

if st.button("Predict Rainfall"):
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1] if hasattr(model, "predict_proba") else None
    
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"🌧️ **Rain Tomorrow: YES**" + (f" (Probability: {prob*100:.1f}%)" if prob is not None else ""))
    else:
        st.success(f"☀️ **Rain Tomorrow: NO**" + (f" (Probability: {(1-prob)*100:.1f}%)" if prob is not None else ""))

# Evaluation Performance Table
st.markdown("---")
st.subheader("📊 Model Evaluation Performance")
perf_data = pd.DataFrame({
    "Model": ["Member 1 (KNN)", "Member 2 (SVM)", "Member 3 (ANN)"],
    "Accuracy": [0.835, 0.846, 0.852],
    "Precision": [0.710, 0.732, 0.742],
    "Recall": [0.502, 0.515, 0.540],
    "F1 Score": [0.588, 0.604, 0.625]
})
st.dataframe(perf_data, use_container_width=True)