import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Rainfall Prediction System", page_icon="🌧️", layout="wide"
)

st.title("🌧️ Australian Rainfall Prediction System")
st.markdown(
    "Predict whether it will rain tomorrow based on weather metrics using Machine Learning models."
)

# Load Models & Scaler
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

models = {
    "Member 1: KNN": joblib.load("models/knn_model.pkl"),
    "Member 2: Random Forest": joblib.load("models/random_forest_model.pkl"),
    "Member 3: ANN": joblib.load("models/ann_model.pkl"),
}

# Sidebar Input Form
st.sidebar.header("📊 Input Weather Parameters")


def user_input_features():
    min_temp = st.sidebar.slider("Min Temp (°C)", -5.0, 40.0, 12.0)
    max_temp = st.sidebar.slider("Max Temp (°C)", 0.0, 45.0, 23.0)
    rainfall = st.sidebar.slider("Rainfall Today (mm)", 0.0, 100.0, 0.0)
    evaporation = st.sidebar.slider("Evaporation (mm)", 0.0, 20.0, 5.0)
    sunshine = st.sidebar.slider("Sunshine (hours)", 0.0, 15.0, 8.0)
    wind_gust_speed = st.sidebar.slider("Wind Gust Speed (km/h)", 0, 130, 40)
    humidity_9am = st.sidebar.slider("Humidity 9am (%)", 0, 100, 70)
    humidity_3pm = st.sidebar.slider("Humidity 3pm (%)", 0, 100, 50)
    pressure_9am = st.sidebar.slider("Pressure 9am (hPa)", 980, 1040, 1017)
    pressure_3pm = st.sidebar.slider("Pressure 3pm (hPa)", 980, 1040, 1015)
    cloud_9am = st.sidebar.slider("Cloud 9am (0-8)", 0, 8, 4)
    cloud_3pm = st.sidebar.slider("Cloud 3pm (0-8)", 0, 8, 4)
    temp_9am = st.sidebar.slider("Temp 9am (°C)", -5.0, 40.0, 16.0)
    temp_3pm = st.sidebar.slider("Temp 3pm (°C)", -5.0, 45.0, 21.0)
    rain_today = st.sidebar.selectbox("Rain Today?", ["No", "Yes"])

    rain_today_num = 1 if rain_today == "Yes" else 0

    # Build row matching feature shape
    input_data = pd.DataFrame(0, index=[0], columns=feature_names)
    input_data["MinTemp"] = min_temp
    input_data["MaxTemp"] = max_temp
    input_data["Rainfall"] = rainfall
    input_data["Evaporation"] = evaporation
    input_data["Sunshine"] = sunshine
    input_data["WindGustSpeed"] = wind_gust_speed
    input_data["Humidity9am"] = humidity_9am
    input_data["Humidity3pm"] = humidity_3pm
    input_data["Pressure9am"] = pressure_9am
    input_data["Pressure3pm"] = pressure_3pm
    input_data["Cloud9am"] = cloud_9am
    input_data["Cloud3pm"] = cloud_3pm
    input_data["Temp9am"] = temp_9am
    input_data["Temp3pm"] = temp_3pm
    input_data["RainToday"] = rain_today_num

    return input_data


input_df = user_input_features()

# Main Layout: 2 Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("⚙️ Select Model & Predict")
    selected_model_name = st.selectbox("Choose Classification Model:", list(models.keys()))
    selected_model = models[selected_model_name]

    if st.button("Predict Rain Tomorrow"):
        scaled_input = scaler.transform(input_df)
        prediction = selected_model.predict(scaled_input)[0]

        st.divider()
        if prediction == 1:
            st.error("🌧️ **Prediction Result: It is likely to RAIN tomorrow!**")
        else:
            st.success("☀️ **Prediction Result: It will NOT rain tomorrow.**")

with col2:
    st.subheader("📈 Model Evaluation Performance")
    # Performance summary from trained models
    metrics_data = {
        "Model": ["Member 1 (KNN)", "Member 2 (Random Forest)", "Member 3 (ANN)"],
        "Accuracy": [0.8350, 0.8580, 0.8520],
        "Precision": [0.7100, 0.7750, 0.7420],
        "Recall": [0.5020, 0.5310, 0.5400],
        "F1 Score": [0.5880, 0.6300, 0.6250],
    }
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)