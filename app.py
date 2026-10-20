import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Energy Forecast")

st.title("Hourly Energy Consumption Forecast")
st.write("Electricity demand prediction using XGBoost")

@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.pkl")

model = load_model()
expected_features = model.get_booster().feature_names

st.subheader("Select Forecast Time")

date = st.date_input("Date", value=pd.Timestamp("2017-01-01"))
hour = st.slider("Hour", 0, 23, 12)

st.subheader("Historical Energy Values")

lag_1 = st.number_input("Consumption 1 Hour Ago", value=5000.0)
lag_24 = st.number_input("Consumption 24 Hours Ago", value=5000.0)
lag_168 = st.number_input("Consumption 168 Hours Ago", value=5000.0)
rolling_24 = st.number_input("24-Hour Rolling Mean", value=5000.0)
rolling_168 = st.number_input("168-Hour Rolling Mean", value=5000.0)

if st.button("Predict Energy Consumption"):
    dt = pd.Timestamp(date) + pd.Timedelta(hours=hour)

    features = {
        "hour": dt.hour,
        "day_of_week": dt.dayofweek,
        "month": dt.month,
        "year": dt.year,
        "is_weekend": int(dt.dayofweek >= 5),
        "hour_sin": np.sin(2 * np.pi * dt.hour / 24),
        "hour_cos": np.cos(2 * np.pi * dt.hour / 24),
        "month_sin": np.sin(2 * np.pi * dt.month / 12),
        "month_cos": np.cos(2 * np.pi * dt.month / 12),
        "lag_1": lag_1,
        "lag_24": lag_24,
        "lag_168": lag_168,
        "rolling_24": rolling_24,
        "rolling_168": rolling_168
    }

    missing = [f for f in expected_features if f not in features]

    if missing:
        st.error(f"Missing model features: {missing}")
    else:
        data = pd.DataFrame(
            [[features[f] for f in expected_features]],
            columns=expected_features
        )

        try:
            prediction = model.predict(data)[0]
            st.success(
                f"Predicted Energy Consumption: {prediction:,.2f} MW"
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.caption(
    "Prediction quality depends on accurate historical values "
    "and consistency with the training features."
)
