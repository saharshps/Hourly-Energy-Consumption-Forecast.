import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Custom 30-Day Energy Forecast", layout="wide")

st.title("Hourly Energy Consumption Forecast Generator")
st.write("Select any start date/time between 2007 and 2026 to simulate a 30-day future load profile.")

@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.pkl")

try:
    model = load_model()
    expected_features = [
        'hour', 'day_of_week', 'month', 'year', 'is_weekend', 
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
        'lag_1', 'lag_24', 'lag_168', 'rolling_24', 'rolling_168'
    ]
except Exception as e:
    st.error(f"Failed to load framework file 'xgboost_model.pkl': {e}")
    st.stop()

# --- TARGET TIME HORIZON SELECTION INTERFACE ---
st.subheader("🗓️ Select Forecast Configuration")
col_date, col_hour = st.columns(2)

with col_date:
    # Allows setting any target date boundary from 2007 through late 2026
    start_date = st.date_input(
        "Start Forecast Date",
        value=pd.Timestamp("2018-06-01"),
        min_value=pd.Timestamp("2007-01-01"),
        max_value=pd.Timestamp("2026-12-31")
    )

with col_hour:
    start_hour = st.slider("Start Hour of Day", min_value=0, max_value=23, value=12)

# Combine inputs into a clean target timestamp
target_start_time = pd.Timestamp(start_date) + pd.Timedelta(hours=start_hour)

if st.button("Generate 30-Day Forecast Schedule", type="primary"):
    
    # 1. Synthesize a mathematically consistent 168-hour historical baseline frame ending at target_start_time
    history_dates = pd.date_range(end=target_start_time, periods=168, freq="h")
    
    # Base load varies naturally depending on the month chosen to simulate realistic grid parameters
    base_load_by_season = 5000 + (np.sin(2 * np.pi * target_start_time.month / 12) * 800)
    
    active_series = pd.DataFrame({
        "Datetime": history_dates,
        "PJMW_MW": np.random.normal(base_load_by_season, 300, 168)
    })
    
    # 2. Setup the 30-day (720 hour) future prediction window
    forecast_horizon = 24 * 30
    forecast_dates = pd.date_range(start=target_start_time + pd.Timedelta(hours=1), periods=forecast_horizon, freq="h")
    
    progress_bar = st.progress(0.0)
    
    # 3. Core Autoregressive Execution Sequence
    for step, next_time in enumerate(forecast_dates):
        hour_val = next_time.hour
        day_of_week = next_time.dayofweek
        month_val = next_time.month
        year_val = next_time.year
        is_weekend = int(day_of_week >= 5)
        
        lag_1_val = float(active_series["PJMW_MW"].iloc[-1])
        lag_24_val = float(active_series["PJMW_MW"].iloc[-24])
        lag_168_val = float(active_series["PJMW_MW"].iloc[-168])
        
        rolling_24_val = float(active_series["PJMW_MW"].iloc[-24:].mean())
        rolling_168_val = float(active_series["PJMW_MW"].iloc[-168:].mean())
        
        feature_map = {
            "hour": hour_val, 
            "day_of_week": day_of_week, 
            "month": month_val, 
            "year": year_val, 
            "is_weekend": is_weekend,
            "hour_sin": np.sin(2 * np.pi * hour_val / 24), 
            "hour_cos": np.cos(2 * np.pi * hour_val / 24),
            "month_sin": np.sin(2 * np.pi * month_val / 12), 
            "month_cos": np.cos(2 * np.pi * month_val / 12),
            "lag_1": lag_1_val, 
            "lag_24": lag_24_val, 
            "lag_168": lag_168_val,
            "rolling_24": rolling_24_val, 
            "rolling_168": rolling_168_val
        }
        
        input_row = pd.DataFrame([[feature_map[f] for f in expected_features]], columns=expected_features)
        
        # FIX 1: Force all fields into pure numeric floats for XGBoost
        input_row = input_row.astype(float)
        
        # FIX 2: Extract index [0] to fetch the single scalar from the prediction array safely
        next_pred = float(model.predict(input_row)[0])
        
        new_row = pd.DataFrame({"Datetime": [next_time], "PJMW_MW": [next_pred]})
        active_series = pd.concat([active_series, new_row], ignore_index=True)
        
        progress_bar.progress((step + 1) / forecast_horizon)
        
    progress_bar.empty()
    
    # 4. Extract exclusively the 30-day forecast payload
    final_forecast = active_series.tail(forecast_horizon).copy().reset_index(drop=True)
    final_forecast.columns = ["Forecast Time", "Predicted Load (MW)"]
    
    # 5. Display Analytics Interface
    st.subheader(f"📊 30-Day Forecast Timeline (Starting: {target_start_time.strftime('%Y-%m-%d %H:%M')})")
    
    # FIX 3: Pass integer parameter '2' explicitly to st.columns
    col1, col2 = st.columns(2)
    with col1:
        st.write("Forecast Schedule DataFrame:")
        st.dataframe(final_forecast.style.format({"Predicted Load (MW)": "{:,.2f}"}), height=400)
    with col2:
        st.write("30-Day Extrapolated Trendline Visualization:")
        st.line_chart(data=final_forecast, x="Forecast Time", y="Predicted Load (MW)", use_container_width=True)
        
    st.download_button(
        label="Download Generated 30-Day Outlook (CSV)",
        data=final_forecast.to_csv(index=False),
        file_name=f"energy_forecast_{start_date}.csv",
        mime="text/csv",
        type="primary"
    )
