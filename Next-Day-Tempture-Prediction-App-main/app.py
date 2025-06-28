import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Streamlit page config
st.set_page_config(page_title="Next Day Temperature Prediction", layout="wide")

# App title
st.title("🌡️ Next Day Temperature Predictor")
st.markdown("Upload a CSV file with `Date` and `Temp` columns to predict the next day's temperature.")

# Upload CSV
uploaded_file = st.file_uploader("Upload your temperature CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
        df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")
        df = df.dropna()

        # Load the model and scaler
        model = joblib.load("next_day_temp.pkl")
        scaler = joblib.load("scaler.pkl")

        # Normalize temperature
        data_scaled = scaler.transform(df["Temp"].values.reshape(-1, 1))

        # Prepare last 30 days sequence
        seq_length = 30
        if len(data_scaled) < seq_length:
            st.error(f"❌ Not enough data. Need at least {seq_length} rows.")
        else:
            last_sequence = data_scaled[-seq_length:].reshape(1, seq_length, 1)
            next_temp_scaled = model.predict(last_sequence)
            next_temp_scaled = np.clip(next_temp_scaled, 0, 1)
            next_temp = scaler.inverse_transform(next_temp_scaled)

            # Display result
            st.success(f"🌞 The predicted temperature for the next day is: **{next_temp[0][0]:.2f} °C**")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
else:
    st.info("📂 Please upload a CSV file to proceed.")
