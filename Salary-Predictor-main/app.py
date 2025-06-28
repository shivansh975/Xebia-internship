import streamlit as st
import joblib
import numpy as np
import pandas as pd

#Load the model and Scaler
model = joblib.load("predict_salary.pk1")
scaler = joblib.load("scaler.pk1")

#Design the layout of our basic app
st.set_page_config(page_title = "Salary Predictor",layout = "centered")
st.title("Salary Prediction App")
st.subheader("Predict your salary based on the number of years of experience")
st.write("select the years of experience to see the estimated salary")


# Create a dropdown for the years of experience
years = [x for x in range(0,20)]
years_exp = st.selectbox("Years of Experience:",years)

# Predict the Salary
if st.button("Predict Salary"):
    input_data = np.array([[years_exp]])
    input_scaled = scaler.transform(input_data)
    predicted_salary = model.predict(input_scaled)
    st.success("Estimated salary: $ "+ str(predicted_salary[0][0]))

