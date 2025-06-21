import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load("linear_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="California House Price Predictor", page_icon="🏠")
st.title("🏠 California House Price Predictor")
st.markdown("Enter the details below to predict the **median house value** in California.")

# Input fields with validation
MedInc = st.number_input("Median Income (0.5 - 15.0)", 0.5, 15.0, 5.0, 0.1, key="MedInc")
HouseAge = st.number_input("House Age (1 - 52)", 1, 52, 30, key="HouseAge")
AveRooms = st.number_input("Average Rooms (0.8 - 142)", 0.8, 141.9, 5.5, 0.1, key="AveRooms")
AveBedrms = st.number_input("Average Bedrooms (0.3 - 34)", 0.3, 34.0, 1.0, 0.1, key="AveBedrms")
Population = st.number_input("Population (3 - 35682)", 3, 35682, 1000, key="Population")
AveOccup = st.number_input("Average Occupants (0.7 - 1243)", 0.7, 1243.0, 3.0, 0.1, key="AveOccup")
Latitude = st.number_input("Latitude (32.5 - 42.0)", 32.5, 42.0, 34.0, 0.01, key="Latitude")
Longitude = st.number_input("Longitude (-125 to -113)", -125.0, -113.0, -118.0, 0.01, key="Longitude")

# ✅ Predict button with unique key
if st.button("Predict", key="predict_button"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population,
                            AveOccup, Latitude, Longitude]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f"💰 Predicted House Price: **${prediction * 100000:,.2f}**")
