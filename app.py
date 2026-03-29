import streamlit as st
import pickle
import numpy as np

# Load model
model, scaler = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Energy Classifier", layout="centered")

st.title("⚡ Energy Consumption Classifier")
st.write("Predict whether energy usage is Normal or High")

# Inputs
T1 = st.number_input("Living Room Temperature (T1)")
RH_1 = st.number_input("Living Room Humidity (RH_1)")
T2 = st.number_input("Kitchen Temperature (T2)")
RH_2 = st.number_input("Kitchen Humidity (RH_2)")
T_out = st.number_input("Outdoor Temperature (T_out)")
RH_out = st.number_input("Outdoor Humidity (RH_out)")

# Predict
if st.button("Predict"):
    features = np.array([[T1, RH_1, T2, RH_2, T_out, RH_out]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    if prediction == 0:
        st.success("⚡ Normal Energy Consumption")
    else:
        st.error("🔥 High Energy Consumption")