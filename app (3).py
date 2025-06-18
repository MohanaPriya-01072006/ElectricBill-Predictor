import streamlit as st
import pickle
import numpy as np

# Load trained model
@st.cache_resource
def load_model():
    with open("bill_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Streamlit UI
st.title("🔌 Monthly Electricity Bill Predictor")

st.markdown("Enter the number of days and units consumed to estimate your monthly bill.")

# Input fields
days = st.number_input("📅 Number of days in the month", min_value=1, max_value=31, value=30)
units = st.number_input("⚡ Units consumed", min_value=0, value=150)

# Predict
if st.button("🔍 Predict Bill"):
    input_data = np.array([[days, units]])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Monthly Bill: ₹{prediction:.2f}")
