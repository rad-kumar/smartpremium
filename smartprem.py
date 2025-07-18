import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("best_insurance_model.pkl")

st.title("Insurance Premium Prediction App")

# Example inputs (you can customize with sliders, selects, etc.)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
annual_income = st.number_input("Annual Income", min_value=10000, max_value=1000000, value=50000)
num_policies = st.slider("Number of Policies", 1, 10, 2)

# Put inputs into DataFrame
input_df = pd.DataFrame({
    "Age": [age],
    "Annual Income": [annual_income],
    "Number of Policies": [num_policies]
    # Include other columns your model expects
})

# Predict
if st.button("Predict Premium"):
    prediction = model.predict(input_df)
    st.success(f"Estimated Premium Amount: ₹{prediction[0]:,.2f}")
