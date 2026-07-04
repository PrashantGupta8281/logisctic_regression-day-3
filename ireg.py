

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Insurance Sales Prediction",
    page_icon="💵",
    layout="centered"
)

st.title("💵 Project 3: Insurance Sales Prediction")
st.write("Predict Insurance Sales using Logistic Regression")

# -----------------------------------
# Load Dataset
# -----------------------------------
df = pd.read_csv("insurance_data.csv")

st.subheader("Insurance Dataset")
st.dataframe(df)

# -----------------------------------
# Train Model
# -----------------------------------

X = df[["age"]]
y = [premium]

model = LogisticRegression()

model.fit(X, y)

# -----------------------------------
# User Input
# -----------------------------------
st.subheader("Enter Age")

age = st.number_input(
    "Age (in Years)",
    min_value=10,
    max_value=65,
    value=18,
    step=1
)

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("Predict Insurance"):

    prediction = model.predict([[age]])

    st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")

# -----------------------------------
# Model Information
# -----------------------------------
st.subheader("Insurance Details")

st.write("Coefficient:", model.coef_[0])
st.write("Intercept:", model.intercept_)
