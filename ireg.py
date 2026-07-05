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

st.title("💵 Project 3: Insurance Rate Prediction")
st.write("Predict Whether a person is likely to buy life insurance based on the age.")

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
y = ["bought_insurance"]

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
if st.button("Predict"):

    prediction = model.predict([[age]])[0]
    probability = model.predict_proba([[age]])[0][1]

# -----------------------------------
# Model Information
# -----------------------------------
st.subheader("Prediction")

  if prediction == 1:
        st.success("✅ The person is likely to buy life insurance.")
    else:
        st.error("❌ The person is NOT likely to buy life insurance.")

    st.write(f"Probability of Buying Insurance: **{probability*100:.2f}%**")

# Show dataset
if st.checkbox("Show Dataset"):
    st.dataframe(df)
