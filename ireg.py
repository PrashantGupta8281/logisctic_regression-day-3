import streamlit as st
import joblib
import base64

# ---------- Function to Load Background ----------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Transparent white container */
        .main > div {{
            background-color: rgba(255,255,255,0.82);
            padding: 30px;
            border-radius: 15px;
        }}

        h1 {{
            color: #0A4D8C;
            text-align: center;
        }}

        p {{
            color: black;
            font-size:18px;
        }}

        div.stButton > button {{
            background-color: #007ACC;
            color: white;
            border-radius: 10px;
            height: 45px;
            width: 100%;
            font-size: 18px;
        }}

        div.stButton > button:hover {{
            background-color: #005A9E;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Add Background
add_bg_from_local("background.jpg")

# Load Model
model = joblib.load("model.pkl")

# App Title
st.title("🏥 Life Insurance Prediction")

st.write("Enter the person's age to predict whether they are likely to buy life insurance.")

# Input
age = st.number_input("Enter Age", min_value=1, max_value=100, value=25)

# Prediction
if st.button("Predict"):

    prediction = model.predict([[age]])
    probability = model.predict_proba([[age]])

    if prediction[0] == 1:
        st.success("✅ The person is likely to BUY Life Insurance.")
    else:
        st.error("❌ The person is NOT likely to BUY Life Insurance.")

    st.write(f"**Probability of Buying:** {probability[0][1]*100:.2f}%")
    st.write(f"**Probability of Not Buying:** {probability[0][0]*100:.2f}%")
