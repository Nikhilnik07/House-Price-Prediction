import streamlit as st
import joblib
import numpy as np


# Load trained model
model = joblib.load('/Users/nikhilkumar/Desktop/Ds/Self_Learning_Through_Projects/House_Price_Prediction/model/house_price_model.pkl')


st.set_page_config(page_title="House Price Predictor" , layout='centered')

st.title("🏠 House Price Prediction App")
st.write("Predict house price based on property details")


# Input fields
area = st.number_input("Area (in sqft)",min_value=300,max_value=5000,step=50)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)


# Prediction
if st.button("Predict Price"):
    input_data = np.array([[area,bedrooms,bathrooms]])
    prediction = model.predict(input_data)
    st.success(f"Estimated House Price : ${prediction[0]:.2f} Lakhs")