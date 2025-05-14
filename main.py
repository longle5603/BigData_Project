import streamlit as st
import numpy as np
import joblib
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import pickle
model = joblib.load("house_price_model2.pkl")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

MIN_BHK, MAX_BHK = 1, 8
MIN_SQFT, MAX_SQFT = 300.0, 11890.0
MIN_BATH, MAX_BATH = 1, 8
MIN_BALCONY, MAX_BALCONY = 0, 3

st.title("🏠 Bangalore House Price Prediction (BERT Version)")

st.markdown("**📌 Enter house details:**")
bhk = st.number_input("Bedrooms (BHK)", min_value=MIN_BHK, max_value=MAX_BHK, step=1)
total_sqft = st.number_input("Total Area (in sqft)", value=1000.0)
bath = st.number_input("Bathrooms", min_value=MIN_BATH, max_value=MAX_BATH, step=1)
balcony = st.number_input("Balconies", min_value=MIN_BALCONY, max_value=MAX_BALCONY, step=1)

with open("trained_locations.pkl", "rb") as f:
    trained_locations = pickle.load(f)

location = st.selectbox("Select Location", sorted(trained_locations))

if total_sqft / bhk < 300:
    st.warning("⚠️ Area per BHK seems too small.")
if bath > bhk + 1:
    st.warning("⚠️ Too many bathrooms for selected BHK.")

if st.button("Predict Price"):
    if location.strip() == "":
        st.error("Please enter a valid location.")
        st.stop()

    # Encode location using BERT
    with st.spinner("Encoding location with BERT..."):
        inputs = tokenizer(location, return_tensors="pt")
        outputs = bert_model(**inputs)
        location_emb = outputs.last_hidden_state[:, 0, :].detach().numpy().flatten()

    features = np.concatenate(([total_sqft, bath, balcony, bhk], location_emb))
    feature_df = pd.DataFrame([features])

    price = model.predict(feature_df)[0]
    if price < 0:
        st.error("❌ Invalid configuration, house doesn't exist.")
    else:
        st.success(f"💰 Estimated House Price: **₹{price:,.2f} Lakh INR**")
