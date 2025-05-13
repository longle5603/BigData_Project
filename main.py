import streamlit as st
import numpy as np
import joblib
import pickle

# Load the trained model and location-to-index mapping
model = joblib.load("house_price_model.pkl")

with open("location_to_index.pkl", "rb") as f:
    location_to_index = pickle.load(f)

# Number of location features (used during training)
NUM_LOCATIONS = model.coef_.shape[0] - 4

# --- Input range limits ---
MIN_BHK, MAX_BHK = 1, 8
MIN_SQFT, MAX_SQFT = 300.0, 11890.0
MIN_BATH, MAX_BATH = 1, 8
MIN_BALCONY, MAX_BALCONY = 0, 3

# --- Streamlit UI ---
st.title("🏠 Bangalore House Price Prediction")

st.markdown(f"""
**📌 Please enter the following information:**

- `BHK (Bedrooms)`: {MIN_BHK} to {MAX_BHK}
- `Total area (sqft)`: {MIN_SQFT} to {MAX_SQFT}
- `Number of bathrooms`: {MIN_BATH} to {MAX_BATH}
- `Number of balconies`: {MIN_BALCONY} to {MAX_BALCONY}
""")

# --- Input fields ---
bhk = st.number_input("Number of Bedrooms (BHK)", min_value=MIN_BHK, max_value=MAX_BHK, step=1)
total_sqft = st.number_input("Total Area (in sqft)", value=1000.0)
bath = st.number_input("Number of Bathrooms", min_value=MIN_BATH, max_value=MAX_BATH, step=1)
balcony = st.number_input("Number of Balconies", min_value=MIN_BALCONY, max_value=MAX_BALCONY, step=1)

location = st.selectbox("Select Location", sorted(location_to_index.keys()))

# --- Input validation ---
errors = []

if not (MIN_SQFT <= total_sqft <= MAX_SQFT):
    errors.append(f"Total area must be between {MIN_SQFT} and {MAX_SQFT} sqft.")

if errors:
    for error in errors:
        st.warning(error)
else:
    # One-hot encoding for location
    location_vec = np.zeros(NUM_LOCATIONS)
    location_idx = location_to_index.get(location)

    if location_idx is not None and location_idx < NUM_LOCATIONS:
        location_vec[location_idx] = 1
    else:
        st.error("Selected location is invalid or was not part of training data.")
        st.stop()

    # Construct feature vector
    feature_vector = np.concatenate(([total_sqft, bath, balcony, bhk], location_vec))

    # Predict price
    predicted_price = model.predict([feature_vector])[0]

    # Show result
    st.success(f"💰 Estimated House Price: **₹{predicted_price:,.2f} Lakh INR**")
