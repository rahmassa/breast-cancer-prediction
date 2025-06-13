import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load('models/logreg_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Form input
st.title("Prediksi Sifat Tumor Payudara")
st.subheader("Masukkan nilai fitur:")

# 9 fitur input
concave_points_worst = st.number_input('Concave Points Worst')
perimeter_worst = st.number_input('Perimeter Worst')
concave_points_mean = st.number_input('Concave Points Mean')
radius_worst = st.number_input('Radius Worst')
perimeter_mean = st.number_input('Perimeter Mean')
area_worst = st.number_input('Area Worst')
radius_mean = st.number_input('Radius Mean')
area_mean = st.number_input('Area Mean')
concavity_mean = st.number_input('Concavity Mean')

# Susun urutan input sesuai selected_features
input_data = np.array([[concave_points_worst, perimeter_worst, concave_points_mean,
                        radius_worst, perimeter_mean, area_worst,
                        radius_mean, area_mean, concavity_mean]])

# Scaling
input_scaled = scaler.transform(input_data)

# Tombol Prediksi
if st.button("Prediksi"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.subheader("Hasil Prediksi:")
    if prediction[0] == 0:
        st.write("Tumor Jinak")
    else:
        st.write("Tumor Ganas")
