import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("models/logreg_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Judul dan deskripsi
st.title("🩺 Aplikasi Deteksi Sifat Tumor Payudara")
st.caption("Deteksi apakah tumor bersifat **Jinak** atau **Ganas** berdasarkan fitur medis pasien.")

st.write("---")
st.subheader("Silakan masukkan nilai untuk masing-masing fitur berikut:")

# Input user (dengan bantuan/help)
f1 = st.number_input("Concave Points Worst", 0.0, 0.3, step=0.001,
                     help="Contoh: 0.14 – Banyaknya lekukan tajam pada batas tumor")
f2 = st.number_input("Perimeter Worst", 50.0, 250.0, step=0.1,
                     help="Contoh: 132.9 – Keliling terbesar dari tumor")
f3 = st.number_input("Concave Points Mean", 0.0, 0.2, step=0.001,
                     help="Contoh: 0.08 – Rata-rata jumlah lekukan batas tumor")
f4 = st.number_input("Radius Worst", 7.0, 30.0, step=0.1,
                     help="Contoh: 25.38 – Radius paling besar dari inti sel tumor")
f5 = st.number_input("Perimeter Mean", 40.0, 150.0, step=0.1,
                     help="Contoh: 87.46 – Keliling rata-rata tumor")
f6 = st.number_input("Area Worst", 200.0, 2500.0, step=1.0,
                     help="Contoh: 2019.0 – Luas maksimum sel tumor")
f7 = st.number_input("Radius Mean", 6.0, 28.0, step=0.1,
                     help="Contoh: 17.99 – Rata-rata radius sel tumor")
f8 = st.number_input("Area Mean", 100.0, 1500.0, step=1.0,
                     help="Contoh: 1001.0 – Luas rata-rata tumor")
f9 = st.number_input("Concavity Mean", 0.0, 0.5, step=0.001,
                     help="Contoh: 0.3 – Rata-rata kedalaman lengkungan tepi tumor")

# Tombol Deteksi
if st.button("🧪 Deteksi"):
    input_data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9]])
    input_scaled = scaler.transform(input_data)
    hasil = model.predict(input_scaled)[0]

    st.subheader("Hasil Deteksi:")
    if hasil == 0:
        st.error("⚠️ Terdeteksi: Tumor Ganas (Malignant)")
    else:
        st.success("✅ Terdeteksi: Tumor Jinak (Benign)")
