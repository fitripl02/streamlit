import streamlit as st
import pandas as pd
import joblib

st.title("🧮 Formulir Prediksi")

model = joblib.load("Modul/model.pkl")  # Load model
scaler = joblib.load("Modul/scaler.pkl")  # Jika ada preprocessing

# Form input
st.subheader("Masukkan Data untuk Prediksi:")
feature_1 = st.number_input("Fitur 1")
feature_2 = st.number_input("Fitur 2")
# Tambahkan fitur sesuai dataset

if st.button("🔮 Prediksi"):
    input_df = pd.DataFrame([[feature_1, feature_2]], columns=["Fitur1", "Fitur2"])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f"Hasil Prediksi: Klaster {prediction[0]}")
