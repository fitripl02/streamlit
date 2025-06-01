import streamlit as st
import pandas as pd
import joblib

st.title("📈 Hasil Penelitian Model")

st.subheader("📂 Model yang Digunakan")
st.write("Model clustering/prediksi: KMeans (misalnya)")

# Contoh: tampilkan cluster yang telah diprediksi
data = pd.read_csv("Modul/dataset_clustered.csv")  # Dataset hasil model
st.write(data.head())

st.subheader("📍Distribusi Klaster")
st.bar_chart(data['cluster'].value_counts())
