import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🗂️ Data Awal & Eksplorasi Data")

# Contoh load data
data = pd.read_csv("Modul/dataset.csv")  # Ganti dengan path dataset kamu

st.subheader("📁 Tampilan Data")
st.dataframe(data)

st.subheader("📌 Statistik Deskriptif")
st.write(data.describe())

st.subheader("📊 Visualisasi Korelasi")
fig, ax = plt.subplots()
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
