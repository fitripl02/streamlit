import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.title("🔮 Halaman 3: Prediksi Klaster Restoran Baru")

# Input user
resto_rating = st.slider("Rating Restoran", 1.0, 5.0, 4.0)
operation_hours = st.slider("Jam Operasi", 1.0, 24.0, 10.0)
wifi = st.selectbox("Wifi Tersedia", [0, 1])
toilet = st.selectbox("Ada Toilet", [0, 1])
cash_only = st.selectbox("Hanya Tunai", [0, 1])

# Prediksi Klaster
if st.button("Prediksi Klaster"):
    input_data = np.array([[resto_rating, operation_hours, wifi, toilet, cash_only]])

    # Dummy model: fit KMeans ulang (seharusnya load model dari pickle di versi produksi)
    dummy_data = np.random.rand(100, 5)  # contoh data acak untuk fitting
    scaler = StandardScaler()
    dummy_scaled = scaler.fit_transform(dummy_data)
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(dummy_scaled)

    # Prediksi user
    user_scaled = scaler.transform(input_data)
    pred = model.predict(user_scaled)
    st.success(f"Restoran ini diprediksi masuk ke Klaster: **{int(pred[0])}**")
