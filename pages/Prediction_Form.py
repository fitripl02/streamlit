import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.title("🔮 Halaman 3: Prediksi Klaster Restoran Baru")

# === 1. Input User ===
resto_name = st.text_input("📝 Nama Restoran", placeholder="Contoh: Resto Sederhana")
resto_rating = st.slider("⭐ Rating Restoran", 1.0, 5.0, 4.0)
operation_hours = st.slider("🕒 Jam Operasi per Hari", 1.0, 24.0, 10.0)
wifi = st.selectbox("📶 Wifi Tersedia?", [0, 1])
toilet = st.selectbox("🚻 Ada Toilet?", [0, 1])
cash_only = st.selectbox("💵 Hanya Menerima Tunai?", [0, 1])

# === 2. Prediksi Klaster ===
if st.button("🔍 Prediksi Klaster"):
    if resto_name.strip() == "":
        st.warning("⚠️ Harap masukkan nama restoran terlebih dahulu.")
    else:
        input_data = np.array([[resto_rating, operation_hours, wifi, toilet, cash_only]])

        # Dummy model: fit KMeans ulang (untuk produksi sebaiknya load model tersimpan)
        dummy_data = np.random.rand(100, 5)
        scaler = StandardScaler()
        dummy_scaled = scaler.fit_transform(dummy_data)
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(dummy_scaled)

        # Prediksi klaster
        user_scaled = scaler.transform(input_data)
        pred = model.predict(user_scaled)

        st.success(f"🍽️ **{resto_name}** diprediksi termasuk dalam **Klaster: {int(pred[0])}**")
