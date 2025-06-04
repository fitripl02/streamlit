import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Klaster Restoran", page_icon="🔮", layout="wide")

# Judul halaman
st.title("🔮 Prediksi Klaster untuk Restoran Baru")
st.markdown("""
**Gunakan tool ini untuk memprediksi ke klaster mana restoran baru akan masuk berdasarkan karakteristiknya.**
""")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('semarang_resto_dataset.csv')

df = load_data()

# Pilih fitur untuk clustering
features = ['resto_rating', 'average_operation_hours', 'wifi_facility', 'toilet_facility', 'cash_payment_only']
X = df[features]

# Scaling data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering dengan KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
df['cluster'] = kmeans.predict(X_scaled)

# Deskripsi klaster (contoh interpretasi)
cluster_descriptions = {
    0: "Restoran biasa dengan fasilitas terbatas",
    1: "Restoran premium dengan fasilitas lengkap",
    2: "Restoran dengan jam operasi panjang tapi fasilitas minimal"
}

# Sidebar untuk input user
st.sidebar.header("🛠️ Parameter Restoran Baru")
resto_rating = st.sidebar.slider("Rating Restoran (1-5)", 1.0, 5.0, 3.5, 0.1)
operation_hours = st.sidebar.slider("Rata-rata Jam Operasi per Hari", 1.0, 24.0, 12.0, 0.5)
wifi = st.sidebar.radio("Fasilitas Wifi", [("Tersedia", 1), ("Tidak Tersedia", 0)], index=0)[1]
toilet = st.sidebar.radio("Fasilitas Toilet", [("Tersedia", 1), ("Tidak Tersedia", 0)], index=0)[1]
cash_only = st.sidebar.radio("Pembayaran", [("Tunai Saja", 1), ("Menerima Non-Tunai", 0)], index=1)[1]

# Tombol prediksi
if st.sidebar.button("🚀 Prediksi Klaster", help="Klik untuk memprediksi klaster"):
    # Preprocessing input
    input_data = np.array([[resto_rating, operation_hours, wifi, toilet, cash_only]])
    user
