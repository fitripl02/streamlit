import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🤖 Halaman 2: Klastering Restoran (KMeans)")

# --- 1. Load dataset
url = "https://raw.githubusercontent.com/fitripl02/streamlit/refs/heads/main/semarang_resto_dataset.csv"
df = pd.read_csv(url)

# --- 2. Ambil fitur numerik yang valid
expected_cols = ['rating', 'hours_open', 'wifi', 'cash_only', 'debit']
missing_cols = [col for col in expected_cols if col not in df.columns]

if missing_cols:
    st.error(f"Kolom berikut tidak ditemukan di dataset: {missing_cols}")
else:
    df_numerik = df[expected_cols].dropna().copy()

# --- 3. Standardisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numerik)

# --- 4. Klastering KMeans
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)  # n_init penting untuk versi terbaru
labels = kmeans.fit_predict(X_scaled)
df_numerik['cluster'] = labels

# --- 5. PCA untuk visualisasi
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df_numerik['pca1'] = pca_result[:, 0]
df_numerik['pca2'] = pca_result[:, 1]

# --- 6. Visualisasi klaster
st.subheader("📌 Visualisasi Hasil Klaster")
fig, ax = plt.subplots()
sns.scatterplot(data=df_numerik, x='pca1', y='pca2', hue='cluster', palette='Set2', ax=ax)
st.pyplot(fig)

# --- 7. Tampilkan data hasil klaster
st.subheader("📄 Contoh Data Terklaster")
st.dataframe(df_numerik.head())

