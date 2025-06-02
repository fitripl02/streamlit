import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("🤖 Halaman 2: Hasil Pelatihan Model Klasterisasi")

# --- 1. Load data
url = "https://raw.githubusercontent.com/fitripl02/streamlit/refs/heads/main/semarang_resto_dataset.csv"
df = pd.read_csv(url)

# --- 2. Kolom numerik untuk clustering
numerik_cols = ['rating', 'rating_number', 'operating_hours', 'wifi', 'toilet', 'cash_only', 'debit_card']
df_numerik = df[numerik_cols].dropna().copy()

# --- 3. Standardisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numerik)

# --- 4. KMeans
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df_numerik['cluster'] = labels

# --- 5. PCA untuk visualisasi 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df_numerik['pca1'] = pca_result[:, 0]
df_numerik['pca2'] = pca_result[:, 1]

# --- 6. Visualisasi klaster
st.subheader("📍 Visualisasi Klaster dalam 2D (PCA)")
fig, ax = plt.subplots()
sns.scatterplot(data=df_numerik, x='pca1', y='pca2', hue='cluster', palette='Set2', ax=ax)
plt.title("Visualisasi Klaster Restoran")
plt.xlabel("PCA Komponen 1")
plt.ylabel("PCA Komponen 2")
st.pyplot(fig)

# --- 7. Hasil Model: Centroid, Inertia, Distribusi
st.subheader("📈 Hasil Pelatihan Model KMeans")

# Inertia (evaluasi internal)
st.markdown(f"- **Skor inertia**: {kmeans.inertia_:.2f}")

# Distribusi klaster
st.markdown("- **Jumlah anggota per klaster**:")
st.write(df_numerik['cluster'].value_counts().sort_index())

# Centroid (kembalikan ke skala asli)
st.subheader("📌 Titik Pusat Tiap Klaster (Centroid)")
centroids = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=numerik_cols
)
centroids.index = [f"Klaster {i}" for i in range(kmeans.n_clusters)]
st.dataframe(centroids)

# --- 8. Tabel hasil
st.subheader("📄 Contoh Data Terklaster")
st.dataframe(df_numerik.head(10))

