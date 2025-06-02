import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("🤖 Halaman 2: Klasterisasi Restoran di Semarang")

# --- 1. Load data dari GitHub
url = "https://raw.githubusercontent.com/fitripl02/streamlit/refs/heads/main/semarang_resto_dataset.csv" 
df = pd.read_csv(url)

# --- 2. Tampilkan kolom tersedia
st.subheader("📋 Daftar Kolom Dataset")
st.write(df.columns.tolist())

# --- 3. Pilih kolom numerik yang valid
numerik_cols = ['rating', 'rating_number', 'operating_hours', 'wifi', 'toilet', 'cash_only', 'debit_card']
df_numerik = df[numerik_cols].dropna().copy()  

# --- 4. Standardisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numerik)

# --- 5. Latih model KMeans
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df_numerik['cluster'] = labels

# --- 6. Reduksi dimensi dengan PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df_numerik['pca1'] = pca_result[:, 0]
df_numerik['pca2'] = pca_result[:, 1]

# --- 7. Visualisasi klaster
st.subheader("🎯 Visualisasi Klaster Restoran")
fig, ax = plt.subplots()
sns.scatterplot(data=df_numerik, x='pca1', y='pca2', hue='cluster', palette='Set2', ax=ax)
plt.title("Peta 2D Klaster Restoran")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
st.pyplot(fig)

# --- 8. Tampilkan data hasil klaster
st.subheader("📄 Contoh Data Terklaster")
st.dataframe(df_numerik.head(10))

