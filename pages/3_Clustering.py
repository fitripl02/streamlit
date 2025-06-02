import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🤖 Halaman 2: Klastering Restoran (KMeans)")

url = "https://raw.githubusercontent.com/fitripl02/streamlit/refs/heads/main/semarang_resto_dataset.csv"
df = pd.read_csv(url)

# Ambil kolom numerik 
features = ['resto_rating', 'operation_hours', 'wifi', 'toilet', 'cash_only']
df_model = df[features].dropna()

# Standardisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model)

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df_model['cluster'] = clusters

# PCA untuk visualisasi 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df_model['pca1'] = pca_result[:, 0]
df_model['pca2'] = pca_result[:, 1]

# Visualisasi Klaster
st.subheader("📌 Visualisasi Hasil Klaster")
fig, ax = plt.subplots()
sns.scatterplot(data=df_model, x='pca1', y='pca2', hue='cluster', palette='Set1', ax=ax)
st.pyplot(fig)

st.subheader("📄 Contoh Data Terklaster")
st.dataframe(df_model.head())

