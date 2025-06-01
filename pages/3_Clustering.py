import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from Module.data_loader import load_data
from Module.preprocessing import preprocess_data
from Module.visualization import plot_clusters

def main():
    st.title("Analisis Model Klastering")
    
    # Load data
    data = load_data()
    if data.empty:
        st.error("Data tidak berhasil dimuat. Periksa sumber data Anda.")
        return
    
    # Preprocessing
    try:
        X = preprocess_data(data)
        if X.empty:
            st.error("Preprocessing menghasilkan data kosong. Periksa fungsi preprocessing.")
            return
    except Exception as e:
        st.error(f"Error dalam preprocessing: {str(e)}")
        return
    
    # Standardisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.header("1. Pemilihan Jumlah Klaster")
    
    # Metode Elbow dan Silhouette
    st.write("### Metode Elbow dan Silhouette Score")
    k_values = range(2, 11)
    distortions = []
    silhouette_scores = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        distortions.append(kmeans.inertia_)
        if k > 1:  # Silhouette score hanya untuk k > 1
            silhouette_scores.append(silhouette_score(X_scaled, clusters))
    
    # Plot Elbow dan Silhouette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow Method
    ax1.plot(k_values, distortions, 'bx-')
    ax1.set_xlabel('Jumlah Klaster (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Metode Elbow')
    
    # Silhouette Score
    ax2.plot(k_values[1:], silhouette_scores, 'rx-')
    ax2.set_xlabel('Jumlah Klaster (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score')
    
    st.pyplot(fig)
    
    # Pemilihan jumlah klaster
    optimal_k = st.slider(
        "Pilih jumlah klaster berdasarkan plot di atas:", 
        min_value=2, 
        max_value=10, 
        value=3
    )
    
    st.header("2. Hasil Klastering")
    
    # Training model dengan k terpilih
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Menambahkan klaster ke data asli
    data['Cluster'] = clusters
    
    # Evaluasi model
    st.write(f"### Silhouette Score: {silhouette_score(X_scaled, clusters):.3f}")
    st.write(f"### Inertia: {kmeans.inertia_:.2f}")
    
    # Visualisasi klaster
    st.write("### Visualisasi Klaster (Pilih 2 Fitur)")
    
    # Pilihan kolom numerik untuk visualisasi
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    default_cols = numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Sumbu X:", numeric_cols, index=0)
    with col2:
        y_axis = st.selectbox("Sumbu Y:", numeric_cols, 
                             index=1 if len(numeric_cols) > 1 else 0)
    
    if x_axis != y_axis:
        fig = plot_clusters(
            X, clusters, 
            x_axis, y_axis,
            title=f"Klastering dengan K={optimal_k}"
        )
        st.plotly_chart(fig)
    else:
        st.warning("Pilih dua fitur yang berbeda untuk visualisasi")
    
    st.header("3. Interpretasi Klaster")
    
    # Statistik deskriptif per klaster
    st.write("### Karakteristik Tiap Klaster")
    cluster_stats = data.groupby('Cluster').agg(['mean', 'std', 'count'])
    st.dataframe(cluster_stats)
    
    # Rekomendasi jumlah klaster
    if optimal_k >= 2:
        best_silhouette = max(silhouette_scores)
        best_k = k_values[1:][silhouette_scores.index(best_silhouette)]
        if optimal_k != best_k:
            st.info(
                f"Berdasarkan Silhouette Score, jumlah klaster optimal mungkin {best_k} "
                f"(score: {best_silhouette:.3f})"
            )

if __name__ == "__main__":
    main()
