import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from Module.data_loader import load_data
from Module.preprocessing import preprocess_data
from Module.visualization import plot_clusters

def main():
    st.title("Analisis Model Klastering")
    data = load_data()
    X = preprocess_data(data)
    
    st.header("1. Pemilihan Jumlah Klaster")
    st.write("### Metode Elbow")
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(X)
        distortions.append(kmeanModel.inertia_)
    
    fig, ax = plt.subplots()
    ax.plot(K, distortions, 'bx-')
    ax.set_xlabel('k')
    ax.set_ylabel('Distortion')
    ax.set_title('Metode Elbow untuk Optimal k')
    st.pyplot(fig)
    
    optimal_k = st.slider("Pilih jumlah klaster:", 2, 10, 3)
    
    st.header("2. Hasil Klastering")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    st.write(f"### Silhouette Score: {silhouette_score(X, clusters):.3f}")
    
    st.write("### Visualisasi Klaster")
    plot_cols = st.multiselect("Pilih 2 variabel untuk visualisasi:", 
                              X.columns.tolist(), 
                              default=X.columns[:2].tolist())
    
    if len(plot_cols) == 2:
        fig = plot_clusters(X, clusters, plot_cols[0], plot_cols[1])
        st.plotly_chart(fig)
    else:
        st.warning("Pilih tepat 2 variabel untuk visualisasi 2D")
    
    st.header("3. Interpretasi Klaster")
    data['Cluster'] = clusters
    cluster_stats = data.groupby('Cluster').mean()
    st.dataframe(cluster_stats)

if __name__ == "__main__":
    main()
