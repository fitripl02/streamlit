import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from Module.data_loader import load_data
from Module.preprocessing import clean_data
from Module.visualization import show_distributions

def main():
    st.title("Dataset dan Analisis Eksploratori (EDA)")
    
    # Load data
    data = load_data()
    
    st.header("1. Tinjauan Dataset")
    st.write("### 5 Baris Pertama Data")
    st.dataframe(data.head())
    
    st.write("### Statistik Deskriptif")
    st.dataframe(data.describe())
    
    st.write("### Informasi Dataset")
    buffer = io.StringIO()
    data.info(buf=buffer)
    st.text(buffer.getvalue())
    
    st.header("2. Karakteristik Data")
    st.write("### Nilai yang Hilang")
    missing = data.isna().sum()
    st.bar_chart(missing[missing > 0])
    
    st.write("### Distribusi Variabel Numerik")
    num_cols = data.select_dtypes(include=np.number).columns
    selected_col = st.selectbox("Pilih variabel untuk dilihat distribusinya", num_cols)
    show_distributions(data, selected_col)
    
    st.header("3. Korelasi Antar Variabel")
    corr = data.select_dtypes(include=np.number).corr()  # Hanya hitung korelasi untuk numerik
    st.write("### Matriks Korelasi")
    st.dataframe(corr)
    
    st.write("### Heatmap Korelasi")
    fig, ax = plt.subplots(figsize=(10, 8))  # Tambahkan ukuran figure
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
