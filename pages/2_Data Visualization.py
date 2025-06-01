import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # Ditambahkan untuk plot interaktif
from Module.data_loader import load_data
from Module.visualization import plot_interactive

def main():
    st.title("Visualisasi Data Interaktif")
    data = load_data()
    
    # Tambahkan pemeriksaan data kosong
    if data.empty:
        st.warning("Data kosong! Silakan periksa sumber data Anda.")
        return
    
    st.header("1. Visualisasi Dasar")
    st.write("### Histogram Interaktif")
    
    # Filter hanya kolom numerik untuk histogram
    numeric_cols = data.select_dtypes(include=['number']).columns
    column = st.selectbox("Pilih kolom untuk histogram:", numeric_cols)
    bins = st.slider("Jumlah bins:", 5, 100, 20)
    
    fig, ax = plt.subplots()
    ax.hist(data[column], bins=bins, edgecolor='black')
    ax.set_title(f"Distribusi {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig)
    
    st.header("2. Scatter Plot Interaktif")
    # Filter kolom numerik untuk scatter plot
    col1 = st.selectbox("Pilih sumbu X:", numeric_cols, index=0)
    col2 = st.selectbox("Pilih sumbu Y:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
    
    # Tambahkan opsi warna berdasarkan kolom kategorikal (jika ada)
    categorical_cols = data.select_dtypes(exclude=['number']).columns
    color_col = None
    if not categorical_cols.empty:
        color_col = st.selectbox("Pilih kolom untuk warna:", ['None'] + list(categorical_cols))
        if color_col == 'None':
            color_col = None
    
    fig = plot_interactive(data, x=col1, y=col2, color=color_col)
    st.plotly_chart(fig)
    
    st.header("3. Analisis Multivariat")
    st.write("### Pair Plot (5 variabel numerik pertama)")
    
    # Ambil 5 kolom numerik pertama
    sample_data = data[numeric_cols[:min(5, len(numeric_cols))]]
    
    # Tambahkan opsi hue jika ada kolom kategorikal
    hue_col = None
    if not categorical_cols.empty:
        hue_col = st.selectbox("Pilih kolom untuk grouping:", ['None'] + list(categorical_cols))
        if hue_col == 'None':
            hue_col = None
    
    fig = sns.pairplot(sample_data, hue=hue_col, corner=True)
    st.pyplot(fig)
    
    # Tambahkan warning jika figure terlalu besar
    if len(sample_data.columns) > 4:
        st.warning("Pair plot dengan banyak variabel mungkin sulit dibaca. Pertimbangkan untuk memilih subset data.")

if __name__ == "__main__":
    main()
