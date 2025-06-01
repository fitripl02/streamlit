import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from Module.data_loader import load_data
from Module.visualization import plot_interactive

def main():
    st.title("Visualisasi Data Interaktif")
    data = load_data()
    
    st.header("1. Visualisasi Dasar")
    st.write("### Histogram Interaktif")
    column = st.selectbox("Pilih kolom untuk histogram:", data.columns)
    bins = st.slider("Jumlah bins:", 5, 100, 20)
    
    fig, ax = plt.subplots()
    ax.hist(data[column], bins=bins)
    ax.set_title(f"Distribusi {column}")
    st.pyplot(fig)
    
    st.header("2. Scatter Plot Interaktif")
    col1 = st.selectbox("Pilih sumbu X:", data.columns, index=0)
    col2 = st.selectbox("Pilih sumbu Y:", data.columns, index=1)
    
    fig = plot_interactive(data, x=col1, y=col2)
    st.plotly_chart(fig)
    
    st.header("3. Analisis Multivariat")
    st.write("### Pair Plot (5 variabel pertama)")
    sample_data = data.iloc[:, :5]
    fig = sns.pairplot(sample_data)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
