import streamlit as st
from pages import 1_Background, 2_Data_Visualization, 3_Clustering, Prediction_Form

st.set_page_config(page_title="Analisis Klastering Data", layout="wide")

st.title("Proyek Analisis Data dengan Klastering")
st.write("""
Selamat datang di aplikasi analisis data kami. Aplikasi ini menyediakan:
1. Eksplorasi dataset awal
2. Visualisasi data interaktif
3. Analisis model klastering
4. Formulir prediksi berbasis model
""")

page = st.sidebar.selectbox(
    "Pilih Halaman",
    ("Home", "Data dan EDA", "Visualisasi Data", "Model Klastering", "Prediksi")
)

if page == "Home":
    st.write("## Tentang Proyek Ini")
    st.write("""
    Proyek ini bertujuan untuk menganalisis dataset [nama dataset] dengan teknik klastering.
    """)
elif page == "Data dan EDA":
    1_Background.main()
elif page == "Visualisasi Data":
    2_Data_Visualization.main()
elif page == "Model Klastering":
    3_Clustering.main()
elif page == "Prediksi":
    Prediction_Form.main()
