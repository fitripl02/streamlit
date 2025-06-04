import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
import numpy as np

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("semarang_resto_dataset.csv")

data = load_data()

# Sidebar Navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["EDA", "Model & Prediksi", "Formulir Prediksi"])

# Halaman 2: Model dan Form Prediksi
elif page == "Model & Prediksi":
    st.title("Model Pelatihan dan Prediksi")

    st.write("## Model Klasifikasi: Rating Tinggi")
    data_clf = data.dropna(subset=['rating'])
    data_clf['high_rating'] = (data_clf['rating'] >= 4.5).astype(int)
    X_clf = data_clf.select_dtypes(include=[np.number]).drop(columns=['rating', 'jumlah_pengunjung', 'high_rating'])
    y_clf = data_clf['high_rating']
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clf, y_clf, stratify=y_clf, random_state=42)
    clf_model = RandomForestClassifier(random_state=42)
    clf_model.fit(Xc_train, yc_train)
    yc_pred = clf_model.predict(Xc_test)
    st.text("Classification Report:")
    st.text(classification_report(yc_test, yc_pred))

    st.write("## Model Regresi: Jumlah Pengunjung")
    data_reg = data.dropna(subset=['jumlah_pengunjung'])
    X_reg = data_reg.select_dtypes(include=[np.number]).drop(columns=['rating', 'jumlah_pengunjung'])
    y_reg = data_reg['jumlah_pengunjung']
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, random_state=42)
    regr_model = RandomForestRegressor(random_state=42)
    regr_model.fit(Xr_train, yr_train)
    yr_pred = regr_model.predict(Xr_test)
    mse = mean_squared_error(yr_test, yr_pred)
    st.write("Mean Squared Error:", mse)

    st.write("## Form Prediksi")
    st.write("Masukkan fitur numerik untuk memprediksi rating tinggi dan jumlah pengunjung.")
    input_data = {}
    for col in X_clf.columns:
        input_data[col] = st.number_input(f"{col}", value=float(data[col].mean()))
    input_df = pd.DataFrame([input_data])

    pred_class = clf_model.predict(input_df)[0]
    pred_reg = regr_model.predict(input_df)[0]

    st.write("### Hasil Prediksi:")
    st.write("Prediksi Rating Tinggi:", "Ya" if pred_class else "Tidak")
    st.write("Prediksi Jumlah Pengunjung:", int(pred_reg))
