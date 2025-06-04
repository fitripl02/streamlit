import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("semarang_resto_dataset.csv")

data = load_data()

# Halaman 2: Model & Prediksi
elif page == "Model & Prediksi":
    st.title("📊 Halaman 2: Hasil Pelatihan Model & Prediksi")

    # Persiapan data
    df_model = data.dropna()
    df_model['high_rating'] = (df_model['rating'] >= 4.5).astype(int)

    fitur_model = ['average_operation_hours', 'wifi_facility', 'toilet_facility', 'cash_payment_only']

    # --- Klasifikasi: Rating Tinggi ---
    st.subheader("Model Klasifikasi - Prediksi Rating Tinggi (≥ 4.5)")
    X_clf = df_model[fitur_model]
    y_clf = df_model['high_rating']

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_c, y_train_c)
    y_pred_c = clf.predict(X_test_c)

    st.text("Classification Report:")
    st.text(classification_report(y_test_c, y_pred_c))

    # --- Regresi: Jumlah Pengunjung ---
    st.subheader("Model Regresi - Prediksi Jumlah Pengunjung")
    df_model = df_model.dropna(subset=['jumlah_pengunjung'])
    X_reg = df_model[fitur_model]
    y_reg = df_model['jumlah_pengunjung']

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(random_state=42)
    reg.fit(X_train_r, y_train_r)
    y_pred_r = reg.predict(X_test_r)

    mse = mean_squared_error(y_test_r, y_pred_r)
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")

    # --- Formulir Prediksi ---
    st.subheader("🔮 Formulir Prediksi")
    st.sidebar.header("Masukkan Fitur Restoran")
    input_vals = {f: st.sidebar.number_input(f, value=float(data[f].mean())) for f in fitur_model}

    if st.sidebar.button("Prediksi"):
        input_df = pd.DataFrame([input_vals])
        pred_rating = clf.predict(input_df)[0]
        pred_pengunjung = reg.predict(input_df)[0]

        st.success("📈 Hasil Prediksi")
        st.write(f"• Prediksi Rating Tinggi: {'✅ Ya' if pred_rating else '❌ Tidak'}")
        st.write(f"• Prediksi Jumlah Pengunjung: {int(pred_pengunjung)} orang")

