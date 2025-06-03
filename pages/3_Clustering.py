import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib

st.title("🔄 Halaman 4: Hasil Analisis Churn Pelanggan")

# Simulasi data churn (jika belum ada data asli)
st.info("📌 Contoh simulasi hasil model churn pelanggan")

# Simulasi data prediksi & hasil
y_true = np.random.choice([0, 1], size=200, p=[0.7, 0.3])  # 0 = loyal, 1 = churn
y_pred = y_true.copy()
noise = np.random.choice([0, 1], size=200, p=[0.9, 0.1])
y_pred = np.abs(y_pred - noise)  # simulasikan prediksi keliru

# Classification report
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.subheader("📊 Classification Report")
st.dataframe(report_df.style.format(precision=2))

# Confusion matrix
st.subheader("🧮 Confusion Matrix")
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Loyal", "Churn"], yticklabels=["Loyal", "Churn"])
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
st.pyplot(fig)

# Simulasi feature importance
st.subheader("📌 Fitur yang Paling Mempengaruhi Churn")
feature_importance = pd.Series({
    "Frekuensi Kunjungan": 0.35,
    "Rata-rata Rating": 0.25,
    "Jenis Restoran": 0.15,
    "Jumlah Transaksi": 0.12,
    "Area Lokasi": 0.13
}).sort_values(ascending=True)

fig2, ax2 = plt.subplots()
feature_importance.plot(kind="barh", ax=ax2, color='teal')
ax2.set_xlabel("Tingkat Pengaruh")
st.pyplot(fig2)

