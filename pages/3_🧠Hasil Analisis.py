import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

st.title("📊 Halaman Hasil Pelatihan Model Restoran Semarang")

# === 1. Load Model dan Data Validasi ===
try:
    model = joblib.load("semarang_model.pkl")
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv").squeeze()
except:
    st.warning("Data validasi tidak tersedia. Menampilkan contoh simulasi.")

    y_test = np.random.choice(["Tinggi", "Sedang", "Rendah"], size=200)
    y_pred = np.random.choice(["Tinggi", "Sedang", "Rendah"], size=200)
    feature_importance = pd.Series({
        "Harga Rata-rata": 0.3,
        "Rating Pelanggan": 0.25,
        "Jumlah Review": 0.2,
        "Jenis Masakan": 0.15,
        "Lokasi": 0.1
    }).sort_values(ascending=True)
else:
    y_pred = model.predict(X_test)
    feature_importance = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=True)

# === 2. Evaluasi Model ===
st.subheader("📈 Evaluasi Kinerja Model")

acc = accuracy_score(y_test, y_pred)
st.write(f"- **Akurasi**: {acc:.2f}")

report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
st.dataframe(report_df.style.format(precision=2))

# === 3. Confusion Matrix ===
st.subheader("🧮 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred, labels=report_df.index[:-3])
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=report_df.index[:-3], yticklabels=report_df.index[:-3])
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
st.pyplot(fig)

# === 4. Feature Importance ===
st.subheader("📌 Fitur yang Paling Berpengaruh")
fig2, ax2 = plt.subplots()
feature_importance.plot(kind="barh", ax=ax2, color='skyblue')
ax2.set_xlabel("Tingkat Kepentingan")
ax2.set_ylabel("Fitur")
st.pyplot(fig2)
