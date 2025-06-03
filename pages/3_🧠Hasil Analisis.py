import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib

st.title("📊 Halaman: Hasil Pelatihan Model Restoran")

# === Load model dan data validasi ===
try:
    model = joblib.load("resto_model.pkl")
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv").squeeze()
except:
    st.warning("⚠️ Model atau data validasi tidak ditemukan. Menampilkan simulasi.")

    # Simulasi prediksi
    y_test = np.random.choice([0, 1], size=200, p=[0.7, 0.3])
    y_pred = y_test.copy()
    noise = np.random.choice([0, 1], size=200, p=[0.9, 0.1])
    y_pred = np.abs(y_pred - noise)

    # Simulasi feature importance
    feature_importance = pd.Series({
        "Rating": 0.30,
        "Harga": 0.25,
        "Jenis Restoran": 0.20,
        "Jumlah Review": 0.15,
        "Lokasi": 0.10
    }).sort_values(ascending=True)
else:
    y_pred = model.predict(X_test)

    feature_importance = pd.Series(
        model.feature_importances_,
        index=X_test.columns
    ).sort_values(ascending=True)

# === Evaluasi ===
st.subheader("📈 Evaluasi Model")

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

st.write(f"""
- **Akurasi**: {acc:.2f}  
- **Precision**: {prec:.2f}  
- **Recall**: {rec:.2f}  
- **F1-score**: {f1:.2f}
""")

report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
st.dataframe(report_df.style.format(precision=2))

# === Confusion Matrix ===
st.subheader("🧮 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
st.pyplot(fig)

# === Feature Importance ===
st.subheader("📌 Fitur yang Paling Mempengaruhi")
fig2, ax2 = plt.subplots()
feature_importance.plot(kind="barh", ax=ax2, color='orange')
ax2.set_xlabel("Tingkat Kepentingan")
ax2.set_ylabel("Fitur")
st.pyplot(fig2)
