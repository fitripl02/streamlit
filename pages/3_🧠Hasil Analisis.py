import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
import numpy as np

st.set_page_config(page_title="Evaluasi Model", layout="wide")
st.title("📉 Evaluasi Kinerja Model Restoran")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("semarang_resto_dataset.csv")

data = load_data()

# Persiapan data
st.subheader("🔍 Persiapan Data")
df_model = data.dropna()
df_model['high_rating'] = (df_model['rating'] >= 4.5).astype(int)
fitur_model = ['average_operation_hours', 'wifi_facility', 'toilet_facility', 'cash_payment_only']

# -------------------------
# 🎯 Model Klasifikasi
# -------------------------
st.subheader("🎯 Klasifikasi: Prediksi Rating Tinggi")

X_clf = df_model[fitur_model]
y_clf = df_model['high_rating']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)

st.text("Classification Report:")
st.code(classification_report(y_test_c, y_pred_c), language='text')

# -------------------------
# 📏 Model Regresi
# -------------------------
st.subheader("📏 Regresi: Prediksi Jumlah Pengunjung")

df_model = df_model.dropna(subset=['jumlah_pengunjung'])
X_reg = df_model[fitur_model]
y_reg = df_model['jumlah_pengunjung']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg = RandomForestRegressor(random_state=42)
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)

mse = mean_squared_error(y_test_r, y_pred_r)
st.write(f"**Mean Squared Error (MSE)**: `{mse:.2f}`")

fig, ax = plt.subplots()
sns.scatterplot(x=y_test_r, y=y_pred_r, ax=ax)
ax.set_xlabel("Aktual")
ax.set_ylabel("Prediksi")
ax.set_title("Prediksi vs Aktual - Jumlah Pengunjung")
st.pyplot(fig)
