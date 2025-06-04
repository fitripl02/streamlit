import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Halaman 3: Hasil Analisis", layout="wide")
st.title("🔄 Halaman 3: Hasil Evaluasi Model Klasifikasi Rating Tinggi")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("semarang_resto_dataset.csv")

data = load_data()

# Siapkan data klasifikasi: Rating tinggi >= 4.5
data = data.dropna(subset=['rating'])
data['high_rating'] = (data['rating'] >= 4.5).astype(int)

# Fitur yang digunakan
features = ['average_operation_hours', 'wifi_facility', 'toilet_facility', 'cash_payment_only']
X = data[features]
y = data['high_rating']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ======= 📊 Classification Report =======
st.subheader("📊 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format(precision=2))

# ======= 🧮 Confusion Matrix =======
st.subheader("🧮 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
labels = ["Rating < 4.5", "Rating ≥ 4.5"]

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel("Prediksi")
ax.set_ylabel("Aktual")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# ======= 📌 Feature Importance =======
st.subheader("📌 Fitur yang Paling Mempengaruhi Prediksi Rating")
importances = model.feature_importances_
feat_df = pd.Series(importances, index=features).sort_values()

fig2, ax2 = plt.subplots()
feat_df.plot(kind='barh', ax=ax2, color='seagreen')
ax2.set_xlabel("Tingkat Pengaruh")
ax2.set_title("Feature Importance (Random Forest)")
st.pyplot(fig2)


