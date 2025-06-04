import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Evaluasi Model Klasifikasi", layout="wide")
st.title("🎯 Evaluasi Model: Prediksi Restoran dengan Rating Tinggi (≥ 4.5)")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("semarang_resto_dataset.csv")

data = load_data()

# Buat label klasifikasi: high_rating = 1 jika rating >= 4.5
data = data.dropna(subset=['rating'])
data['high_rating'] = (data['rating'] >= 4.5).astype(int)

# Pilih fitur untuk model
fitur = ['average_operation_hours', 'wifi_facility', 'toilet_facility', 'cash_payment_only']
X = data[fitur]
y = data['high_rating']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi model
st.subheader("📋 Classification Report")
st.code(classification_report(y_test, y_pred), language='text')

# Confusion matrix
st.subheader("📊 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "High"], yticklabels=["Low", "High"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Fitur penting
st.subheader("🔍 Pentingnya Fitur")
importances = model.feature_importances_
feat_importance = pd.DataFrame({
    'Fitur': fitur,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

st.dataframe(feat_importance)

