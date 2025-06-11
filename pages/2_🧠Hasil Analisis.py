import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score  # Impor yang ditambahkan
from xgboost import XGBRegressor

st.set_page_config(page_title="Machine Learning Models", layout="wide", page_icon="⚙️")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    :root {
        --primary-color: #2563eb;
        --primary-light: #3b82f6;
        --primary-lighter: #60a5fa;
        --primary-lightest: #dbeafe;
        --secondary-color: #1e40af;
        --secondary-light: #1d4ed8;
        --accent-color: #0ea5e9;
        --accent-light: #38bdf8;
        --neutral-50: #f8fafc;
        --neutral-100: #f1f5f9;
        --neutral-200: #e2e8f0;
        --neutral-600: #475569;
        --neutral-700: #334155;
        --neutral-800: #1e293b;
        --success-color: #059669;
        --success-light: #10b981;
    }
    
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(37, 99, 235, 0.15);
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 3.2em;
        margin: 0;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .main-subtitle {
        font-size: 1.3em;
        margin-top: 12px;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 400;
    }
    
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.2em;
        }
        
        .main-subtitle {
            font-size: 1.1em;
        }
    }
</style>
""", unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('semarang_resto_dataset.csv')
        df['resto_type'] = df['resto_type'].fillna('Tidak Diketahui')
        for col in df.columns:
            if df[col].dtype == 'int64':
                df[col] = df[col].astype('float64')
            elif df[col].dtype == 'object':
                df[col] = df[col].astype('string')
        return df
    except FileNotFoundError:
        st.error("File 'semarang_resto_dataset.csv' tidak ditemukan! Pastikan file ada di direktori yang sama.")
        st.stop()

# Function to load label encoder
def load_label_encoder():
    try:
        le = joblib.load('model/label_encoder.pkl')
        return le
    except FileNotFoundError:
        st.error("File label_encoder.pkl tidak ditemukan! Pastikan file ada di folder model/.")
        st.stop()

st.markdown("""
<div class="main-header">
    <h1 class="main-title">⚙️Models</h1>
    <p class="main-subtitle">Training dan Evaluasi Model Prediksi</p>
</div>
""", unsafe_allow_html=True)

st.title("📈 Hasil Pelatihan Model")

df = load_data()
le = load_label_encoder()

df = df.drop_duplicates(subset=['resto_name', 'resto_address'], keep='first')
features = ['average_operation_hours', 'sell_halal_food', 'wifi_facility', 'toilet_facility', 
            'suitable_for_children', 'dine_in', 'take_away', 'delivery', 'open_space', 'resto_type']
df['resto_type_encoded'] = le.transform(df['resto_type'])
X = df[features].copy()
X['resto_type'] = df['resto_type_encoded']
y = df['resto_rating']
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train XGBoost model with GridSearchCV
st.subheader("Pelatihan Model")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'min_child_weight': [1, 3]
}
xgb_model = XGBRegressor(random_state=42)
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluate model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')

st.subheader("Performa Model")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Mean Squared Error**: {mse:.3f}")
with col2:
    st.write(f"**R² Score**: {r2:.3f}")
st.write(f"**R² Cross-Validation (mean)**: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
st.write(f"**Best Parameters**: {grid_search.best_params_}")

# Feature Importance
st.subheader("Pentingnya Fitur")
importance = pd.DataFrame({'Fitur': X.columns, 'Pentingnya': best_model.feature_importances_})
fig = px.bar(importance, x='Fitur', y='Pentingnya', title="Pentingnya Fitur")
fig.update_layout(xaxis_title="Fitur", yaxis_title="Pentingnya", xaxis_tickangle=45)
st.plotly_chart(fig, use_container_width=True)

# Prediction vs Actual
st.subheader("Prediksi vs Rating Aktual")
fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Rating Aktual', 'y': 'Rating Prediksi'}, title="Prediksi vs Aktual")
fig.add_shape(type='line', x0=0, y0=0, x1=5, y1=5, line=dict(color='red', dash='dash'))
st.plotly_chart(fig, use_container_width=True)

# Residual Plot
st.subheader("Residual Plot")
residuals = y_test - y_pred
fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Rating Prediksi', 'y': 'Residual'}, title="Residual Plot")
fig.add_hline(y=0, line_dash="dash", line_color="red")
st.plotly_chart(fig, use_container_width=True)
