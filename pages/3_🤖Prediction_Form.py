import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Rating Prediction", layout="wide", page_icon="⭐")

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

# Function to load model and label encoder
def load_model():
    try:
        model = joblib.load('model/model.pkl')
        le = joblib.load('model/label_encoder.pkl')
        return model, le
    except FileNotFoundError:
        st.error("File model tidak ditemukan! Pastikan model.pkl dan label_encoder.pkl ada di folder model/.")
        st.stop()

# Initialize session state to store prediction results
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None

st.markdown("""
<div class="main-header">
    <h1 class="main-title">⭐ Rating Prediction</h1>
    <p class="main-subtitle">Prediksi Rating Restoran Baru</p>
</div>
""", unsafe_allow_html=True)

model, le = load_model()

st.subheader("Masukkan Detail Restoran")
with st.form("formulir_prediksi"):
    col1, col2 = st.columns(2)
    with col1:
        resto_type = st.selectbox("Jenis Restoran", le.classes_)
        avg_hours = st.number_input("Rata-rata Jam Operasional", min_value=0.0, max_value=24.0, value=10.0)
        halal_food = st.checkbox("Menyediakan Makanan Halal")
        wifi = st.checkbox("Fasilitas WiFi")
        toilet = st.checkbox("Fasilitas Toilet")
    with col2:
        children = st.checkbox("Cocok untuk Anak-anak")
        dine_in = st.checkbox("Tersedia Dine-In")
        take_away = st.checkbox("Tersedia Take-Away")
        delivery = st.checkbox("Tersedia Delivery")
        open_space = st.checkbox("Tersedia Ruang Terbuka")
    submitted = st.form_submit_button("Prediksi Rating")

    if submitted:
        if avg_hours <= 0:
            st.warning("Jam operasional harus lebih dari 0!")
        else:
            input_data = pd.DataFrame({
                'average_operation_hours': [avg_hours],
                'sell_halal_food': [1 if halal_food else 0],
                'wifi_facility': [1 if wifi else 0],
                'toilet_facility': [1 if toilet else 0],
                'suitable_for_children': [1 if children else 0],
                'dine_in': [1 if dine_in else 0],
                'take_away': [1 if take_away else 0],
                'delivery': [1 if delivery else 0],
                'open_space': [1 if open_space else 0],
                'resto_type': [le.transform([resto_type])[0]]
            })

            try:
                prediction = model.predict(input_data)[0]
                st.subheader("Hasil Prediksi")
                st.markdown(f"<h3 style='color: var(--primary-color);'>Rating Restoran: {prediction:.2f}/5 ⭐</h3>", unsafe_allow_html=True)

                fig = px.pie(values=[prediction, 5-prediction], names=['Rating', 'Sisa'], hole=0.4,
                             title="Visualisasi Rating", color_discrete_sequence=['var(--primary-color)', 'var(--neutral-200)'])
                st.plotly_chart(fig, use_container_width=True)

                # Store prediction data in session state
                st.session_state.prediction_data = pd.DataFrame({
                    'Resto_Type': [resto_type],
                    'Average_Operation_Hours': [avg_hours],
                    'Predicted_Rating': [prediction]
                })
            except Exception as e:
                st.error(f"Terjadi error dalam prediksi: {str(e)}")

# Display download button outside of the form if prediction data exists
if st.session_state.prediction_data is not None:
    csv = st.session_state.prediction_data.to_csv(index=False)
    st.download_button("Unduh Hasil Prediksi", csv, "prediction_result.csv", "text/csv", use_container_width=True)
