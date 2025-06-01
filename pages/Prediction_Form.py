import streamlit as st
import pandas as pd
import joblib
from Module.preprocessing import preprocess_input

def main():
    st.title("Formulir Prediksi Klaster")
    
    # Load model
    model = joblib.load('model/kmeans_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    
    st.header("Masukkan Data")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            feature1 = st.number_input("Fitur 1", min_value=0.0, max_value=100.0, value=50.0)
            feature2 = st.number_input("Fitur 2", min_value=0.0, max_value=100.0, value=50.0)
        
        with col2:
            feature3 = st.number_input("Fitur 3", min_value=0.0, max_value=100.0, value=50.0)
            feature4 = st.number_input("Fitur 4", min_value=0.0, max_value=100.0, value=50.0)
        
        submitted = st.form_submit_button("Prediksi")
        
        if submitted:
            input_data = pd.DataFrame([[feature1, feature2, feature3, feature4]], 
                                    columns=['feature1', 'feature2', 'feature3', 'feature4'])
            
            # Preprocess
            processed_data = preprocess_input(input_data, scaler)
            
            # Predict
            cluster = model.predict(processed_data)[0]
            
            st.success(f"Data masuk ke dalam Klaster {cluster}")
            
            st.write("### Karakteristik Klaster Ini:")
            cluster_info = {
                0: "Deskripsi klaster 0...",
                1: "Deskripsi klaster 1...",
                2: "Deskripsi klaster 2..."
            }
            st.write(cluster_info.get(cluster, "Tidak ada informasi"))

if __name__ == "__main__":
    main()
