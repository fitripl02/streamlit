import streamlit as st
import pandas as pd
import joblib
from Module.preprocessing import preprocess_input

def main():
    st.title("Formulir Prediksi Klaster")
    
    # Load model dengan error handling
    try:
        model = joblib.load('model/kmeans_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
    except FileNotFoundError:
        st.error("Model atau scaler tidak ditemukan. Pastikan file model ada di direktori 'model/'")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {str(e)}")
        st.stop()
    
    st.header("Masukkan Data")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            feature1 = st.number_input("Fitur 1", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
            feature2 = st.number_input("Fitur 2", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        
        with col2:
            feature3 = st.number_input("Fitur 3", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
            feature4 = st.number_input("Fitur 4", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        
        submitted = st.form_submit_button("Prediksi")
        
        if submitted:
            input_data = pd.DataFrame([[feature1, feature2, feature3, feature4]], 
                                    columns=['feature1', 'feature2', 'feature3', 'feature4'])
            
            try:
                # Preprocess
                processed_data = preprocess_input(input_data, scaler)
                
                # Predict
                cluster = model.predict(processed_data)[0]
                
                # Menampilkan hasil dengan lebih menarik
                st.success(f"**Hasil Prediksi:** Data masuk ke dalam Klaster {cluster}")
                
                # Informasi klaster yang lebih detail
                st.subheader(f"Karakteristik Klaster {cluster}")
                
                cluster_info = {
                    0: {
                        "deskripsi": "Klaster dengan nilai fitur 1 dan 2 yang tinggi",
                        "karakteristik": ["Fitur 1: Rata-rata tinggi", "Fitur 2: Variasi sedang"],
                        "rekomendasi": "Perhatikan pola khusus dari kelompok ini"
                    },
                    1: {
                        "deskripsi": "Klaster dengan nilai fitur 3 yang dominan",
                        "karakteristik": ["Fitur 3: Nilai konsisten tinggi", "Fitur 4: Variasi rendah"],
                        "rekomendasi": "Kelompok ini memiliki pola yang stabil"
                    },
                    2: {
                        "deskripsi": "Klaster dengan nilai fitur 4 yang unik",
                        "karakteristik": ["Fitur 4: Distribusi lebar", "Fitur 1: Nilai rendah"],
                        "rekomendasi": "Perlu analisis lebih lanjut untuk kelompok ini"
                    }
                }
                
                info = cluster_info.get(cluster, {
                    "deskripsi": "Informasi tidak tersedia",
                    "karakteristik": [],
                    "rekomendasi": ""
                })
                
                st.markdown(f"**Deskripsi:** {info['deskripsi']}")
                
                if info['karakteristik']:
                    st.markdown("**Karakteristik:**")
                    for char in info['karakteristik']:
                        st.markdown(f"- {char}")
                
                if info['rekomendasi']:
                    st.markdown(f"**Rekomendasi:** {info['rekomendasi']}")
                
                # Tambahkan visualisasi sederhana
                st.subheader("Posisi Relatif dalam Klaster")
                st.write("Berikut adalah distribusi nilai fitur Anda dibandingkan klaster ini:")
                
                # Contoh visualisasi sederhana
                comparison_data = pd.DataFrame({
                    'Fitur': ['Fitur 1', 'Fitur 2', 'Fitur 3', 'Fitur 4'],
                    'Nilai Anda': [feature1, feature2, feature3, feature4],
                    'Rata-rata Klaster': [50, 60, 40, 70]
                })
                
                st.bar_chart(comparison_data.set_index('Fitur'))
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data: {str(e)}")

if __name__ == "__main__":
    main()
