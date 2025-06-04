import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('semarang_resto_dataset.csv')
    return data

data = load_data()
# Halaman 2: Model Training
elif page == "Model Training":
    st.title("Pelatihan Model Prediksi Rating Restoran")
    
    # Persiapan data
    st.write("### Persiapan Data untuk Model")
    
    # Pilih fitur dan target
    features = ['resto_type', 'rating_numbers', 'average_operation_hours', 
                'cash_payment_only', 'debit_card_payment', 'credit_card_payment',
                'wifi_facility', 'sell_halal_food', 'vegetarian_menu', 'delivery']
    target = 'resto_rating'
    
    # Encode categorical features
    le = LabelEncoder()
    data['resto_type_encoded'] = le.fit_transform(data['resto_type'])
    
    # Bagi data menjadi fitur dan target
    X = data[['resto_type_encoded', 'rating_numbers', 'average_operation_hours', 
              'cash_payment_only', 'debit_card_payment', 'credit_card_payment',
              'wifi_facility', 'sell_halal_food', 'vegetarian_menu', 'delivery']]
    y = data[target]
    
    # Bagi data menjadi train dan test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.write(f"Jumlah data training: {X_train.shape[0]}")
    st.write(f"Jumlah data testing: {X_test.shape[0]}")
    
    # Latih model
    st.write("### Pelatihan Model Random Forest")
    
    # Parameter model
    n_estimators = st.slider("Jumlah Estimator", 10, 200, 100)
    max_depth = st.slider("Kedalaman Maksimum", 2, 20, 10)
    
    if st.button("Latih Model"):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.write("### Hasil Evaluasi Model")
        st.write(f"Akurasi Model: {accuracy:.2f}")
        
        st.write("#### Laporan Klasifikasi")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write(report_df)
        
        # Feature importance
        st.write("#### Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
        ax.set_title('Feature Importance')
        st.pyplot(fig)
