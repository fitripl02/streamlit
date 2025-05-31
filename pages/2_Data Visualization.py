import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import streamlit as st

st.set_page_config(page_title="Peta Restoran Semarang", layout="wide")
st.title("Peta Persebaran Restoran di Semarang")

# Load dataset
file_path = '/mnt/data/semarang_resto_dataset.csv'
df = pd.read_csv(semarang_resto_dataset)

# Tampilkan jumlah data
st.markdown(f"Jumlah restoran: **{len(df)}**")

# Validasi kolom
required_cols = {'nama_resto', 'latitude', 'longitude'}
if not required_cols.issubset(df.columns):
    st.error("Dataset harus memiliki kolom: nama_resto, latitude, longitude")
    st.stop()

# Buat peta folium
map_center = [-6.9667, 110.4167]  # Koordinat Semarang
m = folium.Map(location=map_center, zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)

# Tambahkan marker
for _, row in df.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=row['nama_resto'],
        icon=folium.Icon(color='blue', icon='cutlery', prefix='fa')
    ).add_to(marker_cluster)

# Tampilkan peta
st_data = st_folium(m, width=900, height=600)
