import pandas as pd
import folium
from folium.plugins import MarkerCluster

df = pd.read_csv("semarang_resto_dataset.csv")

required_columns = {'nama_resto', 'latitude', 'longitude'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Dataset harus memiliki kolom: {required_columns}")

semarang_center = [-6.9667, 110.4167]  # Koordinat pusat kota Semarang
semarang_map = folium.Map(location=semarang_center, zoom_start=12)

marker_cluster = MarkerCluster().add_to(semarang_map)

for _, row in df.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=row['nama_resto'],
        icon=folium.Icon(color='green', icon='cutlery', prefix='fa')
    ).add_to(marker_cluster)
  
semarang_map.save("peta_resto_semarang.html")
print("✅ Peta berhasil disimpan sebagai 'peta_resto_semarang.html'")
