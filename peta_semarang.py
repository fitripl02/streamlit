import pandas as pd
import folium
from folium.plugins import MarkerCluster

# Load data
df = pd.read_csv("semarang_resto_dataset.csv")

# Validasi kolom
if not {'nama_resto', 'latitude', 'longitude'}.issubset(df.columns):
    raise ValueError("Dataset harus memiliki kolom: nama_resto, latitude, longitude")

# Buat peta
map_center = [-6.9667, 110.4167]
m = folium.Map(location=map_center, zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)

for _, row in df.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=row['nama_resto'],
        icon=folium.Icon(color='blue', icon='cutlery', prefix='fa')
    ).add_to(marker_cluster)

m.save("peta_restoran_semarang.html")
print
