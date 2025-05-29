import streamlit as st

st.set_page_config(page_title="Background")
st.title("Background")
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Restaurant Dashboard</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css" />
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            background-color: #2c3e50;
            color: white;
        }
        .header {
            background-color: #34495e;
            padding: 20px;
        }
        .header h2 {
            margin: 0;
            font-size: 24px;
        }
        .section {
            margin: 20px;
            padding: 20px;
            background-color: #3b4a60;
            border-radius: 8px;
        }
        #map {
            height: 400px;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h2>🍽️ Selamat Datang di Dashboard Analisis Restoran</h2>
    </div>

    <div class="section">
        <h3>Background</h3>
        <p>
            Restaurant business is booming in Indonesia and in other countries. It seems that COVID-19 does not really shut down this business although the number of visitors decreased when lockdown is put in effect. This decrease in the number of visiting customers is a problem but there are restaurants that worked around this problem by using grab and gojek to deliver their food to the customers in their homes. According to a research by Arlindo Madeira et al. although restaurant owners were pessimistic that their restaurant will return to what it was before the pandemic, they have tough up strategies to run their business after the COVID-19 passes over. This means that the restaurant business is here to stay, with some changes.
        </p>
        <p>
            Cinere is a suburb with many restaurants. If someone wants to open a new restaurant in Cinere, she should study the existing restaurants there. In this case, clustering will be very helpful in profiling the restaurants in Cinere. From that profile, she can, for example, decide to open a restaurant in a certain category which has a high rating.
        </p>
    </div>

    <div class="section">
        <h3>Restaurants in Cinere</h3>
        <div id="map"></div>
    </div>

    <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
    <script>
        var map = L.map('map').setView([-6.3428, 106.8166], 13); // Centered around Cinere

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);

        // Example marker (you can customize or load from a dataset)
        var marker = L.marker([-6.3428, 106.8166]).addTo(map);
        marker.bindPopup("Example Restaurant Location").openPopup();
    </script>
</body>
</html>
