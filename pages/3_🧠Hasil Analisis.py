import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === 1. Load dataset ===
df = pd.read_csv("semarang_resto_dataset.csv")

# === 2. Buat kolom target rating tinggi (rating ≥ 4.5) ===
if 'rating' not in df.columns:
    raise ValueError("Kolom 'rating' tidak ditemukan dalam dataset.")
df['rating_tinggi'] = (df['rating'] >= 4.5).astype(int)

# === 3. Pra-pemrosesan sederhana ===
# Drop kolom non-numerik & yang tidak relevan (kecuali eksplisit disebut)
drop_cols = ['nama_resto', 'alamat', 'rating']  # tambah jika ada
X = df.drop(columns=drop_cols + ['rating_tinggi'], errors='ignore')
y = df['rating_tinggi']

# Hilangkan baris dengan nilai null
X = X.select_dtypes(include=[np.number]).dropna()
y = y.loc[X.index]

# === 4. Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Train model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 6. Save model dan data uji ===
joblib.dump(model, "rating_classifier.pkl")
X_test.to_csv("X_test_rating.csv", index=False)
y_test.to_csv("y_test_rating.csv", index=False)

# === 7. Save hasil evaluasi ===
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=["Tidak Tinggi", "Tinggi"])

with open("results_rating.txt", "w") as f:
    f.write("=== Laporan Evaluasi Model Rating Tinggi ===\n\n")
    f.write(report)

print("✅ Model dan hasil pelatihan berhasil disimpan.")
