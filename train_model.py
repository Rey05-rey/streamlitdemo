import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Konfigurasi Dataset & Fitur
# ==========================================
# Pastikan nama file ini sesuai dengan hasil output program preprocessing-mu
DATASET = "DatasetBRIv2.csv" 
MODEL = "model_BRI_forecasting.pkl"

# 10 Fitur Independen (H-2 dan H-1)
FEATURES = [
    "Open_t-2", "High_t-2", "Low_t-2", "Close_t-2", "Volume_t-2",
    "Open_t-1", "High_t-1", "Low_t-1", "Close_t-1", "Volume_t-1"
]

# Variabel Dependen (Target Prediksi)
TARGET = "Close_t"

splits = {
    "90:10": 0.1,
    "80:20": 0.2,
    "70:30": 0.3,
    "60:40": 0.4,
    "50:50": 0.5,
}

# ==========================================
# 2. Muat Data
# ==========================================
df = pd.read_csv(DATASET)
X = df[FEATURES]
y = df[TARGET]

results = []
models = {}

# ==========================================
# 3. Pelatihan dan Evaluasi Model
# ==========================================
for name, test_size in splits.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Menghitung Metrik
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)

    results.append([name, mae, rmse, mape, r2])
    models[name] = model

# ==========================================
# 4. Tampilkan Hasil
# ==========================================
results_df = pd.DataFrame(
    results,
    columns=["Split", "MAE", "RMSE", "MAPE (%)", "R²"]
)

best_index = results_df["RMSE"].idxmin()
best_split = results_df.loc[best_index, "Split"]
best_model = models[best_split]

print("\nEVALUASI MODEL REGRESI LINEAR BERGANDA")
print(results_df.round(4).to_string(index=False))
print(f"\nSplit terbaik berdasarkan RMSE: {best_split}")

coef_df = pd.DataFrame({
    "Variabel": FEATURES,
    "Koefisien": best_model.coef_
})

print("\nKOEFISIEN MODEL TERBAIK")
print(coef_df.to_string(index=False, float_format=lambda x: f"{x:.12f}"))
print(f"\nIntercept: {best_model.intercept_:.6f}")

# ==========================================
# 5. Simpan Model Terbaik
# ==========================================
with open(MODEL, "wb") as f:
    pickle.dump(best_model, f)

print(f"\n✅ Model terbaik berhasil disimpan sebagai {MODEL}")