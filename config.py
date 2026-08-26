import pandas as pd
from pathlib import Path

# ==========================================
# KONFIGURASI NAMA FILE & FITUR
# ==========================================
MODEL_PATH = Path("ModelBRI.pkl") 

FEATURES = [
    "Open_t-2", "High_t-2", "Low_t-2", "Close_t-2", "Volume_t-2",
    "Open_t-1", "High_t-1", "Low_t-1", "Close_t-1", "Volume_t-1"
]

# Data Hasil Evaluasi 5 Skenario
EVAL_DATA = pd.DataFrame({
    "Split": ["90:10", "80:20", "70:30", "60:40", "50:50"],
    "MAE": [67.7801, 77.6651, 87.3646, 81.9558, 77.0202],
    "RMSE": [89.7726, 103.4310, 116.4463, 109.4345, 102.6369],
    "MAPE (%)": [1.7452, 1.9558, 2.0237, 1.8190, 1.6820],
    "R²": [0.7419, 0.8905, 0.9663, 0.9790, 0.9792]
})