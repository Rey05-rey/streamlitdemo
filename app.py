from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================
# 1. KONFIGURASI NAMA FILE & FITUR
# ==========================================
MODEL_PATH = Path("ModelBRI.pkl") 

FEATURES = [
    "Open_t-2", "High_t-2", "Low_t-2", "Close_t-2", "Volume_t-2",
    "Open_t-1", "High_t-1", "Low_t-1", "Close_t-1", "Volume_t-1"
]

# Data Hasil Evaluasi 5 Skenario (tetap float agar grafik Plotly tidak error)
EVAL_DATA = pd.DataFrame({
    "Split": ["90:10", "80:20", "70:30", "60:40", "50:50"],
    "MAE": [67.7801, 77.6651, 87.3646, 81.9558, 77.0202],
    "RMSE": [89.7726, 103.4310, 116.4463, 109.4345, 102.6369],
    "MAPE (%)": [1.7452, 1.9558, 2.0237, 1.8190, 1.6820],
    "R²": [0.7419, 0.8905, 0.9663, 0.9790, 0.9792]
})

# ==========================================
# 2. FUNGSI UTILITY & LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    """Memuat model Regresi Linear Berganda yang telah dilatih."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH.name} tidak ditemukan. Pastikan file berada di folder yang sama.")
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)

def validate_ohlc(o, h, l, c, day_label):
    """Validasi sederhana konsistensi input harga pada suatu hari."""
    errors = []
    if h < o: errors.append(f"[{day_label}] Nilai High tidak boleh lebih kecil dari Open.")
    if l > o: errors.append(f"[{day_label}] Nilai Low tidak boleh lebih besar dari Open.")
    if h < l: errors.append(f"[{day_label}] Nilai High tidak boleh lebih kecil dari Low.")
    if c > h or c < l: errors.append(f"[{day_label}] Nilai Close harus berada di antara rentang High dan Low.")
    return errors 

def format_rupiah(val: float) -> str:
    """Format angka ke format Rupiah standar Indonesia (titik ribuan, koma desimal)."""
    return f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def format_koma(val: float, decimals: int = 4) -> str:
    """Format desimal float menjadi string dengan pemisah koma."""
    return f"{val:.{decimals}f}".replace(".", ",")

# ==========================================
# 3. KOMPONEN TAMPILAN (UI)
# ==========================================
def show_manual_prediction(inputs, model):
    """Menampilkan hasil prediksi manual dan Visualisasi Evaluasi Model."""
    features = pd.DataFrame([inputs], columns=FEATURES)
    prediction = float(model.predict(features)[0])
    close_t1 = inputs[8] 

    st.markdown("---")
    st.subheader("Hasil Prediksi")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Prediksi Harga Penutupan", 
            value=f"Rp {format_rupiah(prediction)}"
        )

    difference = prediction - close_t1
    difference_pct = (difference / close_t1 * 100 if close_t1 != 0 else 0.0)
    with col2:
        st.metric(
            label="Proyeksi Selisih thd Close H-1", 
            value=f"Rp {format_rupiah(difference)}", 
            delta=f"{format_koma(difference_pct, 2)}%"
        )
    
    st.markdown("---")
    st.subheader("Visualisasi Hasil Evaluasi Model (5 Skenario Pembagian Data)")
    st.write("Tabel dan grafik di bawah ini menunjukkan perbandingan performa algoritma Regresi Linear Berganda pada seluruh skenario rasio pembagian data (Latih : Uji) yang menjadi dasar model ini.")
    
    # Format tabel evaluasi agar menggunakan pemisah koma
    st.dataframe(
        EVAL_DATA.style.format({
            "MAE": lambda x: format_koma(x, 4),
            "RMSE": lambda x: format_koma(x, 4),
            "MAPE (%)": lambda x: format_koma(x, 4),
            "R²": lambda x: format_koma(x, 4),
        }),
        use_container_width=True, 
        hide_index=True
    )
    
    col_fig1, col_fig2 = st.columns(2)
    
    with col_fig1:
        fig1 = px.line(
            EVAL_DATA, x="Split", y=["MAE", "RMSE"], markers=True, 
            title="Perbandingan Error: MAE vs RMSE",
            labels={"value": "Nilai Error", "variable": "Metrik"}
        )
        fig1.update_traces(textposition="top center")
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_fig2:
        fig2 = px.line(
            EVAL_DATA, x="Split", y=["MAPE (%)", "R²"], markers=True, 
            title="Perbandingan Akurasi: MAPE vs R²",
            labels={"value": "Nilai Evaluasi", "variable": "Metrik"}
        )
        fig2.update_traces(textposition="top center")
        st.plotly_chart(fig2, use_container_width=True)


def main():
    st.set_page_config(page_title="Prediksi Saham BRI", page_icon="📈", layout="wide")
    st.title("📈 Dashboard Prediksi Harga Saham BRI")
    st.caption("Implementasi Regresi Linear Berganda untuk memprediksi harga penutupan berdasarkan 10 fitur historis (H-2 dan H-1).")

    try:
        model = load_model()
    except Exception as error:
        st.error(str(error))
        st.stop()

    tab_manual, tab_info = st.tabs([
        "Prediksi", "Informasi Penelitian"
    ])

    with tab_manual:
        st.subheader("Input Data Historis Saham")
        
        col_t2, col_t1 = st.columns(2)
        
        with col_t2:
            st.markdown("#### Hari Pertama (H-2)")
            o_t2 = st.number_input("Open (H-2)", min_value=0.0, value=4320, step=5.0)
            h_t2 = st.number_input("High (H-2)", min_value=0.0, value=4370, step=5.0)
            l_t2 = st.number_input("Low (H-2)", min_value=0.0, value=4280, step=5.0)
            c_t2 = st.number_input("Close (H-2)", min_value=0.0, value=4370, step=5.0)
            v_t2 = st.number_input("Volume (H-2)", min_value=0.0, value=180030000, step=1000.0)
            
        with col_t1:
            st.markdown("#### Hari Kedua (H-1)")
            o_t1 = st.number_input("Open (H-1)", min_value=0.0, value=4360, step=5.0)
            h_t1 = st.number_input("High (H-1)", min_value=0.0, value=4450, step=5.0)
            l_t1 = st.number_input("Low (H-1)", min_value=0.0, value=4320, step=5.0)
            c_t1 = st.number_input("Close (H-1)", min_value=0.0, value=4450, step=5.0)
            v_t1 = st.number_input("Volume (H-1)", min_value=0.0, value=466130000, step=1000.0)

        if st.button("Prediksi Harga Penutupan", type="primary", use_container_width=True):
            inputs = [o_t2, h_t2, l_t2, c_t2, v_t2, o_t1, h_t1, l_t1, c_t1, v_t1]
            
            if any(val is None for val in inputs):
                st.error("⚠️ Harap isi semua parameter masukan terlebih dahulu sebelum melakukan prediksi!")
            else:
                validation_errors = validate_ohlc(o_t2, h_t2, l_t2, c_t2, "Hari H-2") + validate_ohlc(o_t1, h_t1, l_t1, c_t1, "Hari H-1")
                if validation_errors:
                    for msg in validation_errors: 
                        st.error(msg)
                else:
                    show_manual_prediction(inputs, model)

    with tab_info:
        st.subheader("Informasi Penelitian")
        st.write("Sistem prediksi ini dibangun untuk mengevaluasi kemampuan algoritma Regresi Linear Berganda dalam memprediksi harga saham dengan menggunakan 10 variabel historis (lag 2 hari).")

    st.divider()
    st.caption("Reynaldi Syachputra Dewanata | Prediksi Harga Saham BRI Menggunakan Regresi Linear")

if __name__ == "__main__":
    main()