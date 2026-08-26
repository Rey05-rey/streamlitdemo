import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Impor variabel dan fungsi dari file terpisah
from config import FEATURES, EVAL_DATA
from utils import load_model, validate_ohlc


def show_manual_prediction(inputs, model):
    """Menampilkan hasil prediksi manual dan Visualisasi Evaluasi Model."""
    features = pd.DataFrame([inputs], columns=FEATURES)
    prediction = float(model.predict(features)[0])
    close_t1 = inputs[8] 

    st.markdown("---")
    st.subheader("Hasil Prediksi")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Prediksi Harga Penutupan", value=f"Rp {prediction:,.2f}")

    difference = prediction - close_t1
    difference_pct = (difference / close_t1 * 100 if close_t1 != 0 else 0.0)
    with col2:
        st.metric(label="Proyeksi Selisih thd Close H-1", value=f"Rp {difference:,.2f}", delta=f"{difference_pct:.2f}%")
    
    st.markdown("---")
    st.subheader("Visualisasi Hasil Evaluasi Model (5 Skenario Pembagian Data)")
    st.write("Tabel dan grafik di bawah ini menunjukkan perbandingan performa algoritma Regresi Linear Berganda pada seluruh skenario rasio pembagian data (Latih : Uji) yang menjadi dasar model ini.")
    
    st.dataframe(EVAL_DATA.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE (%)": "{:.4f}", "R²": "{:.4f}"}), use_container_width=True, hide_index=True)
    
    col_fig1, col_fig2 = st.columns(2)
    
    with col_fig1:
        fig1 = px.line(EVAL_DATA, x="Split", y=["MAE", "RMSE"], markers=True, 
                       title="Perbandingan Error: MAE vs RMSE",
                       labels={"value": "Nilai Error", "variable": "Metrik"})
        fig1.update_traces(textposition="top center")
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_fig2:
        fig2 = px.line(EVAL_DATA, x="Split", y=["MAPE (%)", "R²"], markers=True, 
                       title="Perbandingan Akurasi: MAPE vs R²",
                       labels={"value": "Nilai Evaluasi", "variable": "Metrik"})
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

    # Hanya menyisakan 2 tab: Prediksi dan Informasi Penelitian
    tab_manual, tab_info = st.tabs([
        "Prediksi", "Informasi Penelitian"
    ])

    with tab_manual:
        st.subheader("Input Data Historis Saham")
        
        col_t2, col_t1 = st.columns(2)
        
        with col_t2:
            st.markdown("#### Hari Pertama (H-2)")
            # Mengubah value=None menjadi angka riil H-2
            o_t2 = st.number_input("Open (H-2)", min_value=0.0, value=4320.0, step=5.0)
            h_t2 = st.number_input("High (H-2)", min_value=0.0, value=4370.0, step=5.0)
            l_t2 = st.number_input("Low (H-2)", min_value=0.0, value=4280.0, step=5.0)
            c_t2 = st.number_input("Close (H-2)", min_value=0.0, value=4370.0, step=5.0)
            v_t2 = st.number_input("Volume (H-2)", min_value=0.0, value=180030000.0, step=1000.0)
            
        with col_t1:
            st.markdown("#### Hari Kedua (H-1)")
            # Mengubah value=None menjadi angka riil H-1
            o_t1 = st.number_input("Open (H-1)", min_value=0.0, value=4360.0, step=5.0)
            h_t1 = st.number_input("High (H-1)", min_value=0.0, value=4450.0, step=5.0)
            l_t1 = st.number_input("Low (H-1)", min_value=0.0, value=4320.0, step=5.0)
            c_t1 = st.number_input("Close (H-1)", min_value=0.0, value=4450.0, step=5.0)
            v_t1 = st.number_input("Volume (H-1)", min_value=0.0, value=466130000.0, step=1000.0)

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