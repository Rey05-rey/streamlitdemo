import pickle
import streamlit as st
from config import MODEL_PATH

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

