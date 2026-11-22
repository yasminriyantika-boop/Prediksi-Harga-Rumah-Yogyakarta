import json
import joblib
import pandas as pd
import streamlit as st

model = joblib.load("model_harga_rumah_xgboost.joblib")
with open("model_meta_xgboost.json", encoding="utf-8") as f:
    meta = json.load(f)

st.title("Prediksi Harga Rumah di Yogyakarta")
st.write("Aplikasi untuk memperkirakan harga rumah berdasarkan lokasi dan spesifikasinya.")

st.sidebar.header("Input Parameter")

lokasi = st.sidebar.selectbox("Lokasi (Kecamatan, Kabupaten/Kota)", meta["daftar_lokasi"])
kamar_tidur = st.sidebar.slider("Jumlah Kamar Tidur", 1, 10, 3)
kamar_mandi = st.sidebar.slider("Jumlah Kamar Mandi", 1, 10, 2)
carport = st.sidebar.slider("Jumlah Carport", 0, 5, 1)
luas_tanah = st.sidebar.slider("Luas Tanah (m2)", 30, 1000, 120)
luas_bangunan = st.sidebar.slider("Luas Bangunan (m2)", 20, 800, 100)

data_input = pd.DataFrame([{
    "kamar_tidur": kamar_tidur,
    "kamar_mandi": kamar_mandi,
    "carport": carport,
    "luas_tanah": luas_tanah,
    "luas_bangunan": luas_bangunan,
    "lokasi": lokasi,
    "kabupaten_kota": lokasi.split(", ")[-1],
}])[meta["fitur"]]

st.subheader("Parameter Input Pengguna:")
st.write(data_input)

kolom_numerik = ["kamar_tidur", "kamar_mandi", "carport", "luas_tanah", "luas_bangunan"]

if st.sidebar.button("Prediksi Harga"):
    try:
        data_prediksi = data_input.astype({kolom: "float64" for kolom in kolom_numerik})
        hasil = model.predict(data_prediksi)[0]
        st.subheader("Hasil Prediksi Harga Rumah:")
        st.write("Harga diperkirakan: Rp " + f"{hasil:,.0f}".replace(",", "."))
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
