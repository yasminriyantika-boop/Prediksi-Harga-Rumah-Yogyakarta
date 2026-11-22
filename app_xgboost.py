import json
import joblib
import pandas as pd
import streamlit as st

model = joblib.load("model_harga_rumah_xgboost.joblib")
with open("model_meta_xgboost.json", encoding="utf-8") as f:
    meta = json.load(f)

st.title("Prediksi Harga Rumah di Yogyakarta")
st.write("Aplikasi untuk memperkirakan harga rumah berdasarkan lokasi dan spesifikasinya.")

LUAS_TANAH_MIN, LUAS_TANAH_MAX = 20, 1000
LUAS_BANGUNAN_MIN, LUAS_BANGUNAN_MAX = 20, 800

def baca_bilangan_bulat(teks, nama, minimum, maksimum):
    teks = teks.strip()
    if teks == "":
        return None, f"{nama} wajib diisi."
    if not (teks.isascii() and teks.isdigit()):
        return None, f"{nama} harus berupa bilangan bulat positif (angka saja, tanpa titik, koma, atau huruf)."
    nilai = int(teks)
    if nilai < minimum or nilai > maksimum:
        return None, f"{nama} harus antara {minimum} dan {maksimum} m2 (kamu mengisi {nilai})."
    return nilai, None

st.sidebar.header("Input Parameter")

lokasi = st.sidebar.selectbox("Lokasi (Kecamatan, Kabupaten/Kota)", meta["daftar_lokasi"])
kamar_tidur = st.sidebar.slider("Jumlah Kamar Tidur", 1, 10, 3)
kamar_mandi = st.sidebar.slider("Jumlah Kamar Mandi", 1, 10, 2)
carport = st.sidebar.slider("Jumlah Carport", 0, 5, 1)
teks_tanah = st.sidebar.text_input(
    "Luas Tanah (m2)", value="120",
    help=f"Bilangan bulat, {LUAS_TANAH_MIN} sampai {LUAS_TANAH_MAX} m2")
luas_tanah, error_tanah = baca_bilangan_bulat(teks_tanah, "Luas tanah", LUAS_TANAH_MIN, LUAS_TANAH_MAX)
if error_tanah:
    st.sidebar.error(error_tanah)

teks_bangunan = st.sidebar.text_input(
    "Luas Bangunan (m2)", value="100",
    help=f"Bilangan bulat, {LUAS_BANGUNAN_MIN} sampai {LUAS_BANGUNAN_MAX} m2")
luas_bangunan, error_bangunan = baca_bilangan_bulat(teks_bangunan, "Luas bangunan", LUAS_BANGUNAN_MIN, LUAS_BANGUNAN_MAX)
if error_bangunan:
    st.sidebar.error(error_bangunan)

daftar_error = [e for e in (error_tanah, error_bangunan) if e]
if daftar_error:
    st.warning("Input belum valid. Perbaiki isian luas di panel kiri, lalu coba lagi.")
    st.stop()

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
