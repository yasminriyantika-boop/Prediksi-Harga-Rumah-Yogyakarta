import json
import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = "model_harga_rumah_xgboost.joblib"   # ganti bila nama file modelmu berbeda
META_FILE = "model_meta_xgboost.json"

FITUR = ["kamar_tidur", "kamar_mandi", "carport", "luas_tanah", "luas_bangunan", "lokasi", "kabupaten_kota"]
KOLOM_NUMERIK = ["kamar_tidur", "kamar_mandi", "carport", "luas_tanah", "luas_bangunan"]

# Batas input luas (mencakup sekitar 98% iklan yang dipakai melatih model)
LUAS_TANAH_MIN, LUAS_TANAH_MAX = 20, 1000
LUAS_BANGUNAN_MIN, LUAS_BANGUNAN_MAX = 20, 800

st.set_page_config(page_title="Prediksi Harga Rumah Yogyakarta", layout="wide")


@st.cache_resource
def muat_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def muat_meta():
    try:
        with open(META_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def lokasi_dari_model(model):
    # Cadangan bila model_meta.json tidak ada: ambil daftar lokasi langsung dari model
    encoder = model.regressor_.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
    return list(encoder.categories_[0])


def baca_bilangan_bulat(teks, nama, minimum, maksimum):
    teks = teks.strip()
    if teks == "":
        return None, f"{nama} belum diisi. Silakan isi dengan angka, misalnya 120."
    if not (teks.isascii() and teks.isdigit()):
        return None, f"{nama} hanya boleh diisi angka bulat, tanpa huruf, titik, koma, atau tanda minus. Contoh: 120 (bukan 120,5 atau 1.000)."
    nilai = int(teks)
    if nilai < minimum:
        return None, f"{nama} terlalu kecil ({nilai} m2). Isi dengan angka antara {minimum} sampai {maksimum} m2."
    if nilai > maksimum:
        return None, f"{nama} terlalu besar ({nilai} m2). Isi dengan angka antara {minimum} sampai {maksimum} m2."
    return nilai, None


model = muat_model()
meta = muat_meta()
semua_lokasi = meta.get("daftar_lokasi") or lokasi_dari_model(model)

# Kelompokkan kecamatan per kabupaten/kota, sehingga pilihan kecamatan tidak tercampur antar kabupaten
kecamatan_per_kab = {}
for lokasi in semua_lokasi:
    nama_kecamatan, nama_kab = lokasi.rsplit(", ", 1)
    kecamatan_per_kab.setdefault(nama_kab, set()).add(nama_kecamatan)
kecamatan_per_kab = {kab: sorted(daftar) for kab, daftar in kecamatan_per_kab.items()}

# ---------------------------------------------------------------- panel kiri: lokasi
st.sidebar.header("Lokasi")
daftar_kab = sorted(kecamatan_per_kab)
kabupaten = st.sidebar.selectbox("Kabupaten/Kota", daftar_kab,
                                 index=daftar_kab.index("Sleman") if "Sleman" in daftar_kab else 0)
kecamatan = st.sidebar.selectbox("Kecamatan", kecamatan_per_kab[kabupaten], key="kecamatan_" + kabupaten)

# ---------------------------------------------------------------- bagian tengah: detail rumah
st.title("Prediksi Harga Rumah di Yogyakarta")
st.write("Pilih lokasi di panel kiri, lalu isi detail rumah di bawah ini untuk memperkirakan harganya.")

st.subheader("Detail Properti")

kol1, kol2, kol3 = st.columns(3)
kamar_tidur = kol1.selectbox("Jumlah Kamar Tidur", list(range(1, 11)), index=2)
kamar_mandi = kol2.selectbox("Jumlah Kamar Mandi", list(range(1, 11)), index=1)
carport = kol3.selectbox("Jumlah Carport (0 = tidak ada)", list(range(0, 11)), index=1)

kol4, kol5 = st.columns(2)

teks_tanah = kol4.text_input(
    "Luas Tanah (m2)", value="120",
    help=f"Isi dengan angka bulat (Contoh: 120), antara {LUAS_TANAH_MIN} sampai {LUAS_TANAH_MAX} m2")
luas_tanah, error_tanah = baca_bilangan_bulat(teks_tanah, "Luas tanah", LUAS_TANAH_MIN, LUAS_TANAH_MAX)
if error_tanah:
    kol4.error(error_tanah)

teks_bangunan = kol5.text_input(
    "Luas Bangunan (m2)", value="100",
    help=f"Isi dengan angka bulat (Contoh: 120), antara {LUAS_BANGUNAN_MIN} sampai {LUAS_BANGUNAN_MAX} m2")
luas_bangunan, error_bangunan = baca_bilangan_bulat(teks_bangunan, "Luas bangunan", LUAS_BANGUNAN_MIN, LUAS_BANGUNAN_MAX)
if error_bangunan:
    kol5.error(error_bangunan)

daftar_error = [e for e in (error_tanah, error_bangunan) if e]

st.write("")
tekan_prediksi = st.button("Prediksi Harga", disabled=bool(daftar_error))

if daftar_error:
    st.warning("Ada isian yang belum sesuai. Silakan baca pesan di bawah kotak isian, perbaiki, lalu coba lagi.")

# ---------------------------------------------------------------- hasil (tetap di tengah)
if tekan_prediksi and not daftar_error:
    data_input = pd.DataFrame([{
        "kamar_tidur": kamar_tidur,
        "kamar_mandi": kamar_mandi,
        "carport": carport,
        "luas_tanah": luas_tanah,
        "luas_bangunan": luas_bangunan,
        "lokasi": kecamatan + ", " + kabupaten,
        "kabupaten_kota": kabupaten,
    }])[FITUR]

    try:
        # Tipe data numerik disamakan (float) dengan data saat model dilatih
        data_prediksi = data_input.astype({kolom: "float64" for kolom in KOLOM_NUMERIK})
        hasil = model.predict(data_prediksi)[0]

        st.subheader("Hasil Prediksi Harga Rumah")
        st.metric("Perkiraan harga", "Rp " + f"{hasil:,.0f}".replace(",", "."))
        st.caption(f"Lokasi: {kecamatan}, {kabupaten}. Ini hanya perkiraan berdasarkan data iklan rumah; harga sebenarnya bisa berbeda.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
