import streamlit as st
import joblib
import re
import string
import time
import os

st.set_page_config(
    page_title="Identifikasi Hoax Indonesia",
    layout="wide"
)

# 1. FUNGSI PREPROCESSING
def clean_text(text):
    text = str(text).lower()
    # Hapus URL, Mention, dan Hashtag
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", " ", text) 
    # Hapus Tanda Baca
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Hapus Angka
    text = re.sub(r"\d+", " ", text) 
    # Hapus Spasi Berlebih
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 2. LOAD MODEL & VECTORIZER (PATH SESUAI REVISI LU)
@st.cache_resource
def load_models():
    try:
        # Path file model BERITA
        path_model_berita = "model_berita.pkl"
        path_vec_berita = "vectorizer_berita.pkl"
        
        # Path file model SOSMED
        path_model_sosmed = "model_sosmed.pkl"
        path_vec_sosmed = "vectorizer_sosmed.pkl"

        # Proses Loading
        model_berita = joblib.load(path_model_berita)
        vectorizer_berita = joblib.load(path_vec_berita)

        model_sosmed = joblib.load(path_model_sosmed)
        vectorizer_sosmed = joblib.load(path_vec_sosmed)

        return model_berita, vectorizer_berita, model_sosmed, vectorizer_sosmed
    except Exception as e:
        st.error(f"❌ Gagal memuat model. Pastikan file .pkl sudah ada di folder yang sama!")
        st.error(f"Detail Error: {e}")
        st.stop()

# Inisialisasi Otak Model
mnb_berita, vec_berita, mnb_sosmed, vec_sosmed = load_models()

# 3. CSS CUSTOM (DARK MODE STYLE)
st.markdown("""
<style>
body, .stApp { background-color: #0E1117; color: white; }
textarea { background-color: #ffffff !important; color: #000000 !important; }
.disclaimer { font-size: 12px; color: #ff4b4b; font-style: italic; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# 4. JUDUL & DISCLAIMER UTAMA
st.title("📰 Identifikasi Berita & Sosmed Hoax")
st.warning("""
**PENTING (DISCLAIMER):** Sistem ini dibangun untuk **membantu mengidentifikasi** potensi hoax berdasarkan pola bahasa. 
Hasil prediksi model **bukanlah klaim kebenaran mutlak**.
""")

# 5. FUNGSI TAMPILKAN HASIL
def show_prediction_results(prob, pred, input_text):
    label = "REAL ✅" if pred == 0 else "HOAX ❌"
    color = "green" if pred == 0 else "red"
    
    st.markdown("---")
    st.subheader("📊 Hasil Analisis Model")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### Prediksi: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
    with col2:
        confidence = prob[pred]
        st.write(f"Tingkat Keyakinan Model:")
        st.progress(float(confidence))
        st.write(f"**{confidence*100:.2f}%**")

    # Disclaimer di setiap hasil
    st.markdown(f"""
    <div class='disclaimer'>
    *Catatan: Skor {confidence*100:.2f}% menunjukkan tingkat kemiripan teks dengan pola data yang dipelajari model. 
    Hasil ini tidak dapat dijadikan satu-satunya bukti hukum tanpa verifikasi lebih lanjut.
    </div>
    """, unsafe_allow_html=True)

# 6. TAB NAVIGASI
tab1, tab2 = st.tabs(["📰 Analisis Berita", "📱 Analisis Media Sosial"])

# TAB 1: BERITA
with tab1:
    st.info("Analisis teks artikel atau berita daring dengan narasi yang panjang.")
    teks_berita = st.text_area("Masukkan Teks Berita:", height=200, key="txt_berita")
    
    if st.button("🚀 Deteksi Berita", type="primary"):
        if not teks_berita.strip():
            st.warning("⚠️ Masukkan teks berita terlebih dahulu.")
        else:
            with st.spinner("Menganalisis pola berita..."):
                t_clean = clean_text(teks_berita)
                fitur = vec_berita.transform([t_clean])
                prob = mnb_berita.predict_proba(fitur)[0]
                pred = mnb_berita.predict(fitur)[0]
                show_prediction_results(prob, pred, teks_berita)

# TAB 2: SOSMED
with tab2:
    st.info("Analisis teks pendek seperti caption Instagram, tweet, atau broadcast WhatsApp.")
    teks_sosmed = st.text_area("Masukkan Teks Sosmed:", height=150, key="txt_sosmed")
    
    if st.button("🚀 Deteksi Sosmed", type="primary"):
        if not teks_sosmed.strip():
            st.warning("⚠️ Masukkan teks postingan terlebih dahulu.")
        else:
            with st.spinner("Menganalisis pola bahasa sosmed..."):
                t_clean = clean_text(teks_sosmed)
                fitur = vec_sosmed.transform([t_clean])
                prob = mnb_sosmed.predict_proba(fitur)[0]
                pred = mnb_sosmed.predict(fitur)[0]
                show_prediction_results(prob, pred, teks_sosmed)

# 7. FOOTER
st.markdown("---")
st.caption("© 2026 Muhamad Rizal Rifaldi | Sistem Identifikasi Hoax Berbasis Naive Bayes")