import streamlit as st
import joblib
from pathlib import Path

# =========================
# Konfigurasi halaman
# =========================
st.set_page_config(
    page_title="Deteksi Spam SMS",
    page_icon="📩",
    layout="wide"
)

# =========================
# Lokasi model
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# =========================
# Styling sederhana
# =========================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

.judul-box {
    background: linear-gradient(135deg, #f8fbff, #eef5ff);
    padding: 28px;
    border-radius: 18px;
    border: 1px solid #d9e7ff;
    margin-bottom: 20px;
}

.judul-besar {
    font-size: 38px;
    font-weight: 700;
    color: #1f2d3d;
    margin-bottom: 8px;
}

.subjudul {
    font-size: 17px;
    color: #51606f;
    margin-bottom: 0;
}

.info-box {
    background-color: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 18px;
    margin-top: 16px;
    margin-bottom: 16px;
}

.hasil-box {
    padding: 18px;
    border-radius: 14px;
    margin-top: 10px;
    border: 1px solid #e5e7eb;
}

.hasil-aman {
    background-color: #ecfdf5;
    border: 1px solid #a7f3d0;
}

.hasil-spam {
    background-color: #fef2f2;
    border: 1px solid #fecaca;
}

.label-kecil {
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 4px;
}

.nilai-besar {
    font-size: 28px;
    font-weight: 700;
    color: #111827;
}

section[data-testid="stSidebar"] {
    background-color: #f8fafc;
}

div.stButton > button {
    background-color: #1d4ed8;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.7rem 1rem;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #1e40af;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Sidebar
# =========================
st.sidebar.title("Tentang Proyek")
st.sidebar.write(
    "Aplikasi ini digunakan untuk mengklasifikasikan apakah sebuah pesan "
    "termasuk **spam** atau **bukan spam** berdasarkan model machine learning."
)

st.sidebar.markdown("### Informasi Model")
st.sidebar.write("**Algoritma:** Naive Bayes")
st.sidebar.write("**Ekstraksi Fitur:** TF-IDF")
st.sidebar.write("**Dataset:** SMS Spam Collection")
st.sidebar.write("**Akurasi Uji:** 97,04%")

st.sidebar.markdown("### Catatan")
st.sidebar.info(
    "Model dilatih menggunakan dataset SMS berbahasa Inggris. "
    "Untuk pesan berbahasa Indonesia, hasil prediksi bisa tetap berjalan, "
    "tetapi akurasinya belum tentu sebaik data latih asli."
)

# =========================
# Header utama
# =========================
st.markdown("""
<div class="judul-box">
    <div class="judul-besar">Deteksi Pesan Spam SMS</div>
    <p class="subjudul">
        Aplikasi sederhana untuk mengidentifikasi apakah suatu pesan tergolong spam atau bukan.
        Proyek ini dibuat menggunakan Python, Streamlit, dan model klasifikasi teks.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Analisis Pesan")
    st.write(
        "Masukkan isi pesan pada kolom berikut, lalu klik tombol analisis "
        "untuk melihat hasil klasifikasinya."
    )

    pesan = st.text_area(
        "Isi pesan",
        placeholder="Contoh: Selamat! Anda memenangkan hadiah. Klik tautan berikut untuk mengklaim hadiah Anda.",
        height=180
    )

    analisis = st.button("Analisis Sekarang", use_container_width=True)

with col2:
    st.markdown("""
    <div class="info-box">
        <b>Contoh pesan spam</b><br><br>
        Congratulations! You have won a FREE prize. Click here now!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <b>Contoh pesan normal</b><br><br>
        Hey, are we still meeting tomorrow at 10 AM?
    </div>
    """, unsafe_allow_html=True)

# =========================
# Proses prediksi
# =========================
if analisis:
    if not pesan.strip():
        st.warning("Silakan masukkan pesan terlebih dahulu.")
    else:
        prediksi = model.predict([pesan])[0]
        probabilitas = model.predict_proba([pesan])[0]
        spam_index = list(model.classes_).index("spam")
        skor_spam = float(probabilitas[spam_index])
        skor_aman = 1 - skor_spam

        st.markdown("---")
        st.subheader("Hasil Analisis")

        if prediksi == "spam":
            st.markdown(f"""
            <div class="hasil-box hasil-spam">
                <div class="label-kecil">Kategori pesan</div>
                <div class="nilai-besar">Terdeteksi sebagai SPAM</div>
                <p>Pesan ini memiliki karakteristik yang mirip dengan pesan spam/promosi mencurigakan.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="hasil-box hasil-aman">
                <div class="label-kecil">Kategori pesan</div>
                <div class="nilai-besar">Terdeteksi sebagai BUKAN SPAM</div>
                <p>Pesan ini lebih dekat dengan pola pesan normal.</p>
            </div>
            """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Skor Spam", f"{skor_spam * 100:.2f}%")
        with c2:
            st.metric("Skor Bukan Spam", f"{skor_aman * 100:.2f}%")

        st.write("Tingkat indikasi spam:")
        st.progress(skor_spam)

        st.caption(
            "Catatan: skor di atas merupakan estimasi model dan tidak dapat dijadikan "
            "satu-satunya dasar pengambilan keputusan."
        )

st.markdown("---")
st.markdown(
    """
    <div style="color:#6b7280; font-size:14px;">
    Dibuat untuk portfolio pembelajaran machine learning dan deployment Streamlit.
    </div>
    """,
    unsafe_allow_html=True
)