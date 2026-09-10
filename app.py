"""
Apple vs Orange Classifier — Streamlit Deployment App (Custom CNN)
Author  : Nada Thahira Sosa (2601)
Project : LAS26 Case 1 — Week 2/3/4 Big Data / Machine Learning
Model   : Custom CNN (from scratch)

Run locally:
    streamlit run app.py

Required file in the same folder:
    - custom_cnn_model.h5
"""

import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Apple vs Orange Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

IMG_SIZE = 128
MODEL_PATH = "custom_cnn_model.h5"
MODEL_INFO = {
    "accuracy": 0.9313,
    "f1": 0.9308,
    "params": "157,473",
    "trainable": "156,769",
    "epochs": 47,
}
CLASS_EMOJI = {"Apple": "🍎", "Orange": "🍊"}
CLASS_COLOR = {"Apple": "#EF4444", "Orange": "#F97316"}

# --------------------------------------------------------------------------------------
# STYLE
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
    html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }

    .hero {
        background: linear-gradient(120deg, #EF4444 0%, #F97316 55%, #FACC15 100%);
        padding: 2.2rem 2.4rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 12px 30px rgba(239, 68, 68, 0.25);
    }
    .hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; }
    .hero p  { margin: 0.4rem 0 0 0; font-size: 1rem; opacity: 0.95; }

    .card {
        background: white;
        border-radius: 18px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }

    .result-badge { font-size: 2.6rem; font-weight: 800; margin: 0.2rem 0; }

    .confidence-track {
        width: 100%;
        height: 14px;
        border-radius: 999px;
        background: #F1F5F9;
        overflow: hidden;
        margin-top: 0.4rem;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.6s ease;
    }

    .footer-note {
        text-align: center;
        color: #94A3B8;
        font-size: 0.82rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E2E8F0;
    }

    section[data-testid="stSidebar"] { background: #0F172A; }
    section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# MODEL LOADING (cached — avoids re-loading weights on every rerun)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not Path(path).exists():
        return None
    return tf.keras.models.load_model(path, compile=False)


@st.cache_data(show_spinner=False)
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Resize to 128x128 and scale to [0,1], matching the notebook's preprocessing."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# --------------------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------------------
threshold = 0.50

# --------------------------------------------------------------------------------------
# HERO HEADER
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🍎 Apple vs Orange Classifier 🍊</h1>
        <p>Unggah gambar apel atau jeruk untuk memperoleh hasil klasifikasi dari model Custom CNN beserta tingkat keyakinannya.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_about = st.tabs(["🔍 Prediksi", "📄 Tentang Aplikasi"])

# --------------------------------------------------------------------------------------
# TAB 1 — PREDICT
# --------------------------------------------------------------------------------------
with tab_predict:
    col_upload, col_result = st.columns([1, 1.2], gap="large")

    with col_upload:
        st.markdown("#### 1. Unggah gambar")
        uploaded = st.file_uploader(
            "Format didukung: JPG, JPEG, PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            st.image(uploaded, caption="Preview gambar", use_container_width=True)
        else:
            st.info("Unggah gambar apel atau jeruk untuk memulai prediksi.")

    with col_result:
        st.markdown("#### 2. Hasil Prediksi")
        model = load_model(MODEL_PATH)

        if model is None:
            with st.container(border=True):
                st.warning(
                    f"File model `{MODEL_PATH}` tidak ditemukan di folder deploy. "
                    "Pastikan file .h5 hasil training berada di root repo, sejajar dengan app.py, "
                    "dengan nama persis `custom_cnn_model.h5`."
                )
        elif uploaded is not None:
            batch = preprocess_image(uploaded.getvalue())

            t0 = time.time()
            prob_orange = float(model.predict(batch, verbose=0)[0][0])
            latency = (time.time() - t0) * 1000
            label = "Orange" if prob_orange > threshold else "Apple"
            confidence = prob_orange if label == "Orange" else 1 - prob_orange
            color = CLASS_COLOR[label]

            with st.container(border=True):
                st.markdown(
                    f"<div class='result-badge' style='color:{color}'>"
                    f"{CLASS_EMOJI[label]} {label}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='confidence-track'>"
                    f"<div class='confidence-fill' style='width:{confidence*100:.1f}%; background:{color};'></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Keyakinan model: **{confidence*100:.2f}%**  ·  waktu inferensi ≈ {latency:.0f} ms")
        else:
            with st.container(border=True):
                st.write("Belum ada gambar yang diunggah. Hasil prediksi akan muncul di sini.")

# --------------------------------------------------------------------------------------
# TAB 2 — ABOUT
# --------------------------------------------------------------------------------------
with tab_about:
    st.markdown("#### Tentang Aplikasi")
    st.write(
        "**Apple vs Orange Classifier** adalah aplikasi klasifikasi citra biner yang membedakan "
        "foto apel dan jeruk secara otomatis menggunakan model *Convolutional Neural Network* "
        "(CNN) yang dibangun dan dilatih dari nol (bukan model pre-trained). Aplikasi ini "
        "dikembangkan sebagai bagian dari proyek pembelajaran *Deep Learning* untuk klasifikasi "
        "citra, mencakup tahap eksplorasi data, arsitektur model, pelatihan, evaluasi, hingga "
        "deployment ke lingkungan produksi berbasis web."
    )

    st.markdown("#### Cara Kerja")
    st.write(
        "Saat pengguna mengunggah sebuah gambar, sistem akan mengubah ukurannya menjadi "
        f"{IMG_SIZE}×{IMG_SIZE} piksel dan menormalisasi nilai pikselnya ke rentang 0–1, "
        "mengikuti tahap *preprocessing* yang sama seperti pada proses pelatihan model. "
        "Gambar yang telah diproses kemudian dilewatkan ke model CNN, yang menghasilkan "
        "sebuah nilai probabilitas. Nilai ini dibandingkan dengan ambang keputusan (*decision "
        "threshold*) untuk menentukan apakah gambar tersebut diklasifikasikan sebagai Apple "
        "atau Orange, beserta tingkat keyakinan modelnya."
    )

    st.markdown("#### Arsitektur & Data")
    st.write(
        "Model dibangun menggunakan beberapa lapisan konvolusi (*Convolutional Layer*) yang "
        "dilatih untuk mengenali pola visual seperti warna, tekstur, dan bentuk khas masing-"
        "masing buah. Model dilatih menggunakan dataset berisi 796 citra apel dan jeruk "
        "berukuran 128×128 piksel, dengan pembagian data latih, validasi, dan uji untuk "
        "memastikan performa model dapat diukur secara objektif."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{MODEL_INFO['accuracy']*100:.2f}%")
    m2.metric("F1 Score", f"{MODEL_INFO['f1']:.4f}")
    m3.metric("Total Params", MODEL_INFO["params"])
    m4.metric("Epochs", MODEL_INFO["epochs"])

    st.markdown("#### Batasan Aplikasi")
    st.write(
        "Model dilatih secara khusus untuk membedakan dua kelas buah, yaitu apel dan jeruk. "
        "Akurasi prediksi dapat menurun apabila gambar memiliki pencahayaan yang kurang "
        "memadai, latar belakang yang ramai, objek yang tidak fokus, atau menampilkan buah "
        "selain kedua kelas tersebut. Aplikasi ini dikembangkan untuk tujuan pembelajaran dan "
        "demonstrasi, bukan untuk penggunaan komersial."
    )

    st.markdown("#### Informasi Proyek")
    st.write(
        "Aplikasi ini dikembangkan sebagai bagian dari proyek Big Data / Machine Learning, "
        "menggunakan TensorFlow/Keras untuk pemodelan dan Streamlit untuk antarmuka serta "
        "deployment aplikasi."
    )

st.markdown(
    "<div class='footer-note'>Dibangun dengan Streamlit · TensorFlow/Keras · "
    "Apple vs Orange Binary Classification Project</div>",
    unsafe_allow_html=True,
)
