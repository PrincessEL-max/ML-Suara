"""
Dashboard klasifikasi suara anjing vs kucing (+ penolak suara lain).

Jalankan dari folder yang sama dengan klasifikasi_suara.py:
    pip install streamlit
    python klasifikasi_suara.py        # sekali, supaya model & hasil evaluasi ada
    streamlit run dashboard.py
"""
import tempfile
from pathlib import Path

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import klasifikasi_suara as ks

st.set_page_config(page_title="Klasifikasi Suara", page_icon="🔊", layout="wide")
st.title("🔊 Klasifikasi Suara: Anjing vs Kucing")


@st.cache_resource
def muat_model():
    if not ks.MODEL_KLASIFIKASI.exists():
        return None, None
    klas = joblib.load(ks.MODEL_KLASIFIKASI)
    penolak = joblib.load(ks.MODEL_PENOLAK) if ks.MODEL_PENOLAK.exists() else None
    return klas, penolak


def baca_csv(nama):
    f = ks.OUT_DIR / nama
    return pd.read_csv(f, index_col=0) if f.exists() else None


def tampil_gambar(kolom, nama, judul):
    f = ks.OUT_DIR / nama
    if f.exists():
        kolom.caption(judul)
        kolom.image(str(f))


tab_tes, tab_eval = st.tabs(["🎙️ Tes Suara", "📊 Evaluasi Model"])

# ------------------------- TAB 1: TES SUARA -------------------------
with tab_tes:
    klas, penolak = muat_model()
    if klas is None:
        st.warning("Model belum ada. Jalankan `python klasifikasi_suara.py` dulu.")
    else:
        ambang = st.slider("Ambang penolak (peluang anjing/kucing minimal agar lolos)",
                           0.1, 0.9, ks.AMBANG_PENOLAK, 0.05,
                           help="Lebih tinggi = lebih banyak suara asing ditolak, "
                                "tapi suara anjing/kucing asli juga lebih sering ikut ditolak.")
        kiri, kanan = st.columns(2)
        berkas = kiri.file_uploader("Upload rekaman", type=["wav", "mp3", "ogg", "flac"])
        if hasattr(st, "audio_input"):
            berkas = kanan.audio_input("...atau rekam langsung") or berkas

        if berkas is not None:
            with tempfile.NamedTemporaryFile(suffix=Path(berkas.name).suffix or ".wav", delete=False) as tmp:
                tmp.write(berkas.getvalue())
            st.audio(berkas.getvalue())

            fitur = ks.ekstrak_fitur(tmp.name).reshape(1, -1)
            p_target = penolak["model"].predict_proba(fitur)[0][1] if penolak else 1.0
            prob = klas["model"].predict_proba(fitur)[0]
            i = int(prob.argmax())

            if p_target < ambang:
                st.error(f"### BUKAN anjing/kucing\nPeluang anjing/kucing: {p_target * 100:.1f}%")
            elif prob[i] < ks.AMBANG_YAKIN:
                st.warning(f"### {list(ks.KELAS)[i].upper()} (tidak yakin)\nKeyakinan: {prob[i] * 100:.1f}%")
            else:
                st.success(f"### {list(ks.KELAS)[i].upper()}\nKeyakinan: {prob[i] * 100:.1f}%")

            m1, m2, m3 = st.columns(3)
            m1.metric("Peluang anjing/kucing (penolak)", f"{p_target * 100:.1f}%")
            m2.metric("Peluang anjing", f"{prob[0] * 100:.1f}%")
            m3.metric("Peluang kucing", f"{prob[1] * 100:.1f}%")

            y, sr = librosa.load(tmp.name, sr=ks.SR, duration=ks.DURASI)
            fig, ax = plt.subplots(1, 2, figsize=(11, 3))
            librosa.display.waveshow(y, sr=sr, ax=ax[0])
            ax[0].set_title("Waveform")
            S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max)
            librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="mel", ax=ax[1])
            ax[1].set_title("Mel-spectrogram")
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("Catatan: model hanya dilatih dari klip ESC-50. Rekaman ber-noise tebal "
                       "bisa salah tolak, dan suara hewan/manusia lain kadang masih lolos.")

# ------------------------- TAB 2: EVALUASI -------------------------
with tab_eval:
    t_klas, t_pen = baca_csv("perbandingan_model.csv"), baca_csv("perbandingan_penolak.csv")
    asing = baca_csv("penolak_kategori_asing.csv")
    if t_klas is None:
        st.warning("Belum ada hasil evaluasi. Jalankan `python klasifikasi_suara.py` dulu.")
    else:
        st.subheader("Tahap 2: anjing vs kucing")
        terbaik = t_klas["CV Accuracy"].idxmax()
        st.metric(f"CV Accuracy terbaik ({terbaik})", f"{t_klas['CV Accuracy'].max() * 100:.1f}%")
        st.dataframe(t_klas)
        st.caption("Data test hanya 16 klip, jadi bandingkan model dengan CV Accuracy, bukan Accuracy test.")
        c1, c2 = st.columns(2)
        tampil_gambar(c1, "confusion_matrix.png", "Confusion matrix (data test)")
        tampil_gambar(c2, "roc_curve.png", "Kurva ROC (data test)")

        if t_pen is not None:
            st.divider()
            st.subheader("Tahap 1: penolak suara lain")
            st.dataframe(t_pen)
            if asing is not None:
                st.metric("Rata-rata penolakan pada kategori tak dikenal",
                          f"{asing['tingkat_penolakan'].mean() * 100:.1f}%")
                st.bar_chart(asing["tingkat_penolakan"] * 100, horizontal=True)
                st.caption("Kategori ini tidak pernah dipakai saat training penolak. "
                           "Rata-rata di atas dihitung per kategori.")
        st.divider()
        tampil_gambar(st, "contoh_sinyal.png", "Contoh sinyal tiap kelas")
