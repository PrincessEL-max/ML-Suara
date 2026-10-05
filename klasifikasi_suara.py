"""
Klasifikasi suara ANJING vs KUCING + PENOLAK untuk suara selain keduanya.

Alur dua tahap:
    1. Penolak (gate)  : suara ini anjing/kucing atau BUKAN?  (model biner)
    2. Klasifikasi     : kalau lolos, anjing atau kucing?     (model biner)

Dataset : ESC-50 (https://github.com/karolpiczak/ESC-50)
    data/anjing, data/kucing : 40 klip per kelas
    data/lainnya             : 180 klip dari 36 kategori lain -> training penolak
    data/asing               :  60 klip dari 12 kategori lain -> HANYA evaluasi penolak
                                (kategori ini tidak pernah dilihat saat training)
Fitur   : MFCC + fitur spektral (librosa)
Model   : SVM (RBF), Random Forest, k-NN
Evaluasi: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, cross-validation

Cara pakai:
    pip install librosa soundfile scikit-learn pandas matplotlib requests joblib
    python klasifikasi_suara.py --download                  # unduh dataset (sekali saja)
    python klasifikasi_suara.py                             # latih + evaluasi + simpan model
    python klasifikasi_suara.py --predict rekaman.wav       # tes 1 file suara sendiri
    python klasifikasi_suara.py --predict folder_rekaman/   # tes semua .wav di folder
"""

import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ------------------------- KONFIGURASI -------------------------
DATA_DIR = Path("data")
OUT_DIR = Path("hasil")
KELAS = {"anjing": 0, "kucing": 1}      # nama folder -> label
ESC50_KATEGORI = {"dog": "anjing", "cat": "kucing"}
# Kategori yang SENGAJA tidak dipakai training penolak (untuk uji suara tak dikenal)
KATEGORI_ASING = ["rooster", "cow", "pig", "crow", "crying_baby", "laughing",
                  "coughing", "rain", "siren", "vacuum_cleaner", "keyboard_typing", "clock_alarm"]
ESC50_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
SR = 22050          # sample rate
DURASI = 5.0        # detik
N_MFCC = 20
N_AUG = 2           # jumlah salinan augmentasi noise per klip training
SEED = 42
MODEL_KLASIFIKASI = OUT_DIR / "model_final.joblib"
MODEL_PENOLAK = OUT_DIR / "model_penolak.joblib"
AMBANG_YAKIN = 0.70     # klasifikasi anjing/kucing di bawah ini -> ditandai tidak yakin
AMBANG_PENOLAK = 0.50   # peluang "anjing/kucing" di bawah ini -> ditolak


# ------------------------- 1. DATASET -------------------------
def _unduh_satu(args):
    nama_file, tujuan = args
    if not tujuan.exists():
        tujuan.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"{ESC50_URL}/audio/{nama_file}", timeout=60)
        r.raise_for_status()
        tujuan.write_bytes(r.content)


def unduh_dataset():
    """Unduh ESC-50 subset. Simpan fold + kategori asli ke data/metadata.csv."""
    meta = pd.read_csv(f"{ESC50_URL}/meta/esc50.csv").sort_values("filename")
    tugas = []  # (nama_file, tujuan, fold, kategori)

    # a) anjing & kucing: semua klip
    for kat, folder in ESC50_KATEGORI.items():
        for _, r in meta[meta["category"] == kat].iterrows():
            tugas.append((r["filename"], DATA_DIR / folder / r["filename"], r["fold"], kat))

    # b) negatif: per kategori, 1 klip untuk tiap fold (fold 1-5)
    for kat in sorted(meta["category"].unique()):
        if kat in ESC50_KATEGORI:
            continue
        folder = "asing" if kat in KATEGORI_ASING else "lainnya"
        sub = meta[meta["category"] == kat].groupby("fold").head(1)
        for _, r in sub.iterrows():
            tugas.append((r["filename"], DATA_DIR / folder / r["filename"], r["fold"], kat))

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_unduh_satu, [(t[0], t[1]) for t in tugas]))

    pd.DataFrame([{"file": str(t[1]), "fold": t[2], "kategori": t[3]} for t in tugas]
                 ).to_csv(DATA_DIR / "metadata.csv", index=False)
    print(f"Selesai: {len(tugas)} klip di '{DATA_DIR}/'")


def baca_folder(nama, label=None):
    """Daftar file .wav di data/<nama> (+ fold & kategori kalau ada metadata.csv)."""
    files = sorted((DATA_DIR / nama).glob("*.wav"))
    df = pd.DataFrame({"file": [str(f) for f in files]})
    df["kelas"], df["label"] = nama, label
    meta_path = DATA_DIR / "metadata.csv"
    if meta_path.exists() and not df.empty:
        df = df.merge(pd.read_csv(meta_path), on="file", how="left")
    return df


def muat_anjing_kucing():
    df = pd.concat([baca_folder(n, l) for n, l in KELAS.items()], ignore_index=True)
    if df.empty:
        raise SystemExit("Dataset kosong. Jalankan dengan --download dulu.")
    return df


def ambil_fold(df):
    return df["fold"].values if "fold" in df and df["fold"].notna().all() else None


def pisah_data(y, fold):
    """Kembalikan idx_train, idx_test, objek CV, groups."""
    idx = np.arange(len(y))
    if fold is not None:   # ESC-50: test = fold 5, supaya tidak bocor antar rekaman sumber
        te = fold == 5
        return idx[~te], idx[te], GroupKFold(n_splits=5), fold
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    return tr, te, StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED), None


# ------------------------- 2. EKSTRAKSI FITUR -------------------------
def augmentasi(y, sr, rng):
    """Simulasi rekaman HP: lewat 16 kHz, gain acak, noise putih SNR 10-30 dB."""
    y = librosa.resample(librosa.resample(y, orig_sr=sr, target_sr=16000), orig_sr=16000, target_sr=sr)
    y = y * rng.uniform(0.3, 1.5)
    snr = rng.uniform(10, 30)
    return y + rng.standard_normal(len(y)) * np.sqrt(np.mean(y ** 2) / 10 ** (snr / 10))


def ekstrak_fitur(path, rng=None):
    y, sr = librosa.load(path, sr=SR, duration=DURASI)
    y, _ = librosa.effects.trim(y, top_db=30)          # buang hening di awal/akhir
    if len(y) < 2048:                                  # jaga-jaga klip terlalu pendek
        y = np.pad(y, (0, 2048 - len(y)))
    if rng is not None:                                # mode augmentasi (hanya saat training/uji noise)
        y = augmentasi(y, sr, rng)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    def ms(x):  # mean + std per baris
        return np.concatenate([x.mean(axis=1), x.std(axis=1)])

    return np.concatenate([ms(mfcc), ms(delta), ms(centroid), ms(bandwidth),
                           ms(rolloff), ms(contrast), ms(zcr), ms(rms)])


def bangun_matriks_fitur(df):
    print(f"Ekstraksi fitur dari {len(df)} file...")
    return np.vstack([ekstrak_fitur(f) for f in df["file"]])


def fitur_aug(df, n=N_AUG, seed=SEED):
    """n salinan fitur ber-noise untuk semua baris df (cache tidak dipakai, cukup cepat)."""
    print(f"Augmentasi noise: {n} x {len(df)} file...")
    rng = np.random.default_rng(seed)
    return [np.vstack([ekstrak_fitur(f, rng) for f in df["file"]]) for _ in range(n)]


def latih_aug(model, X, XA, y, tr):
    """Latih di data asli + salinan augmentasi, HANYA untuk indeks tr (test tidak disentuh)."""
    Xtr = np.vstack([X[tr]] + [a[tr] for a in XA])
    return clone(model).fit(Xtr, np.tile(y[tr], len(XA) + 1))


def cv_aug(model, X, XA, y, cv, groups):
    """Prediksi cross-validation; augmentasi hanya di fold training, validasi tetap bersih."""
    pred = np.zeros(len(y), int)
    for tr, va in cv.split(X, y, groups):
        pred[va] = latih_aug(model, X, XA, y, tr).predict(X[va])
    return pred


# ------------------------- 3. MODEL -------------------------
def buat_model(seimbang=False):
    """seimbang=True -> class_weight='balanced' (untuk penolak, data tidak seimbang)."""
    cw = "balanced" if seimbang else None
    return {
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, probability=True,
                                                         class_weight=cw, random_state=SEED)),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight=cw, random_state=SEED),
        "k-NN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }


# ------------------------- 4. EVALUASI -------------------------
def hitung_metrik(y_true, y_pred, y_prob):
    """Metrik biner; kelas positif = label 1."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


def plot_contoh_sinyal(df):
    """Waveform + mel-spectrogram satu contoh per kelas."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 6))
    for i, nama in enumerate(KELAS):
        path = df[df["kelas"] == nama]["file"].iloc[0]
        y, sr = librosa.load(path, sr=SR, duration=DURASI)
        librosa.display.waveshow(y, sr=sr, ax=ax[0, i])
        ax[0, i].set_title(f"Waveform - {nama}")
        S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max)
        librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="mel", ax=ax[1, i])
        ax[1, i].set_title(f"Mel-spectrogram - {nama}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "contoh_sinyal.png", dpi=150)
    plt.close()


def evaluasi_anjing_kucing(df, X):
    """Tahap 2: anjing vs kucing. Simpan model final."""
    y = df["label"].values.astype(int)
    nama_kelas = list(KELAS)
    tr, te, cv, groups = pisah_data(y, ambil_fold(df))
    print("Split:", "train = fold 1-4, test = fold 5 (anti-leakage)" if groups is not None
          else "acak stratified 80/20")
    print(f"Train: {len(tr)} | Test: {len(te)} | Jumlah fitur: {X.shape[1]}\n")

    XA, XN = fitur_aug(df), fitur_aug(df, 1, SEED + 99)[0]   # XN = versi noisy untuk uji ketahanan
    hasil, cm_semua, roc_data = [], {}, {}
    for nama, model in buat_model().items():
        model = latih_aug(model, X, XA, y, tr)
        pred = model.predict(X[te])
        prob = model.predict_proba(X[te])[:, 1]
        m = hitung_metrik(y[te], pred, prob)
        m["Acc (noisy)"] = accuracy_score(y[te], model.predict(XN[te]))
        m["CV Accuracy"] = accuracy_score(y, cv_aug(model, X, XA, y, cv, groups))
        hasil.append({"Model": nama, **m})
        cm_semua[nama] = confusion_matrix(y[te], pred)
        roc_data[nama] = roc_curve(y[te], prob)
        print(f"=== {nama} ===")
        print(classification_report(y[te], pred, target_names=nama_kelas, digits=3))

    tabel = pd.DataFrame(hasil).set_index("Model").round(3)
    print("=== PERBANDINGAN MODEL (anjing vs kucing) ===")
    print(tabel.to_string())
    tabel.to_csv(OUT_DIR / "perbandingan_model.csv")

    terbaik = tabel["CV Accuracy"].idxmax()
    joblib.dump({"model": latih_aug(buat_model()[terbaik], X, XA, y, np.arange(len(y))), "nama": terbaik}, MODEL_KLASIFIKASI)
    print(f"\nModel klasifikasi final: {terbaik} (dilatih di semua data)")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, (nama, cm) in zip(ax, cm_semua.items()):
        ConfusionMatrixDisplay(cm, display_labels=nama_kelas).plot(ax=a, colorbar=False, cmap="Blues")
        a.set_title(nama)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5.5, 5))
    for nama, (fpr, tpr, _) in roc_data.items():
        plt.plot(fpr, tpr, label=nama)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Kurva ROC anjing vs kucing (data test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curve.png", dpi=150)
    plt.close()


def latih_penolak(df_t, X_t):
    """Tahap 1: penolak. Positif (1) = anjing/kucing, negatif (0) = suara lain."""
    df_l = baca_folder("lainnya", 0)
    if df_l.empty:
        print("\n[Penolak dilewati] data/lainnya kosong. Jalankan --download.")
        return
    print("\n" + "=" * 60 + "\nPENOLAK: anjing/kucing vs suara lain\n" + "=" * 60)
    X_l = bangun_matriks_fitur(df_l)
    X = np.vstack([X_t, X_l])
    y = np.r_[np.ones(len(X_t)), np.zeros(len(X_l))].astype(int)
    fold = None
    if ambil_fold(df_t) is not None and ambil_fold(df_l) is not None:
        fold = np.r_[ambil_fold(df_t), ambil_fold(df_l)]
    tr, te, cv, groups = pisah_data(y, fold)
    print(f"Train: {len(tr)} | Test: {len(te)} | anjing/kucing: {y.sum()} | lainnya: {(y == 0).sum()}\n")

    XA = [np.vstack(p) for p in zip(fitur_aug(df_t), fitur_aug(df_l))]
    XN = np.vstack([fitur_aug(df_t, 1, SEED + 99)[0], fitur_aug(df_l, 1, SEED + 99)[0]])
    hasil = []
    for nama, model in buat_model(seimbang=True).items():
        model = latih_aug(model, X, XA, y, tr)
        pred = model.predict(X[te])
        prob = model.predict_proba(X[te])[:, 1]
        m = hitung_metrik(y[te], pred, prob)
        m["Bal. Accuracy"] = balanced_accuracy_score(y[te], pred)
        m["Recall (noisy)"] = recall_score(y[te], model.predict(XN[te]))
        m["CV Bal. Accuracy"] = balanced_accuracy_score(y, cv_aug(model, X, XA, y, cv, groups))
        hasil.append({"Model": nama, **m})
    tabel = pd.DataFrame(hasil).set_index("Model").round(3)
    print("=== PERBANDINGAN MODEL PENOLAK (data test, positif = anjing/kucing) ===")
    print(tabel.to_string())
    tabel.to_csv(OUT_DIR / "perbandingan_penolak.csv")

    terbaik = tabel["CV Bal. Accuracy"].idxmax()
    final = latih_aug(buat_model(seimbang=True)[terbaik], X, XA, y, np.arange(len(y)))
    joblib.dump({"model": final, "nama": terbaik}, MODEL_PENOLAK)
    print(f"\nPenolak final: {terbaik} (dilatih di semua data)")

    # --- Uji pada kategori yang TIDAK PERNAH dilihat saat training ---
    df_a = baca_folder("asing", 0)
    if df_a.empty:
        return
    X_a = bangun_matriks_fitur(df_a)
    df_a["ditolak"] = final.predict(X_a) == 0
    per_kat = df_a.groupby("kategori")["ditolak"].mean().sort_values()
    print("\n=== UJI KATEGORI TAK DIKENAL (penolak final, belum pernah melihat kategori ini) ===")
    print((per_kat * 100).round(0).astype(int).astype(str).add("% ditolak").to_string())
    print(f"\nTotal ditolak dengan benar: {df_a['ditolak'].sum()}/{len(df_a)} "
          f"({df_a['ditolak'].mean() * 100:.1f}%)")
    per_kat.to_csv(OUT_DIR / "penolak_kategori_asing.csv", header=["tingkat_penolakan"])

    plt.figure(figsize=(7, 4.5))
    plt.barh(per_kat.index, per_kat.values * 100, color="steelblue")
    plt.xlabel("% klip ditolak (benar)")
    plt.title("Penolak pada kategori yang tak pernah dilihat saat training")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "penolak_kategori_asing.png", dpi=150)
    plt.close()


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = muat_anjing_kucing()
    print(df["kelas"].value_counts().to_string(), "\n")
    plot_contoh_sinyal(df)
    X = bangun_matriks_fitur(df)
    evaluasi_anjing_kucing(df, X)
    latih_penolak(df, X)
    print(f"\nOutput tersimpan di '{OUT_DIR}/'")


# ------------------------- 5. TES DENGAN SUARA SENDIRI -------------------------
def prediksi(target):
    """Prediksi 1 file .wav atau semua .wav di sebuah folder."""
    if not MODEL_KLASIFIKASI.exists():
        raise SystemExit("Model belum ada. Jalankan 'python klasifikasi_suara.py' dulu.")
    klas = joblib.load(MODEL_KLASIFIKASI)["model"]
    penolak = joblib.load(MODEL_PENOLAK)["model"] if MODEL_PENOLAK.exists() else None
    nama_kelas = list(KELAS)

    target = Path(target)
    files = sorted(target.glob("*.wav")) if target.is_dir() else [target]
    if not files:
        raise SystemExit(f"Tidak ada file .wav di {target}")
    if penolak is None:
        print("(Penolak belum ada, semua suara dipaksa jadi anjing/kucing)\n")

    for f in files:
        fitur = ekstrak_fitur(str(f)).reshape(1, -1)
        if penolak is not None:
            p_target = penolak.predict_proba(fitur)[0][1]
            if p_target < AMBANG_PENOLAK:
                print(f"{f.name:30s} -> BUKAN anjing/kucing (peluang anjing/kucing {p_target*100:5.1f}%)")
                continue
        prob = klas.predict_proba(fitur)[0]
        i = int(prob.argmax())
        catatan = "" if prob[i] >= AMBANG_YAKIN else "  <- TIDAK YAKIN"
        print(f"{f.name:30s} -> {nama_kelas[i]:8s} (yakin {prob[i]*100:5.1f}%){catatan}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="unduh dataset ESC-50")
    ap.add_argument("--predict", metavar="PATH", help="prediksi file .wav / folder berisi .wav")
    args = ap.parse_args()
    if args.predict:
        prediksi(args.predict)
    else:
        if args.download:
            unduh_dataset()
        main()
