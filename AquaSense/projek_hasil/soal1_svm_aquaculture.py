"""
SOAL 1 - Model AI: Klasifikasi Kualitas Air Budidaya (Tilapia) dengan SVM
Anggota 1: Lead AI & Machine Learning

Output:
  - svm_aquaculture_model.pkl   (model SVM terlatih)
  - scaler_aquaculture.pkl      (StandardScaler)
  - model_meta.pkl              (nama fitur + nama kelas, untuk Anggota 2 / FastAPI)
  - output_soal1/*.png          (grafik untuk laporan)
  - output_soal1/hasil_evaluasi.txt & deskripsi_laporan.md (bahan laporan)

Cara pakai:
  pip install pandas numpy scikit-learn matplotlib seaborn openpyxl joblib
  python soal1_svm_aquaculture.py
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")

# =====================================================================
# 0. KONFIGURASI  (sesuaikan bila perlu)
# =====================================================================
DATA_PATH = "Data_Model_IoTMLCQ_2024.xlsx"   # ganti bila nama file Anda "Copy_of_Data_Model_IoTMLCQ_2024.xlsx"
SHEET_NAME = 0              # nama/nomor sheet
TARGET_COL = "Health Status"   # label: "Stable" vs "At Risk"

# Fitur = 4 parameter kualitas air dari sensor IoT.
# Kolom lain SENGAJA tidak dipakai (lihat EXCLUDED_NOTES di bawah).
FEATURE_COLS = ["Temperature (°C)", "Dissolved Oxygen (mg/L)", "pH", "Turbidity (NTU)"]
EXCLUDED_NOTES = {
    "Thermal Risk Index": "DATA LEAKAGE - isinya identik 100% dengan label Health Status (High=At Risk, Normal=Stable)",
    "Corrective Measures / Corrective Interventions": "tindakan sesudah masalah terjadi (bocoran label), bukan parameter air",
    "Datetime, Month, Month_Num, month_x, month_y, day, hour": "fitur waktu; 702 dari 705 data At Risk terjadi di Maret, model hanya akan menghafal bulan",
    "Low Oxygen Alert": "konstan (hanya 1 nilai 'Safe')",
    "Average/High/Low Temperature (°C)": "konstan (std = 0)",
    "Average Fish Weight, Survival Rate, Disease Occurrence": "kondisi ikan (hasil), bukan input sensor kualitas air",
    "Precipitation, oxigeno_scaled, ph2, turbidez, Oxygenation *": "tidak diperlukan / duplikat versi scaled dari fitur utama",
}
DROP_COLS = []              # kolom tambahan yang ingin dibuang manual
CLIP_OUTLIERS = False       # False: nilai ekstrem pada data ini adalah kondisi 'At Risk' asli, jangan diubah
TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_PATH = "svm_aquaculture_model.pkl"
SCALER_PATH = "scaler_aquaculture.pkl"
META_PATH = "model_meta.pkl"
OUT_DIR = "output_soal1"
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================================
# 1. LOAD DATA
# =====================================================================
print("=" * 60, "\n1. LOAD DATA\n" + "=" * 60)
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
df.columns = [str(c).strip() for c in df.columns]
print("Ukuran awal :", df.shape)
print("Kolom       :", list(df.columns))
print(df.head())

# =====================================================================
# 2. PREPROCESSING
# =====================================================================
print("\n" + "=" * 60, "\n2. DATA PREPROCESSING\n" + "=" * 60)
n_awal = len(df)

# 2a. Tentukan target
target = TARGET_COL if TARGET_COL else df.columns[-1]
if target not in df.columns:
    raise ValueError(f"Kolom target '{target}' tidak ditemukan. Cek TARGET_COL.")
print("Kolom target:", target)

# 2b. Buang kolom tidak relevan (manual + otomatis: datetime / ID)
drop = [c for c in DROP_COLS if c in df.columns]
for c in df.columns:
    if c == target:
        continue
    if pd.api.types.is_datetime64_any_dtype(df[c]) or c.lower() in ("id", "no", "index", "timestamp", "time", "date", "datetime"):
        drop.append(c)
df = df.drop(columns=list(set(drop)))
print("Kolom dibuang:", sorted(set(drop)) if drop else "-")

# 2c. Hapus duplikat & baris tanpa label
dup = df.duplicated().sum()
df = df.drop_duplicates()
df = df.dropna(subset=[target])
print(f"Duplikat dihapus: {dup}")

# 2d. Fitur: pakai FEATURE_COLS bila diisi; paksa numerik, imputasi median untuk missing value
X = (df[FEATURE_COLS] if FEATURE_COLS else df.drop(columns=[target])).apply(pd.to_numeric, errors="coerce")
X = X.dropna(axis=1, how="all")
missing = int(X.isna().sum().sum())
X = X.fillna(X.median(numeric_only=True))
print(f"Missing value diisi median: {missing}")

# 2e. Outlier: dideteksi dengan IQR; di-clip hanya jika CLIP_OUTLIERS=True
Q1, Q3 = X.quantile(0.25), X.quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
n_outlier = int(((X < lower) | (X > upper)).sum().sum())
if CLIP_OUTLIERS:
    X = X.clip(lower=lower, upper=upper, axis=1)
print(f"Nilai di luar batas IQR: {n_outlier} ({'di-clip' if CLIP_OUTLIERS else 'dipertahankan, valid sebagai kondisi At Risk'})")

# 2f. Encode label
y_raw = df[target]
if (not pd.api.types.is_numeric_dtype(y_raw)) or y_raw.nunique() <= 10:
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    class_names = [str(c) for c in le.classes_]
else:
    raise ValueError(
        f"Target '{target}' bernilai kontinu ({y_raw.nunique()} nilai unik). "
        "SVM klasifikasi butuh label kategori; ubah target atau buat kelas (mis. Baik/Buruk).")
feature_names = list(X.columns)
print("Fitur       :", feature_names)
print("Kelas       :", class_names)
print("Distribusi  :", dict(zip(class_names, np.bincount(y))))
print(f"Ukuran akhir: {X.shape[0]} baris (awal {n_awal})")

# 2g. Split (stratified) -> baru scaling (fit HANYA di data train, mencegah data leakage)
strat = y if np.bincount(y).min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# --- Grafik EDA ---
plt.figure(figsize=(6, 4))
sns.countplot(x=pd.Series(y).map(dict(enumerate(class_names))), palette="viridis")
plt.title("Distribusi Kelas Kualitas Air"); plt.xlabel("Kelas"); plt.ylabel("Jumlah")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/01_distribusi_kelas.png", dpi=200); plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Korelasi Antar Fitur"); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_korelasi_fitur.png", dpi=200); plt.close()

# =====================================================================
# 3. TRAINING + HYPERPARAMETER TUNING (GridSearchCV)
# =====================================================================
print("\n" + "=" * 60, "\n3. TRAINING SVM + HYPERPARAMETER TUNING\n" + "=" * 60)
param_grid = [
    {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1]},
    {"kernel": ["linear"], "C": [0.1, 1, 10, 100]},
    {"kernel": ["poly"], "C": [0.1, 1, 10], "degree": [2, 3], "gamma": ["scale"]},
]
min_class = np.bincount(y_train).min()
cv = StratifiedKFold(n_splits=max(2, min(5, int(min_class))), shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(SVC(class_weight="balanced", random_state=RANDOM_STATE),
                    param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
grid.fit(X_train_s, y_train)
print("Parameter terbaik :", grid.best_params_)
print(f"F1-macro CV terbaik: {grid.best_score_:.4f}")

# Model final (probability=True agar API bisa menampilkan tingkat keyakinan)
best_model = SVC(**grid.best_params_, class_weight="balanced",
                 probability=True, random_state=RANDOM_STATE)
best_model.fit(X_train_s, y_train)

# =====================================================================
# 4. TESTING & EVALUASI
# =====================================================================
print("\n" + "=" * 60, "\n4. EVALUASI MODEL\n" + "=" * 60)
y_pred = best_model.predict(X_test_s)
y_pred_train = best_model.predict(X_train_s)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
acc_train = accuracy_score(y_train, y_pred_train)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

print(f"Accuracy (train): {acc_train:.4f}")
print(f"Accuracy (test) : {acc:.4f}")
print(f"Precision macro : {prec:.4f}")
print(f"Recall macro    : {rec:.4f}")
print(f"F1-score macro  : {f1:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# --- Grafik evaluasi ---
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix - SVM (Data Test)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/03_confusion_matrix.png", dpi=200); plt.close()

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix Ternormalisasi"); plt.xlabel("Prediksi"); plt.ylabel("Aktual")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/04_confusion_matrix_normalized.png", dpi=200); plt.close()

plt.figure(figsize=(6, 4))
vals = [acc, prec, rec, f1]
bars = plt.bar(["Accuracy", "Precision", "Recall", "F1-Score"], vals,
               color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
plt.ylim(0, 1.1); plt.title("Metrik Evaluasi Model SVM")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/05_metrik_evaluasi.png", dpi=200); plt.close()

# Per-kelas precision/recall
rep = classification_report(y_test, y_pred, target_names=class_names,
                            output_dict=True, zero_division=0)
pc = pd.DataFrame({k: rep[k] for k in class_names}).T[["precision", "recall", "f1-score"]]
pc.plot(kind="bar", figsize=(7, 4), rot=0, colormap="viridis")
plt.ylim(0, 1.1); plt.title("Precision, Recall, F1 per Kelas")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/06_metrik_per_kelas.png", dpi=200); plt.close()

# Hasil tuning (rbf) sebagai heatmap
res = pd.DataFrame(grid.cv_results_)
rbf = res[res["param_kernel"] == "rbf"]
if len(rbf):
    piv = rbf.pivot_table(index="param_C", columns="param_gamma",
                          values="mean_test_score", aggfunc="max")
    plt.figure(figsize=(6, 4))
    sns.heatmap(piv.astype(float), annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Hasil Tuning RBF (F1-macro CV)"); plt.xlabel("gamma"); plt.ylabel("C")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/07_hasil_tuning.png", dpi=200); plt.close()

# =====================================================================
# 5. SIMPAN MODEL, SCALER, METADATA
# =====================================================================
print("\n" + "=" * 60, "\n5. SIMPAN MODEL\n" + "=" * 60)
joblib.dump(best_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump({"feature_names": feature_names, "class_names": class_names,
             "target": target, "best_params": grid.best_params_}, META_PATH)
print("Tersimpan:", MODEL_PATH, SCALER_PATH, META_PATH)

# Uji muat ulang (pastikan file .pkl bisa dipakai di FastAPI)
m2, s2 = joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
sample = pd.DataFrame([X_test.iloc[0]], columns=feature_names)
pred = m2.predict(s2.transform(sample))[0]
proba = m2.predict_proba(s2.transform(sample))[0].max()
print(f"Uji muat ulang -> prediksi: {class_names[pred]} (keyakinan {proba:.2%}) | aktual: {class_names[y_test[0]]}")

# =====================================================================
# 6. BAHAN LAPORAN
# =====================================================================
with open(f"{OUT_DIR}/hasil_evaluasi.txt", "w", encoding="utf-8") as f:
    f.write(f"Parameter terbaik: {grid.best_params_}\n")
    f.write(f"F1-macro CV: {grid.best_score_:.4f}\n")
    f.write(f"Accuracy train: {acc_train:.4f}\nAccuracy test: {acc:.4f}\n")
    f.write(f"Precision macro: {prec:.4f}\nRecall macro: {rec:.4f}\nF1 macro: {f1:.4f}\n\n")
    f.write(f"Confusion Matrix:\n{cm}\n\n{report}")

dist = dict(zip(class_names, [int(v) for v in np.bincount(y)]))
excl = "\n".join(f"- {k}: {v}" for k, v in EXCLUDED_NOTES.items())
md = f"""# Bahan Laporan - Soal 1 (Model AI)

## A. Deskripsi Dataset
Dataset **IoT Monitoring of Water Quality and Tilapia** (file `{os.path.basename(DATA_PATH)}`) berisi data
pemantauan kualitas air budidaya ikan nila yang direkam sensor IoT. Setelah pembersihan, data berjumlah
**{X.shape[0]} baris** (awal {n_awal}) dengan **{len(feature_names)} fitur**: {", ".join(feature_names)}.
Kolom target: **{target}** dengan kelas {class_names}. Distribusi kelas: {dist}.

## B. Preprocessing
1. Menghapus kolom tidak relevan dan **{dup} data duplikat**.
2. Mengisi **{missing} nilai kosong** dengan median tiap fitur.
3. Deteksi outlier metode IQR: **{n_outlier} nilai** di luar batas, dipertahankan karena merupakan pembacaan sensor valid (kondisi At Risk).
4. Encoding label ke angka (LabelEncoder).
5. Membagi data train/test = {int((1-TEST_SIZE)*100)}:{int(TEST_SIZE*100)} secara stratified (random_state={RANDOM_STATE}).
6. **Standardisasi** fitur dengan `StandardScaler` (fit hanya pada data latih untuk mencegah data leakage).

Kolom yang **tidak dipakai sebagai fitur**:
{excl}

## C. Model: Support Vector Machine (SVM)
SVM mencari hyperplane optimal yang memaksimalkan margin antar kelas, dan dengan fungsi kernel
mampu memisahkan data non-linear. Cocok untuk data sensor numerik berdimensi menengah.
Tuning dengan **GridSearchCV** ({cv.get_n_splits()}-fold Stratified CV, skor F1-macro) pada kernel rbf/linear/poly,
nilai C, gamma, dan degree. `class_weight='balanced'` dipakai untuk menangani ketidakseimbangan kelas.

**Parameter terbaik:** `{grid.best_params_}`

## D. Hasil Evaluasi (Data Test)
| Metrik | Nilai |
|---|---|
| Accuracy | {acc:.4f} |
| Precision (macro) | {prec:.4f} |
| Recall (macro) | {rec:.4f} |
| F1-score (macro) | {f1:.4f} |

Accuracy train = {acc_train:.4f} (bandingkan dengan test untuk menilai overfitting).

Confusion matrix:
```
{cm}
```
Classification report:
```
{report}
```
Grafik: lihat folder `{OUT_DIR}/` (03_confusion_matrix.png, 05_metrik_evaluasi.png, dst).

## E. Penyimpanan Model
Model disimpan sebagai `{MODEL_PATH}` dan scaler `{SCALER_PATH}` menggunakan joblib,
serta `{META_PATH}` berisi urutan fitur & nama kelas agar dapat dipakai di endpoint FastAPI.
"""
with open(f"{OUT_DIR}/deskripsi_laporan.md", "w", encoding="utf-8") as f:
    f.write(md)

print("\nSelesai. Cek folder:", OUT_DIR)
