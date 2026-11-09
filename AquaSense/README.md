# AquaSense — Analisis Kualitas Air Budidaya Ikan Nila

Aplikasi web untuk mengklasifikasikan kondisi kualitas air kolam ikan nila
(**Stable** / **At Risk**) menggunakan model **Support Vector Machine**, lengkap
dengan penyimpanan riwayat ke database dan rekomendasi tindakan.

Dibuat untuk Tugas Besar mata kuliah **Framework**.

---

## 1. Fitur

| Fitur | Keterangan |
|---|---|
| Model AI | SVM kernel polynomial (C=1, degree=2), StandardScaler, ekspor `.pkl` |
| Halaman Home | Tujuan aplikasi, detail dataset, konfigurasi & performa model, grafik evaluasi |
| Halaman Prediksi | Form 4 parameter sensor + slider, validasi, hasil prediksi, probabilitas, rekomendasi |
| Halaman Riwayat | Tabel data dari database, pencarian nama kolam, filter status, hapus data |
| Halaman Tentang Kami | Profil dan pembagian tugas 5 anggota kelompok |
| Database | SQLite + SQLAlchemy (Create, Read, Delete) |
| REST API | `POST /api/predict`, `GET /api/logs`, `DELETE /api/logs/{id}`, `GET /health` |

---

## 2. Struktur Project

```
frmwork/
├── main.py                     # Route FastAPI (Home, Predict, History, Contact, API)
├── database.py                 # Engine, session, Base SQLAlchemy
├── models_db.py                # Tabel water_quality_logs
├── ml_service.py               # Load model .pkl, prediksi, rekomendasi, info model
├── requirements.txt
├── models/
│   ├── svm_aquaculture_model.pkl
│   ├── scaler_aquaculture.pkl
│   └── model_meta.pkl
├── templates/                  # Jinja2 (Soal 2A)
│   ├── base.html               # Layout, navbar, footer
│   ├── index.html              # Home
│   ├── predict.html            # Form input + hasil prediksi
│   ├── history.html            # Tabel riwayat + modal hapus
│   └── contact.html            # Profil anggota
├── static/
│   ├── css/style.css           # Design system & responsive layout
│   ├── js/main.js              # Slider, validasi, hapus via fetch, toast, navbar mobile
│   └── img/
│       ├── favicon.svg
│       └── charts/             # Grafik evaluasi model untuk laporan
├── projek_hasil/               # Arsip hasil Soal 1 (script training, grafik, laporan)
└── aquasense.db                # Database SQLite (dibuat otomatis saat server jalan)
```

---

## 3. Cara Menjalankan

```bash
# 1. (Opsional) buat virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Install dependency
pip install -r requirements.txt

# 3. Jalankan server
uvicorn main:app --reload
```

Buka **http://127.0.0.1:8000**.

Database `aquasense.db` dan tabelnya dibuat otomatis saat aplikasi pertama kali
dijalankan. Dokumentasi API interaktif tersedia di **http://127.0.0.1:8000/docs**.

---

## 4. Skema Database

Tabel `water_quality_logs`:

| Kolom | Tipe | Keterangan |
|---|---|---|
| `id` | INTEGER (PK) | Auto increment |
| `pond_name` | VARCHAR(100) | Nama kolam |
| `temperature` | FLOAT | Suhu air (°C) |
| `dissolved_oxygen` | FLOAT | Oksigen terlarut (mg/L) |
| `ph` | FLOAT | pH air |
| `turbidity` | FLOAT | Kekeruhan (NTU) |
| `prediction` | VARCHAR(20) | `Stable` / `At Risk` |
| `confidence` | FLOAT | Probabilitas kelas terpilih (0–1) |
| `recommendation` | VARCHAR(500) | Rekomendasi tindakan |
| `created_at` | DATETIME | Waktu analisis |

---

## 5. Integrasi Model ke FastAPI

```python
# ml_service.py
row = pd.DataFrame([[temperature, dissolved_oxygen, ph, turbidity]],
                   columns=FEATURE_NAMES)   # urutan fitur dari model_meta.pkl
scaled = scaler.transform(row)              # StandardScaler hasil training
index = model.predict(scaled)[0]
probabilities = model.predict_proba(scaled)[0]
```

Urutan fitur **wajib** sama dengan saat training:
`Temperature (°C)`, `Dissolved Oxygen (mg/L)`, `pH`, `Turbidity (NTU)`.

---

## 6. Pembagian Tugas

| Anggota | Peran | Fokus |
|---|---|---|
| Refi | Lead AI & Machine Learning | Soal 1: preprocessing, tuning SVM, evaluasi, ekspor `.pkl` |
| Bernhard | Backend & Database Engineer | Soal 2B: FastAPI, SQLAlchemy, CRUD, integrasi model |
| **Devin** | **Frontend Engineer & UI/UX** | **Soal 2A: Jinja2, CSS, JS, Home, Contact, form, tabel riwayat** |
| Vino | Technical Writer & Lead Laporan | Soal 4: laporan PDF, identifikasi masalah, dataset, algoritma |
| Lutfi | QA, System Integrator & Video Producer | Soal 4C: end-to-end testing, video demo |

---

## 7. Contoh Data untuk Demo

| Skenario | Suhu | DO | pH | Turbidity | Hasil |
|---|---|---|---|---|---|
| Kondisi ideal | 27.5 | 6.7 | 7.89 | 3.3 | Stable |
| Data sensor at risk | 28.3 | 7.85 | 7.97 | 2.79 | At Risk |

> Model dilatih pada rentang data yang sempit (suhu 26.5–28.35 °C, DO 6.34–7.89 mg/L,
> pH 7.34–7.98, turbidity 2.79–3.98 NTU). Jika input di luar rentang tersebut, aplikasi
> menampilkan peringatan bahwa prediksi perlu diverifikasi manual.
