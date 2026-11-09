# Bahan Laporan - Soal 1 (Model AI)

## A. Deskripsi Dataset
Dataset **IoT Monitoring of Water Quality and Tilapia** (file `Data_Model_IoTMLCQ_2024.xlsx`) berisi data
pemantauan kualitas air budidaya ikan nila yang direkam sensor IoT. Setelah pembersihan, data berjumlah
**4313 baris** (awal 4383) dengan **4 fitur**: Temperature (°C), Dissolved Oxygen (mg/L), pH, Turbidity (NTU).
Kolom target: **Health Status** dengan kelas ['At Risk', 'Stable']. Distribusi kelas: {'At Risk': 705, 'Stable': 3608}.

## B. Preprocessing
1. Menghapus kolom tidak relevan dan **70 data duplikat**.
2. Mengisi **0 nilai kosong** dengan median tiap fitur.
3. Deteksi outlier metode IQR: **723 nilai** di luar batas, dipertahankan karena merupakan pembacaan sensor valid (kondisi At Risk).
4. Encoding label ke angka (LabelEncoder).
5. Membagi data train/test = 80:20 secara stratified (random_state=42).
6. **Standardisasi** fitur dengan `StandardScaler` (fit hanya pada data latih untuk mencegah data leakage).

Kolom yang **tidak dipakai sebagai fitur**:
- Thermal Risk Index: DATA LEAKAGE - isinya identik 100% dengan label Health Status (High=At Risk, Normal=Stable)
- Corrective Measures / Corrective Interventions: tindakan sesudah masalah terjadi (bocoran label), bukan parameter air
- Datetime, Month, Month_Num, month_x, month_y, day, hour: fitur waktu; 702 dari 705 data At Risk terjadi di Maret, model hanya akan menghafal bulan
- Low Oxygen Alert: konstan (hanya 1 nilai 'Safe')
- Average/High/Low Temperature (°C): konstan (std = 0)
- Average Fish Weight, Survival Rate, Disease Occurrence: kondisi ikan (hasil), bukan input sensor kualitas air
- Precipitation, oxigeno_scaled, ph2, turbidez, Oxygenation *: tidak diperlukan / duplikat versi scaled dari fitur utama

## C. Model: Support Vector Machine (SVM)
SVM mencari hyperplane optimal yang memaksimalkan margin antar kelas, dan dengan fungsi kernel
mampu memisahkan data non-linear. Cocok untuk data sensor numerik berdimensi menengah.
Tuning dengan **GridSearchCV** (5-fold Stratified CV, skor F1-macro) pada kernel rbf/linear/poly,
nilai C, gamma, dan degree. `class_weight='balanced'` dipakai untuk menangani ketidakseimbangan kelas.

**Parameter terbaik:** `{'C': 1, 'degree': 2, 'gamma': 'scale', 'kernel': 'poly'}`

## D. Hasil Evaluasi (Data Test)
| Metrik | Nilai |
|---|---|
| Accuracy | 1.0000 |
| Precision (macro) | 1.0000 |
| Recall (macro) | 1.0000 |
| F1-score (macro) | 1.0000 |

Accuracy train = 0.9994 (bandingkan dengan test untuk menilai overfitting).

Confusion matrix:
```
[[141   0]
 [  0 722]]
```
Classification report:
```
              precision    recall  f1-score   support

     At Risk       1.00      1.00      1.00       141
      Stable       1.00      1.00      1.00       722

    accuracy                           1.00       863
   macro avg       1.00      1.00      1.00       863
weighted avg       1.00      1.00      1.00       863

```
Grafik: lihat folder `output_soal1/` (03_confusion_matrix.png, 05_metrik_evaluasi.png, dst).

## E. Penyimpanan Model
Model disimpan sebagai `svm_aquaculture_model.pkl` dan scaler `scaler_aquaculture.pkl` menggunakan joblib,
serta `model_meta.pkl` berisi urutan fitur & nama kelas agar dapat dipakai di endpoint FastAPI.
