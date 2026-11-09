"""Service untuk memuat model SVM dan menjalankan prediksi kualitas air."""

from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "svm_aquaculture_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler_aquaculture.pkl")
meta = joblib.load(MODEL_DIR / "model_meta.pkl")

FEATURE_NAMES = meta["feature_names"]
CLASS_NAMES = meta["class_names"]
BEST_PARAMS = meta["best_params"]

STATUS_LABEL = {"At Risk": "At Risk", "Stable": "Stable"}

# Informasi model & dataset untuk halaman Home dan bahan laporan.
MODEL_INFO = {
    "algorithm": "Support Vector Machine (SVM)",
    "kernel": "Polynomial (degree 2)",
    "best_params": BEST_PARAMS,
    "cv_f1": 0.9989,
    "accuracy": 1.0000,
    "precision": 1.0000,
    "recall": 1.0000,
    "f1": 1.0000,
    "train_accuracy": 0.9994,
    "rows_raw": 4383,
    "rows_clean": 4313,
    "classes": {"Stable": 3608, "At Risk": 705},
    "features": FEATURE_NAMES,
}

FEATURE_META = [
    {
        "key": "temperature",
        "column": FEATURE_NAMES[0],
        "label": "Suhu Air",
        "unit": "°C",
        "icon": "temperature",
        "min": 20.0,
        "max": 35.0,
        "step": 0.1,
        "default": 27.5,
        "ideal": "25 – 30 °C",
        "hint": "Suhu memengaruhi metabolisme ikan dan kadar oksigen terlarut.",
    },
    {
        "key": "dissolved_oxygen",
        "column": FEATURE_NAMES[1],
        "label": "Oksigen Terlarut (DO)",
        "unit": "mg/L",
        "icon": "oxygen",
        "min": 0.0,
        "max": 12.0,
        "step": 0.01,
        "default": 6.7,
        "ideal": "≥ 5 mg/L",
        "hint": "DO rendah adalah penyebab kematian massal ikan yang paling umum.",
    },
    {
        "key": "ph",
        "column": FEATURE_NAMES[2],
        "label": "Derajat Keasaman (pH)",
        "unit": "pH",
        "icon": "ph",
        "min": 0.0,
        "max": 14.0,
        "step": 0.01,
        "default": 7.89,
        "ideal": "6.5 – 9.0",
        "hint": "pH di luar rentang ideal membuat ikan stres dan rentan penyakit.",
    },
    {
        "key": "turbidity",
        "column": FEATURE_NAMES[3],
        "label": "Kekeruhan (Turbidity)",
        "unit": "NTU",
        "icon": "turbidity",
        "min": 0.0,
        "max": 10.0,
        "step": 0.01,
        "default": 3.3,
        "ideal": "< 5 NTU",
        "hint": "Kekeruhan tinggi dapat menyumbat insang dan menurunkan penetrasi cahaya.",
    },
]


def predict_water_quality(temperature: float, dissolved_oxygen: float, ph: float, turbidity: float) -> dict:
    """Menjalankan model SVM dan mengembalikan kelas, keyakinan, serta probabilitas."""
    row = pd.DataFrame(
        [[temperature, dissolved_oxygen, ph, turbidity]],
        columns=FEATURE_NAMES,
    )
    scaled = scaler.transform(row)
    index = int(model.predict(scaled)[0])
    probabilities = model.predict_proba(scaled)[0]

    status = CLASS_NAMES[index]
    confidence = float(probabilities[index])
    proba_map = {name: float(p) for name, p in zip(CLASS_NAMES, probabilities)}

    return {
        "status": status,
        "status_label": STATUS_LABEL.get(status, status),
        "confidence": confidence,
        "probabilities": proba_map,
        "is_risk": status == "At Risk",
        "confidence_percent": round(confidence * 100, 1),
    }


def build_recommendation(status: str, temperature: float, dissolved_oxygen: float, ph: float, turbidity: float) -> str:
    """Menyusun rekomendasi tindakan berdasarkan hasil model dan ambang parameter air."""
    issues = []

    if dissolved_oxygen < 5.0:
        issues.append("oksigen terlarut sangat rendah (< 5 mg/L), nyalakan aerator tambahan")
    elif dissolved_oxygen < 6.4:
        issues.append("oksigen terlarut di bawah optimum, tambah durasi aerasi")

    if temperature > 30.0:
        issues.append("suhu air terlalu panas, beri peneduh dan ganti sebagian air")
    elif temperature < 25.0:
        issues.append("suhu air terlalu dingin, atur pencahayaan/penutup kolam")

    if ph < 6.5:
        issues.append("pH terlalu asam, tambahkan kapur pertanian (CaCO3) secara bertahap")
    elif ph > 9.0:
        issues.append("pH terlalu basa, lakukan pergantian air bertahap")

    if turbidity > 5.0:
        issues.append("air terlalu keruh, lakukan penyaringan dan pengendapan")

    if status == "At Risk":
        if issues:
            return "Kondisi air BERISIKO. Tindakan: " + "; ".join(issues) + "."
        return "Kondisi air BERISIKO menurut model. Lakukan pengecekan ulang seluruh sensor dan pantau kolam setiap 2 jam."
    if issues:
        return "Kondisi air stabil, namun perhatikan: " + "; ".join(issues) + "."
    return "Kondisi air stabil. Lanjutkan pemantauan rutin dan jaga sirkulasi kolam."
