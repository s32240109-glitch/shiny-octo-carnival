"""AquaSense - FastAPI app untuk analisis kualitas air budidaya ikan nila.

Soal 2A (website + Jinja2/CSS/JS), Soal 2B (CRUD database), dan integrasi
model SVM ke endpoint prediksi (Soal 3).
"""

from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import models_db
from database import Base, engine, get_db
from ml_service import (
    FEATURE_META,
    MODEL_INFO,
    build_recommendation,
    predict_water_quality,
)

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="AquaSense", description="Analisis Kualitas Air Budidaya Ikan Nila dengan SVM")
Base.metadata.create_all(bind=engine)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Rentang nilai pada dataset latih, dipakai untuk memberi peringatan out-of-distribution.
TRAIN_RANGES = {
    "temperature": (26.50, 28.35, "°C"),
    "dissolved_oxygen": (6.34, 7.89, "mg/L"),
    "ph": (7.34, 7.98, "pH"),
    "turbidity": (2.79, 3.98, "NTU"),
}
FIELD_LABELS = {
    "temperature": "Suhu",
    "dissolved_oxygen": "DO",
    "ph": "pH",
    "turbidity": "Turbidity",
}


def base_context(request: Request) -> dict:
    return {
        "request": request,
        "model_info": MODEL_INFO,
        "feature_meta": FEATURE_META,
        "year": 2026,
    }


def parse_number(raw: str, label: str, low: float, high: float) -> float:
    try:
        value = float(str(raw).replace(",", ".").strip())
    except (TypeError, ValueError):
        raise ValueError(f"{label} harus berupa angka.")
    if value < low or value > high:
        raise ValueError(f"{label} harus berada di antara {low:g} dan {high:g}.")
    return value


def out_of_range_notes(values: dict) -> list[str]:
    notes = []
    for key, (low, high, unit) in TRAIN_RANGES.items():
        value = values[key]
        if value < low or value > high:
            notes.append(
                f"{FIELD_LABELS[key]} = {value:g} {unit} berada di luar rentang data latih "
                f"({low:g} – {high:g} {unit})."
            )
    return notes


def get_stats(db: Session) -> dict:
    total = db.scalar(select(func.count(models_db.WaterQualityLog.id))) or 0
    at_risk = db.scalar(
        select(func.count(models_db.WaterQualityLog.id)).where(
            models_db.WaterQualityLog.prediction == "At Risk"
        )
    ) or 0
    return {
        "total": total,
        "at_risk": at_risk,
        "stable": total - at_risk,
    }


@app.get("/")
def home(request: Request, db: Annotated[Session, Depends(get_db)]):
    context = base_context(request)
    context.update(
        {
            "active": "home",
            "stats": get_stats(db),
            "recent_logs": db.scalars(
                select(models_db.WaterQualityLog)
                .order_by(models_db.WaterQualityLog.created_at.desc())
                .limit(4)
            ).all(),
        }
    )
    return templates.TemplateResponse(request, "index.html", context)


@app.get("/predict")
def predict_page(request: Request):
    context = base_context(request)
    context.update(
        {
            "active": "predict",
            "values": {item["key"]: item["default"] for item in FEATURE_META},
            "pond_name": "",
            "result": None,
            "error": None,
            "warnings": [],
        }
    )
    return templates.TemplateResponse(request, "predict.html", context)


@app.post("/predict")
def predict_submit(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    pond_name: Annotated[str, Form()] = "",
    temperature: Annotated[str, Form()] = "",
    dissolved_oxygen: Annotated[str, Form()] = "",
    ph: Annotated[str, Form()] = "",
    turbidity: Annotated[str, Form()] = "",
):
    context = base_context(request)
    context["active"] = "predict"
    context["pond_name"] = pond_name.strip()
    context["values"] = {
        "temperature": temperature,
        "dissolved_oxygen": dissolved_oxygen,
        "ph": ph,
        "turbidity": turbidity,
    }
    context["result"] = None

    try:
        values = {
            "temperature": parse_number(temperature, "Suhu air", 0, 50),
            "dissolved_oxygen": parse_number(dissolved_oxygen, "Oksigen terlarut", 0, 25),
            "ph": parse_number(ph, "pH", 0, 14),
            "turbidity": parse_number(turbidity, "Turbidity", 0, 1000),
        }
    except ValueError as exc:
        context["error"] = str(exc)
        context["warnings"] = []
        return templates.TemplateResponse(request, "predict.html", context, status_code=400)

    result = predict_water_quality(**values)
    recommendation = build_recommendation(result["status"], **values)

    log = models_db.WaterQualityLog(
        pond_name=context["pond_name"] or "Kolam Tanpa Nama",
        temperature=values["temperature"],
        dissolved_oxygen=values["dissolved_oxygen"],
        ph=values["ph"],
        turbidity=values["turbidity"],
        prediction=result["status"],
        confidence=result["confidence"],
        recommendation=recommendation,
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    context.update(
        {
            "result": result,
            "recommendation": recommendation,
            "warnings": out_of_range_notes(values),
            "saved_log": log,
            "values": values,
        }
    )
    return templates.TemplateResponse(request, "predict.html", context)


@app.get("/history")
def history(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    q: str = Query("", max_length=100),
    status: str = Query("", max_length=20),
    deleted: int = 0,
):
    stmt = select(models_db.WaterQualityLog)
    if q:
        stmt = stmt.where(models_db.WaterQualityLog.pond_name.ilike(f"%{q}%"))
    if status in ("At Risk", "Stable"):
        stmt = stmt.where(models_db.WaterQualityLog.prediction == status)
    logs = db.scalars(stmt.order_by(models_db.WaterQualityLog.created_at.desc())).all()

    context = base_context(request)
    context.update(
        {
            "active": "history",
            "logs": logs,
            "stats": get_stats(db),
            "q": q,
            "status_filter": status,
            "deleted": bool(deleted),
        }
    )
    return templates.TemplateResponse(request, "history.html", context)


@app.post("/history/{log_id}/delete")
def delete_log_form(log_id: int, db: Annotated[Session, Depends(get_db)]):
    log = db.get(models_db.WaterQualityLog, log_id)
    if log:
        db.delete(log)
        db.commit()
    return RedirectResponse("/history?deleted=1", status_code=303)


@app.delete("/api/logs/{log_id}")
def delete_log_api(log_id: int, db: Annotated[Session, Depends(get_db)]):
    log = db.get(models_db.WaterQualityLog, log_id)
    if not log:
        raise HTTPException(status_code=404, detail="Data tidak ditemukan.")
    db.delete(log)
    db.commit()
    return {"ok": True, "deleted_id": log_id, "stats": get_stats(db)}


@app.get("/api/logs")
def list_logs_api(db: Annotated[Session, Depends(get_db)]):
    logs = db.scalars(
        select(models_db.WaterQualityLog).order_by(models_db.WaterQualityLog.created_at.desc())
    ).all()
    return {
        "count": len(logs),
        "stats": get_stats(db),
        "data": [
            {
                "id": log.id,
                "pond_name": log.pond_name,
                "temperature": log.temperature,
                "dissolved_oxygen": log.dissolved_oxygen,
                "ph": log.ph,
                "turbidity": log.turbidity,
                "prediction": log.prediction,
                "confidence": round(log.confidence * 100, 1),
                "recommendation": log.recommendation,
                "created_at": log.created_at.isoformat(timespec="seconds"),
            }
            for log in logs
        ],
    }


@app.post("/api/predict")
def predict_api(payload: dict):
    try:
        values = {
            "temperature": parse_number(payload.get("temperature"), "Suhu air", 0, 50),
            "dissolved_oxygen": parse_number(payload.get("dissolved_oxygen"), "Oksigen terlarut", 0, 25),
            "ph": parse_number(payload.get("ph"), "pH", 0, 14),
            "turbidity": parse_number(payload.get("turbidity"), "Turbidity", 0, 1000),
        }
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    result = predict_water_quality(**values)
    result["recommendation"] = build_recommendation(result["status"], **values)
    result["warnings"] = out_of_range_notes(values)
    return {"ok": True, **result}


@app.get("/contact")
def contact(request: Request):
    context = base_context(request)
    context["active"] = "contact"
    context["members"] = [
        {
            "name": "Refi",
            "role": "Lead AI & Machine Learning",
            "desc": "Preprocessing dataset, training & tuning SVM, evaluasi model, dan ekspor file .pkl.",
            "tag": "Soal 1",
        },
        {
            "name": "Bernhard",
            "role": "Backend & Database Engineer",
            "desc": "Membangun FastAPI, skema database, endpoint CRUD, dan integrasi model ke /predict.",
            "tag": "Soal 2B",
        },
        {
            "name": "Devin",
            "role": "Frontend Engineer & UI/UX",
            "desc": "Merancang template Jinja2, CSS, JavaScript, halaman Home, Contact, form prediksi, dan tabel riwayat.",
            "tag": "Soal 2A",
        },
        {
            "name": "Vino",
            "role": "Technical Writer & Lead Laporan",
            "desc": "Menyusun laporan PDF, identifikasi masalah, pemilihan dataset, dan dokumentasi teknis.",
            "tag": "Soal 4",
        },
        {
            "name": "Lutfi",
            "role": "QA, System Integrator & Video Producer",
            "desc": "End-to-end testing, skenario pengujian, dan produksi video demonstrasi aplikasi.",
            "tag": "Soal 4C",
        },
    ]
    return templates.TemplateResponse(request, "contact.html", context)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_INFO["algorithm"],
        "features": MODEL_INFO["features"],
    }
