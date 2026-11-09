"""Model tabel database untuk log analisis kualitas air."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class WaterQualityLog(Base):
    __tablename__ = "water_quality_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    pond_name: Mapped[str] = mapped_column(String(100), default="Kolam Tanpa Nama")
    temperature: Mapped[float] = mapped_column(Float)
    dissolved_oxygen: Mapped[float] = mapped_column(Float)
    ph: Mapped[float] = mapped_column(Float)
    turbidity: Mapped[float] = mapped_column(Float)
    prediction: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float] = mapped_column(Float)
    recommendation: Mapped[str] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
