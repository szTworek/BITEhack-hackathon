from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.satellite_image import SatelliteImage


class Ship(Base):
    """Model for detected ships."""

    __tablename__ = "ships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    satellite_image_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("satellite_images.id")
    )
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)  # 0.0 - 1.0

    # Relationship to satellite image
    satellite_image: Mapped["SatelliteImage"] = relationship(
        "SatelliteImage", back_populates="ships"
    )
