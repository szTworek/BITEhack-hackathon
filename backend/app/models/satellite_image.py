from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.ship import Ship


class SatelliteImage(Base):
    """Model for satellite image metadata."""

    __tablename__ = "satellite_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    captured_at: Mapped[datetime] = mapped_column(DateTime)  # When satellite took the image
    bbox: Mapped[list] = mapped_column(JSON)  # [min_lon, min_lat, max_lon, max_lat]
    area_name: Mapped[str] = mapped_column(String(100))
    source: Mapped[str] = mapped_column(String(20))  # "sentinel1" or "sentinel2"

    # Relationship to detected ships
    ships: Mapped[list["Ship"]] = relationship("Ship", back_populates="satellite_image")
