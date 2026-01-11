from pydantic import BaseModel
from typing import Literal, Optional


class PointProperties(BaseModel):
    id: int


class PointGeometry(BaseModel):
    type: Literal["Point"] = "Point"
    coordinates: list[float]  # [lng, lat] - GeoJSON standard


class PointFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: PointGeometry
    properties: PointProperties


class PointsFeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[PointFeature]


class PointDetail(BaseModel):
    id: int
    lat: float
    lng: float
    image_url: Optional[str] = None


class TimeRangeResponse(BaseModel):
    hours: list[str]  # lista unikalnych godzin w formacie "YYYY-MM-DD HH:00"
    points: dict[str, PointsFeatureCollection]  # godzina -> GeoJSON FeatureCollection
