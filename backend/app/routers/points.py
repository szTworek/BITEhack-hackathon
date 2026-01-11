from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Ship
from app.schemas.points import (
    PointsFeatureCollection,
    PointFeature,
    PointGeometry,
    PointProperties,
    PointDetail,
    TimeRangeResponse,
)

router = APIRouter(prefix="/points", tags=["points"])


def _convert_to_geojson(points: list[dict]) -> PointsFeatureCollection:
    """Konwertuje liste punktow do formatu GeoJSON FeatureCollection."""
    features = [
        PointFeature(
            geometry=PointGeometry(coordinates=[p["lng"], p["lat"]]),
            properties=PointProperties(id=p["id"]),
        )
        for p in points
    ]
    return PointsFeatureCollection(features=features)


@router.get("", response_model=PointsFeatureCollection)
async def get_points(db: Session = Depends(get_db)):
    """
    Zwraca wszystkie punkty (statki) w formacie GeoJSON FeatureCollection.

    Odpowiedz jest zoptymalizowana do bezposredniego uzycia z Mapbox GL JS.
    """
    ships = db.query(Ship).all()
    points = [{"id": s.id, "lat": s.latitude, "lng": s.longitude} for s in ships]
    return _convert_to_geojson(points)


@router.get("/timerange", response_model=TimeRangeResponse)
async def get_points_by_timerange(
    start_date: str = Query(..., description="Data poczatkowa w formacie YYYY-MM-DD"),
    end_date: str = Query(..., description="Data koncowa w formacie YYYY-MM-DD"),
    db: Session = Depends(get_db),
):
    """
    Zwraca punkty (statki) pogrupowane po dniach dla podanego przedzialu dat.
    Dane pobierane z bazy danych.
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)  # do konca dnia
    except ValueError:
        raise HTTPException(status_code=400, detail="Nieprawidlowy format daty. Uzyj YYYY-MM-DD")

    if end <= start:
        raise HTTPException(status_code=400, detail="Data koncowa musi byc pozniejsza niz poczatkowa")

    # Pobierz statki z zakresu dat
    ships = db.query(Ship).filter(
        Ship.detected_at >= start,
        Ship.detected_at < end
    ).all()

    # Grupuj po dniu
    points_by_day = {}
    for ship in ships:
        day = ship.detected_at.strftime("%Y-%m-%d")
        if day not in points_by_day:
            points_by_day[day] = []
        points_by_day[day].append({
            "id": ship.id,
            "lat": ship.latitude,
            "lng": ship.longitude
        })

    # Konwertuj do GeoJSON
    days = sorted(points_by_day.keys())
    result = {day: _convert_to_geojson(points_by_day[day]) for day in days}

    return TimeRangeResponse(hours=days, points=result)


@router.get("/{point_id}", response_model=PointDetail)
async def get_point(point_id: int, db: Session = Depends(get_db)):
    """
    Zwraca szczegoly pojedynczego punktu (statku).
    """
    ship = db.query(Ship).filter(Ship.id == point_id).first()
    if not ship:
        raise HTTPException(status_code=404, detail="Punkt nie znaleziony")
    return PointDetail(
        id=ship.id,
        lat=ship.latitude,
        lng=ship.longitude,
        image_url=None
    )
