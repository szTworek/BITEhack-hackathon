import logging
from datetime import datetime, timedelta

from app.celery_app import celery_app
from app.data.bounding_boxes import AREAS
from app.database import SessionLocal
from app.models import SatelliteImage, Ship
from app.services.copernicus import Sentinel1Downloader, Sentinel2Downloader
from app.services.ship_detector import ShipDetector, convert_to_geo

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.pipeline.run_pipeline")
def run_pipeline(
    self,
    date_from_override: str = None,
    date_to_override: str = None,
    save_date_override: datetime = None,
):
    """
    Main pipeline task - runs at 0:00 and 12:00 UTC.

    Downloads satellite images from Sentinel-1 (SAR) and Sentinel-2 (optical)
    for all configured areas and saves metadata to database.

    Args:
        date_from_override: Start date for data query (ISO format, e.g. "2025-01-01T00:00:00Z")
        date_to_override: End date for data query (ISO format)
        save_date_override: Date to use for database entries (instead of current time)
    """
    logger.info("Starting satellite pipeline...")

    # Date range: use overrides if provided, otherwise last 7 days
    if date_from_override and date_to_override:
        date_from_str = date_from_override
        date_to_str = date_to_override
    else:
        date_to = datetime.utcnow()
        date_from = date_to - timedelta(days=14)
        date_from_str = date_from.strftime("%Y-%m-%dT%H:%M:%SZ")
        date_to_str = date_to.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Date to save in database
    save_date = save_date_override if save_date_override else datetime.utcnow()

    logger.info(f"Date range: {date_from_str} to {date_to_str}")
    if save_date_override:
        logger.info(f"Save date override: {save_date}")

    # Initialize downloaders and detector
    sentinel1 = Sentinel1Downloader()
    sentinel2 = Sentinel2Downloader()
    ship_detector = ShipDetector()

    # Database session
    db = SessionLocal()

    results = {"sentinel1": 0, "sentinel2": 0, "ships_detected": 0, "errors": 0}

    try:
        for area_name, area_data in AREAS.items():
            bbox = area_data["bbox"]
            logger.info(f"Processing area: {area_name} ({area_data['desc']})")

            # Sentinel-1 SAR
            try:
                img, acquisition_date = sentinel1.download_area(
                    bbox, date_from_str, date_to_str
                )
                if img is not None:
                    sat_image = SatelliteImage(
                        captured_at=save_date,
                        bbox=bbox,
                        area_name=area_name,
                        source="sentinel1",
                    )
                    db.add(sat_image)
                    db.commit()
                    results["sentinel1"] += 1
                    logger.info(
                        f"Saved Sentinel-1 image for {area_name}, "
                        f"captured at: {acquisition_date}"
                    )

                    # Run ship detection on Sentinel-1 image
                    detections = ship_detector.detect(img)
                    results["ships_detected"] += len(detections)
                    logger.info(
                        f"Detected {len(detections)} ships in Sentinel-1 {area_name}"
                    )

                    # Convert to geographic coordinates and save to DB
                    if detections:
                        geo_detections = convert_to_geo(detections, img.shape, bbox)
                        for geo in geo_detections:
                            ship = Ship(
                                satellite_image_id=sat_image.id,
                                detected_at=save_date,
                                latitude=geo.latitude,
                                longitude=geo.longitude,
                                confidence=geo.confidence,
                            )
                            db.add(ship)
                            logger.info(
                                f"Ship at lat={geo.latitude:.6f}, lon={geo.longitude:.6f}, "
                                f"conf={geo.confidence:.2%}"
                            )
                        db.commit()
            except Exception as e:
                logger.error(f"Error Sentinel-1 {area_name}: {e}")
                results["errors"] += 1

            # Sentinel-2 Optical
            # try:
            #     img, acquisition_date = sentinel2.download_area(
            #         bbox, date_from_str, date_to_str
            #     )
            #     if img is not None:
            #         sat_image = SatelliteImage(
            #             captured_at=save_date,
            #             bbox=bbox,
            #             area_name=area_name,
            #             source="sentinel2",
            #         )
            #         db.add(sat_image)
            #         db.commit()
            #         results["sentinel2"] += 1
            #         logger.info(
            #             f"Saved Sentinel-2 image for {area_name}, "
            #             f"captured at: {acquisition_date}"
            #         )
            #
            #         # Run ship detection on Sentinel-2 image
            #         detections = ship_detector.detect(img)
            #         results["ships_detected"] += len(detections)
            #         logger.info(
            #             f"Detected {len(detections)} ships in Sentinel-2 {area_name}"
            #         )
            #
            #         # Convert to geographic coordinates and save to DB
            #         if detections:
            #             geo_detections = convert_to_geo(detections, img.shape, bbox)
            #             for geo in geo_detections:
            #                 ship = Ship(
            #                     satellite_image_id=sat_image.id,
            #                     detected_at=save_date,
            #                     latitude=geo.latitude,
            #                     longitude=geo.longitude,
            #                     confidence=geo.confidence,
            #                 )
            #                 db.add(ship)
            #                 logger.info(
            #                     f"Ship at lat={geo.latitude:.6f}, lon={geo.longitude:.6f}, "
            #                     f"conf={geo.confidence:.2%}"
            #                 )
            #             db.commit()
            # except Exception as e:
            #     logger.error(f"Error Sentinel-2 {area_name}: {e}")
            #     results["errors"] += 1

    finally:
        db.close()

    logger.info(f"Pipeline completed: {results}")
    return results
