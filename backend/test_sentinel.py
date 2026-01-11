"""
Test script for Sentinel-1 and Sentinel-2 downloaders.
Run from backend directory: python test_sentinel.py
"""
import sys
from datetime import datetime, timedelta

# Add app to path
sys.path.insert(0, ".")

from app.services.copernicus import Sentinel1Downloader, Sentinel2Downloader
from app.services.ship_detector import ShipDetector, convert_to_geo
from app.data.bounding_boxes import AREAS


def test_sentinel_download():
    # Use Dover Strait as test area (busy shipping lane)
    test_area = "algeciras"
    area_data = AREAS[test_area]
    bbox = area_data["bbox"]

    print("=" * 60)
    print(f"Testing area: {test_area}")
    print(f"Description: {area_data['desc']}")
    print(f"Bounding box: {bbox}")
    print("=" * 60)

    # Date range: last 7 days (more likely to have data)
    date_to = datetime.utcnow()
    date_from = date_to - timedelta(days=7)
    date_from_str =  "2025-06-01T00:00:00Z"
    date_to_str ="2026-06-02T23:59:59Z"

    print(f"\nDate range: {date_from_str} to {date_to_str}")

    # Test Sentinel-1
    print("\n" + "-" * 60)
    print("SENTINEL-1 (SAR)")
    print("-" * 60)

    sentinel1 = Sentinel1Downloader()
    try:
        img, acquisition_date = sentinel1.download_area(
            bbox, date_from_str, date_to_str, width=512, height=512
        )

        if img is not None:
            print(f"SUCCESS!")
            print(f"  Image shape: {img.shape}")
            print(f"  Image dtype: {img.dtype}")
            print(f"  Value range: [{img.min():.4f}, {img.max():.4f}]")
            print(f"  Acquisition date: {acquisition_date}")
        else:
            print("FAILED: No image returned (might be no data for this time range)")

    except Exception as e:
        print(f"ERROR: {e}")

    # Test Sentinel-2
    print("\n" + "-" * 60)
    print("SENTINEL-2 (Optical)")
    print("-" * 60)

    sentinel2 = Sentinel2Downloader()
    try:
        img, acquisition_date = sentinel2.download_area(
            bbox, date_from_str, date_to_str, width=512, height=512
        )

        if img is not None:
            print(f"SUCCESS!")
            print(f"  Image shape: {img.shape}")
            print(f"  Image dtype: {img.dtype}")
            print(f"  Value range: [{img.min():.4f}, {img.max():.4f}]")
            print(f"  Acquisition date: {acquisition_date}")
        else:
            print("FAILED: No image returned (might be no data or too cloudy)")

    except Exception as e:
        print(f"ERROR: {e}")

    # Test Ship Detection on downloaded images
    print("\n" + "-" * 60)
    print("SHIP DETECTION (YOLOv8)")
    print("-" * 60)

    detector = ShipDetector()

    # Test on Sentinel-1 image if available
    if "img" in dir() and img is not None:
        print("\nRunning detection on last downloaded image...")
        try:
            detections = detector.detect(img)
            if detections:
                print(f"SUCCESS! Detected {len(detections)} ship(s):")

                # Convert to geographic coordinates
                geo_detections = convert_to_geo(detections, img.shape, bbox)

                for i, (d, geo) in enumerate(zip(detections, geo_detections)):
                    print(f"  [{i+1}] {d.class_name} - confidence: {d.confidence:.2%}")
                    print(f"      pixel bbox: ({d.x1:.1f}, {d.y1:.1f}) -> ({d.x2:.1f}, {d.y2:.1f})")
                    print(f"      geo coords: lat={geo.latitude:.6f}, lon={geo.longitude:.6f}")
            else:
                print("No ships detected in this image")
        except Exception as e:
            print(f"ERROR: {e}")
    else:
        print("No image available for ship detection test")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_sentinel_download()
