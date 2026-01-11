import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class ShipDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str


@dataclass
class GeoShipDetection:
    latitude: float
    longitude: float
    confidence: float
    class_name: str
    pixel_bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) for debugging


class ShipDetector:
    def __init__(
        self, model_path: str = None, confidence_threshold: float = 0.15
    ):
        if model_path is None:
            model_path = Path(__file__).parent / "models" / "final.pt"
        self.model = YOLO(str(model_path))
        self.confidence_threshold = confidence_threshold
        logger.info(f"ShipDetector initialized with model: {model_path}")

    def detect(self, image: np.ndarray) -> list[ShipDetection]:
        """
        Detect ships in a satellite image.

        Args:
            image: numpy array (H, W, 3) - float32 in range 0-1 or uint8 in range 0-255

        Returns:
            List of detected ships with bounding boxes
        """
        # Convert from float 0-1 to uint8 0-255 if needed (required by YOLO)
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).clip(0, 255).astype(np.uint8)

        # Convert to grayscale and back to 3-channel for better detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        results = self.model(gray_3ch, conf=self.confidence_threshold, verbose=False)
        detections = []

        r = results[0]
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                name = self.model.names[cls]

                detections.append(
                    ShipDetection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=conf,
                        class_name=name,
                    )
                )

        logger.info(f"Detected {len(detections)} ships")
        return detections


def convert_to_geo(
    detections: list[ShipDetection],
    image_shape: tuple,
    bbox: list[float],
) -> list[GeoShipDetection]:
    """
    Convert pixel-based detections to geographic coordinates.

    Args:
        detections: List of ShipDetection with pixel coordinates
        image_shape: (height, width, channels) from img.shape
        bbox: [min_lon, min_lat, max_lon, max_lat] - area bounding box

    Returns:
        List of GeoShipDetection with lat/lon coordinates
    """
    from app.services.coord_converter import CoordConverter

    height, width = image_shape[:2]
    min_lon, min_lat, max_lon, max_lat = bbox

    # CoordConverter expects corners format
    corners = {
        "top_left": [min_lon, max_lat],      # top-left: min_lon, max_lat
        "bottom_right": [max_lon, min_lat],  # bottom-right: max_lon, min_lat
    }

    converter = CoordConverter(width, height, corners)
    geo_detections = []

    for det in detections:
        # Get center of bounding box
        center_x = (det.x1 + det.x2) / 2
        center_y = (det.y1 + det.y2) / 2

        lon, lat = converter.pixel_to_geo(center_x, center_y)

        geo_detections.append(
            GeoShipDetection(
                latitude=lat,
                longitude=lon,
                confidence=det.confidence,
                class_name=det.class_name,
                pixel_bbox=(det.x1, det.y1, det.x2, det.y2),
            )
        )

    return geo_detections
