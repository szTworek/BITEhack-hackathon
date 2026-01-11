"""
Sentinel-2 optical image downloader.
High-resolution multispectral imagery (visible and infrared).
"""
import io
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import requests
import tifffile as tiff
from scipy.ndimage import gaussian_filter

from app.config import settings

logger = logging.getLogger(__name__)

AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# Evalscript for Sentinel-2 Enhanced True Color
EVALSCRIPT_ENHANCED_COLOR = """
function setup() {
    return {
        input: ["B04", "B03", "B02"],
        output: { bands: 3, sampleType: "FLOAT32" }
    };
}

function evaluatePixel(sample) {
    // Standard Sentinel-2 normalization (divide by 10000)
    // with strong gain for water visibility
    let gain = 2.5;
    return [
        sample.B04 * gain,
        sample.B03 * gain,
        sample.B02 * gain
    ];
}
"""


class Sentinel2Downloader:
    """Downloads optical images from Sentinel-2 via Copernicus API."""

    def __init__(self):
        self.client_id = settings.copernicus_client_id
        self.client_secret = settings.copernicus_client_secret
        self.token: Optional[str] = None

    def get_token(self) -> str:
        """Authenticate with Copernicus API and get access token."""
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        response = requests.post(AUTH_URL, data=data)
        response.raise_for_status()
        self.token = response.json()["access_token"]
        return self.token

    def _enhance_optical_image(self, img: np.ndarray) -> np.ndarray:
        """Enhance optical image for ship detection."""
        # Manual scaling (0.0 - 0.3 is typical range for water and ships)
        # Everything above 0.4 (clouds) will be white, but water remains visible
        enhanced = np.clip(img / 0.4, 0, 1)

        # Gentle sharpening
        for i in range(3):
            blurred = gaussian_filter(enhanced[:, :, i], sigma=0.5)
            enhanced[:, :, i] = np.clip(
                enhanced[:, :, i] + 0.8 * (enhanced[:, :, i] - blurred), 0, 1
            )

        return enhanced

    def download_area(
        self,
        bbox: list[float],
        date_from: str,
        date_to: str,
        width: int = 1500,
        height: int = 1500,
        max_cloud_coverage: int = 50,
    ) -> tuple[Optional[np.ndarray], Optional[datetime]]:
        """
        Download optical image for given bounding box and time range.

        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            date_from: ISO format datetime string (e.g., "2025-06-01T00:00:00Z")
            date_to: ISO format datetime string
            width: Output image width in pixels
            height: Output image height in pixels
            max_cloud_coverage: Maximum cloud coverage percentage (0-100)

        Returns:
            Tuple of (numpy array with shape (height, width, 3), acquisition datetime)
            or (None, None) on failure
        """
        if not self.token:
            self.get_token()

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "image/tiff",
        }

        payload = {
            "input": {
                "bounds": {"bbox": bbox},
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "maxCloudCoverage": max_cloud_coverage,
                            "timeRange": {"from": date_from, "to": date_to},
                        },
                    }
                ],
            },
            "output": {
                "width": width,
                "height": height,
            },
            "evalscript": EVALSCRIPT_ENHANCED_COLOR,
        }

        try:
            logger.info(f"Downloading Sentinel-2 optical image for bbox {bbox}")
            response = requests.post(PROCESS_URL, headers=headers, json=payload)
            response.raise_for_status()

            # Read TIFF directly from response
            img = tiff.imread(io.BytesIO(response.content))
            img = img.astype(np.float32)

            if img is None:
                logger.error("No image found in response")
                return None, None

            # Enhance for ship detection
            img = self._enhance_optical_image(img)

            # Use current time as approximate acquisition date (metadata not available in simple mode)
            acquisition_date = datetime.utcnow()

            logger.info(
                f"Successfully downloaded Sentinel-2 image, shape: {img.shape}"
            )
            return img, acquisition_date

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error downloading Sentinel-2: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Error downloading Sentinel-2: {e}")
            return None, None
