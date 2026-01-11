"""
Sentinel-1 SAR (Synthetic Aperture Radar) image downloader.
Radar images work regardless of cloud cover and lighting conditions.
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

# Evalscript for Sentinel-1 Dual-Pol RGB visualization
EVALSCRIPT_S1_DUAL_POL = """
function setup() {
    return {
        input: ["VV", "VH"],
        output: { bands: 3, sampleType: "FLOAT32" }
    };
}

function evaluatePixel(sample) {
    // Convert to decibel scale (dB)
    let vv = 10 * Math.log10(Math.max(0.0001, sample.VV));
    let vh = 10 * Math.log10(Math.max(0.0001, sample.VH));

    // Scale dB to 0-1 range (assuming -25dB to 0dB range)
    let r = (vv + 25) / 25;
    let g = (vh + 25) / 25;
    let b = (vv - vh + 10) / 20;  // Polarization ratio highlighting metal

    return [r, g, b];
}
"""


class Sentinel1Downloader:
    """Downloads SAR images from Sentinel-1 via Copernicus API."""

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

    def _enhance_sar_image(self, img: np.ndarray) -> np.ndarray:
        """Enhance SAR image for better ship detection."""
        # Speckle noise reduction with gentle Gaussian blur
        enhanced = np.zeros_like(img)
        for i in range(3):
            enhanced[:, :, i] = gaussian_filter(img[:, :, i], sigma=0.5)

        # Contrast stretching (2nd and 98th percentile)
        for i in range(3):
            vmin, vmax = np.percentile(enhanced[:, :, i], (2, 98))
            enhanced[:, :, i] = np.clip(
                (enhanced[:, :, i] - vmin) / (vmax - vmin + 1e-5), 0, 1
            )

        return enhanced

    def download_area(
        self,
        bbox: list[float],
        date_from: str,
        date_to: str,
        width: int = 1500,
        height: int = 1500,
    ) -> tuple[Optional[np.ndarray], Optional[datetime]]:
        """
        Download SAR image for given bounding box and time range.

        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            date_from: ISO format datetime string (e.g., "2025-06-01T00:00:00Z")
            date_to: ISO format datetime string
            width: Output image width in pixels
            height: Output image height in pixels

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
                        "type": "sentinel-1-grd",
                        "dataFilter": {
                            "timeRange": {"from": date_from, "to": date_to},
                            "acquisitionMode": "IW",
                            "polarization": "DV",  # Dual Vertical (VV+VH)
                        },
                        "processing": {"backscatterCoeff": "SIGMA0_ELLIPSOID"},
                    }
                ],
            },
            "output": {
                "width": width,
                "height": height,
            },
            "evalscript": EVALSCRIPT_S1_DUAL_POL,
        }

        try:
            logger.info(f"Downloading Sentinel-1 SAR image for bbox {bbox}")
            response = requests.post(PROCESS_URL, headers=headers, json=payload)
            response.raise_for_status()

            # Read TIFF directly from response
            img = tiff.imread(io.BytesIO(response.content))

            if img is None:
                logger.error("No image found in response")
                return None, None

            # Enhance for ship detection
            img = self._enhance_sar_image(img)

            # Use current time as approximate acquisition date (metadata not available in simple mode)
            acquisition_date = datetime.utcnow()

            logger.info(
                f"Successfully downloaded Sentinel-1 image, shape: {img.shape}"
            )
            return img, acquisition_date

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error downloading Sentinel-1: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Error downloading Sentinel-1: {e}")
            return None, None
