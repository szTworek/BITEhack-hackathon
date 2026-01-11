# Copernicus satellite services
from .sentinel1 import Sentinel1Downloader
from .sentinel2 import Sentinel2Downloader

__all__ = ["Sentinel1Downloader", "Sentinel2Downloader"]
