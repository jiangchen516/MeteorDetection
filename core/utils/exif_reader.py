"""EXIF 元数据读取."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ImageMetadata:
    """图像元数据."""
    filepath: str
    width: int = 0
    height: int = 0
    datetime_taken: datetime | None = None
    exposure_time: str = ""
    iso: int = 0
    focal_length: float = 0.0
    aperture: float = 0.0
    camera_make: str = ""
    camera_model: str = ""
    gps_lat: float | None = None
    gps_lon: float | None = None
    gps_alt: float | None = None


def read_exif(path: str | Path) -> ImageMetadata:
    """读取图像 EXIF 信息.

    Args:
        path: 图像文件路径.

    Returns:
        ImageMetadata.
    """
    path = Path(path)
    meta = ImageMetadata(filepath=str(path))

    try:
        from PIL import Image, ExifTags
        from PIL.ExifTags import TAGS, GPSTAGS

        img = Image.open(str(path))
        meta.width, meta.height = img.size

        exif_data = img.getexif()
        if not exif_data:
            return meta

        exif_dict = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            exif_dict[tag_name] = value

        # 拍摄时间
        date_str = exif_dict.get("DateTimeOriginal") or exif_dict.get("DateTime")
        if date_str:
            try:
                meta.datetime_taken = datetime.strptime(str(date_str), "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass

        # 曝光参数
        meta.exposure_time = str(exif_dict.get("ExposureTime", ""))
        meta.iso = int(exif_dict.get("ISOSpeedRatings", 0) or 0)
        meta.focal_length = float(exif_dict.get("FocalLength", 0) or 0)
        meta.aperture = float(exif_dict.get("FNumber", 0) or 0)
        meta.camera_make = str(exif_dict.get("Make", ""))
        meta.camera_model = str(exif_dict.get("Model", ""))

        # GPS
        gps_info = exif_dict.get("GPSInfo")
        if gps_info:
            gps_dict = {}
            for key in gps_info.keys():
                decode = GPSTAGS.get(key, key)
                gps_dict[decode] = gps_info[key]

            if "GPSLatitude" in gps_dict and "GPSLongitude" in gps_dict:
                meta.gps_lat = _dms_to_dd(gps_dict["GPSLatitude"],
                                          gps_dict.get("GPSLatitudeRef", "N"))
                meta.gps_lon = _dms_to_dd(gps_dict["GPSLongitude"],
                                          gps_dict.get("GPSLongitudeRef", "E"))
                if "GPSAltitude" in gps_dict:
                    meta.gps_alt = float(gps_dict["GPSAltitude"])

        logger.debug(f"EXIF read: {path.name} ({meta.width}x{meta.height}, "
                     f"ISO={meta.iso}, {meta.camera_make} {meta.camera_model})")

    except Exception as e:
        logger.warning(f"Failed to read EXIF from {path}: {e}")

    return meta


def _dms_to_dd(dms: tuple, ref: str) -> float:
    """度分秒转十进制度."""
    degrees = float(dms[0])
    minutes = float(dms[1])
    seconds = float(dms[2])
    dd = degrees + minutes / 60 + seconds / 3600
    if ref in ("S", "W"):
        dd = -dd
    return dd
