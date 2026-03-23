"""
detection/stamp_detector.py
----------------------------
Rubber stamp detector for invoice images.

Strategy (layered — stops at first positive hit):
  1. HSV colour masking  – stamps are typically red, blue, purple, or green ink
  2. Hough circle detection on the masked region
  3. Contour analysis fallback  – for irregularly shaped or faint stamps
  4. Bottom-right quadrant bias  – stamps almost always appear there

Returns a bounding box [x_min, y_min, x_max, y_max] if detected, else None.

Dependencies:
    pip install opencv-python-headless numpy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class StampDetectorConfig:
    # Minimum contour area to consider (pixels²) — avoids tiny noise
    min_contour_area: int = 2000
    # Fraction of image to treat as the "search region" (bottom-right bias)
    search_region_fraction: float = 0.55
    # Hough circle parameters
    hough_dp: float = 1.2
    hough_min_dist: int = 50
    hough_param1: int = 50
    hough_param2: int = 30
    hough_min_radius: int = 20
    hough_max_radius: int = 200
    # Minimum ratio of non-white pixels inside a candidate region
    min_ink_density: float = 0.08


# ---------------------------------------------------------------------------
# Colour ranges in HSV for stamp ink colours
# HSV ranges: H[0-179], S[0-255], V[0-255]
# ---------------------------------------------------------------------------

_STAMP_COLOUR_RANGES: list[tuple[np.ndarray, np.ndarray]] = [
    # Red (wraps around in HSV — two ranges)
    (np.array([0,   80, 50]),  np.array([12, 255, 255])),
    (np.array([165, 80, 50]),  np.array([179, 255, 255])),
    # Blue
    (np.array([100, 60, 40]),  np.array([130, 255, 255])),
    # Purple / violet
    (np.array([130, 40, 40]),  np.array([165, 255, 255])),
    # Green (less common but used in some Tamil Nadu docs)
    (np.array([40,  60, 40]),  np.array([80, 255, 255])),
    # Dark blue (navy)
    (np.array([95,  50, 20]),  np.array([125, 255, 180])),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_bgr(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _get_search_region(
    img: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, int, int]:
    """
    Crop to the bottom-right quadrant (stamps live here ~95% of the time).
    Returns (cropped_img, x_offset, y_offset).
    """
    h, w = img.shape[:2]
    x_off = int(w * (1 - fraction))
    y_off = int(h * (1 - fraction))
    return img[y_off:h, x_off:w], x_off, y_off


def _build_colour_mask(hsv: np.ndarray) -> np.ndarray:
    """Build a binary mask for all stamp-ink colour ranges combined."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in _STAMP_COLOUR_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)
    # Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask


def _ink_density(region: np.ndarray) -> float:
    """Fraction of non-white pixels in a BGR region."""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    non_white = np.sum(gray < 220)
    total = gray.size
    return non_white / max(total, 1)


def _contour_to_bbox(contour: np.ndarray) -> list[int]:
    x, y, w, h = cv2.boundingRect(contour)
    # Expand slightly for bounding box padding
    pad = 10
    return [max(0, x - pad), max(0, y - pad), x + w + pad, y + h + pad]


def _offset_bbox(bbox: list[int], x_off: int, y_off: int) -> list[int]:
    return [bbox[0] + x_off, bbox[1] + y_off, bbox[2] + x_off, bbox[3] + y_off]


def _clamp_bbox(bbox: list[int], w: int, h: int) -> list[int]:
    return [
        max(0, bbox[0]), max(0, bbox[1]),
        min(w, bbox[2]), min(h, bbox[3]),
    ]


# ---------------------------------------------------------------------------
# Detection strategies
# ---------------------------------------------------------------------------

def _detect_by_hough(
    bgr_region: np.ndarray,
    cfg: StampDetectorConfig,
) -> Optional[list[int]]:
    """
    Hough circle detection on the colour-masked region.
    Returns local bbox or None.
    """
    hsv = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)
    mask = _build_colour_mask(hsv)

    if np.sum(mask) < cfg.min_contour_area:
        return None

    gray_mask = cv2.GaussianBlur(mask, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray_mask,
        cv2.HOUGH_GRADIENT,
        dp=cfg.hough_dp,
        minDist=cfg.hough_min_dist,
        param1=cfg.hough_param1,
        param2=cfg.hough_param2,
        minRadius=cfg.hough_min_radius,
        maxRadius=cfg.hough_max_radius,
    )

    if circles is None:
        return None

    # Pick the circle with the largest radius
    circles = np.round(circles[0, :]).astype(int)
    best = max(circles, key=lambda c: c[2])  # c = (cx, cy, r)
    cx, cy, r = int(best[0]), int(best[1]), int(best[2])
    pad = max(10, r // 4)
    return [cx - r - pad, cy - r - pad, cx + r + pad, cy + r + pad]


def _detect_by_contour(
    bgr_region: np.ndarray,
    cfg: StampDetectorConfig,
) -> Optional[list[int]]:
    """
    Contour-based fallback — finds the largest coloured contiguous region.
    Returns local bbox or None.
    """
    hsv = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)
    mask = _build_colour_mask(hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter by area
    valid = [c for c in contours if cv2.contourArea(c) >= cfg.min_contour_area]
    if not valid:
        return None

    # Pick the largest
    largest = max(valid, key=cv2.contourArea)
    bbox = _contour_to_bbox(largest)

    # Verify ink density inside the candidate
    x1, y1, x2, y2 = bbox
    h, w = bgr_region.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    region_crop = bgr_region[y1:y2, x1:x2]
    if _ink_density(region_crop) < cfg.min_ink_density:
        return None

    return bbox


def _detect_full_image(
    bgr: np.ndarray,
    cfg: StampDetectorConfig,
) -> Optional[list[int]]:
    """Full-image fallback when bottom-right search fails."""
    bbox = _detect_by_hough(bgr, cfg)
    if bbox is None:
        bbox = _detect_by_contour(bgr, cfg)
    return bbox


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class StampDetectionResult:
    present: bool
    bbox: Optional[list[int]]    # [x_min, y_min, x_max, y_max] in original image
    confidence: float
    method: str = "none"


def detect_stamp(
    image: Image.Image | np.ndarray,
    cfg: Optional[StampDetectorConfig] = None,
) -> StampDetectionResult:
    """
    Detect a rubber stamp in an invoice image.

    Args:
        image: PIL Image or OpenCV BGR ndarray.
        cfg:   Optional config override.

    Returns:
        StampDetectionResult with present flag, bbox, and confidence.
    """
    if cfg is None:
        cfg = StampDetectorConfig()

    bgr = _pil_to_bgr(image)
    img_h, img_w = bgr.shape[:2]

    # ── Strategy 1: search bottom-right region ────────────────────────────
    region, x_off, y_off = _get_search_region(bgr, cfg.search_region_fraction)

    bbox_local = _detect_by_hough(region, cfg)
    method = "hough_circle"

    if bbox_local is None:
        bbox_local = _detect_by_contour(region, cfg)
        method = "contour"

    if bbox_local is not None:
        bbox = _offset_bbox(bbox_local, x_off, y_off)
        bbox = _clamp_bbox(bbox, img_w, img_h)
        logger.debug("Stamp detected via %s in search region: %s", method, bbox)
        return StampDetectionResult(present=True, bbox=bbox, confidence=0.88, method=method)

    # ── Strategy 2: full image search ────────────────────────────────────
    bbox_full = _detect_full_image(bgr, cfg)
    if bbox_full is not None:
        bbox_full = _clamp_bbox(bbox_full, img_w, img_h)
        logger.debug("Stamp detected via full-image search: %s", bbox_full)
        return StampDetectionResult(
            present=True, bbox=bbox_full, confidence=0.70, method="full_image"
        )

    logger.debug("No stamp detected.")
    return StampDetectionResult(present=False, bbox=None, confidence=0.80, method="none")
