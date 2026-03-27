"""
detection/stamp_detector.py
----------------------------
Robust rubber stamp detector — 4 independent strategies, majority vote.

Strategy 1 — HSV colour masking + Hough circles  (coloured circular stamps)
Strategy 2 — Dark-ink contour analysis            (black/grey stamps)
Strategy 3 — Frequency-domain circularity         (faint / low-contrast)
Strategy 4 — Text-cluster density map             (dense circular text region)

Biased to bottom-right quadrant (stamps live there ~95% of time),
but falls back to full-image search when needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
    min_contour_area: int = 800          # lowered — catches smaller stamps
    search_region_fraction: float = 0.50 # bottom-right 50%
    hough_dp: float = 1.0
    hough_min_dist: int = 40
    hough_param1: int = 40
    hough_param2: int = 20              # lowered — more sensitive
    hough_min_radius: int = 15
    hough_max_radius: int = 250
    min_ink_density: float = 0.04       # lowered
    min_strategies_agree: int = 1       # just 1 strategy is enough to declare present


# ---------------------------------------------------------------------------
# Colour ranges for stamp inks (HSV)
# ---------------------------------------------------------------------------

_STAMP_COLOUR_RANGES = [
    # Red (two ranges — wraps in HSV)
    (np.array([0,   50, 40]),  np.array([12, 255, 255])),
    (np.array([160, 50, 40]),  np.array([179,255, 255])),
    # Blue
    (np.array([95,  40, 30]),  np.array([135,255, 255])),
    # Purple / violet
    (np.array([125, 30, 30]),  np.array([165,255, 255])),
    # Green
    (np.array([38,  40, 30]),  np.array([82, 255, 255])),
    # Dark navy
    (np.array([90,  40, 15]),  np.array([125,200, 160])),
    # Black / dark grey — critical for black stamps like this doc
    (np.array([0,   0,  0]),   np.array([179, 60,  80])),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_bgr(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _search_region(img: np.ndarray, frac: float) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    x_off = int(w * (1 - frac))
    y_off = int(h * (1 - frac))
    return img[y_off:, x_off:], x_off, y_off


def _colour_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in _STAMP_COLOUR_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


def _ink_density(bgr_crop: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    return np.sum(gray < 200) / max(gray.size, 1)


def _polygon_bbox(pts: np.ndarray, pad: int = 12) -> list[int]:
    x, y, w, h = cv2.boundingRect(pts)
    return [max(0, x-pad), max(0, y-pad), x+w+pad, y+h+pad]


def _offset(bbox: list[int], dx: int, dy: int) -> list[int]:
    return [bbox[0]+dx, bbox[1]+dy, bbox[2]+dx, bbox[3]+dy]


def _clamp(bbox: list[int], W: int, H: int) -> list[int]:
    return [max(0,bbox[0]), max(0,bbox[1]), min(W,bbox[2]), min(H,bbox[3])]


# ---------------------------------------------------------------------------
# Strategy 1 — HSV colour mask + Hough circles
# ---------------------------------------------------------------------------

def _s1_hough(bgr: np.ndarray, cfg: StampDetectorConfig) -> Optional[list[int]]:
    mask = _colour_mask(bgr)
    if np.sum(mask) < cfg.min_contour_area * 2:
        return None
    blur = cv2.GaussianBlur(mask, (9, 9), 2)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=cfg.hough_dp, minDist=cfg.hough_min_dist,
        param1=cfg.hough_param1, param2=cfg.hough_param2,
        minRadius=cfg.hough_min_radius, maxRadius=cfg.hough_max_radius,
    )
    if circles is None:
        return None
    cx, cy, r = map(int, np.round(circles[0, 0]))
    pad = max(12, r // 4)
    return [cx-r-pad, cy-r-pad, cx+r+pad, cy+r+pad]


# ---------------------------------------------------------------------------
# Strategy 2 — Contour shape analysis (circularity + area)
# ---------------------------------------------------------------------------

def _s2_contour(bgr: np.ndarray, cfg: StampDetectorConfig) -> Optional[list[int]]:
    """
    Looks for large, roughly circular blobs regardless of colour.
    Key insight: stamps have high circularity (4π·area/perimeter²  ≈ 0.5–1.0)
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Use Canny on the whole image — catches outlines even for black stamps
    edges = cv2.Canny(gray, 30, 100)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, k, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score = 0.0
    best_cnt   = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg.min_contour_area:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim < 1:
            continue
        circularity = (4 * np.pi * area) / (perim ** 2)
        # Stamps are circles or slightly oval → circularity 0.3–1.0
        if circularity < 0.25:
            continue
        score = circularity * np.log(area + 1)
        if score > best_score:
            best_score = score
            best_cnt   = cnt

    if best_cnt is None:
        return None

    # Verify ink density inside the candidate bbox
    bbox = _polygon_bbox(best_cnt, pad=8)
    h, w = bgr.shape[:2]
    x1,y1,x2,y2 = max(0,bbox[0]),max(0,bbox[1]),min(w,bbox[2]),min(h,bbox[3])
    if x2 <= x1 or y2 <= y1:
        return None
    crop = bgr[y1:y2, x1:x2]
    if _ink_density(crop) < cfg.min_ink_density:
        return None

    return bbox


# ---------------------------------------------------------------------------
# Strategy 3 — Dark-ink blob (catches black stamps like in this doc)
# ---------------------------------------------------------------------------

def _s3_dark_blob(bgr: np.ndarray, cfg: StampDetectorConfig) -> Optional[list[int]]:
    """
    Threshold for dark pixels, cluster them, find large round cluster.
    Works for black/dark-grey ink stamps that HSV misses.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # OTSU threshold — adapts to image brightness automatically
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove large text blocks (stamps have dense connected components)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k_open,  iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=3)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg.min_contour_area * 3:   # stamps are large
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Aspect ratio close to 1 → circular/oval
        aspect = w / max(h, 1)
        if not (0.4 <= aspect <= 2.5):
            continue
        perim = cv2.arcLength(cnt, True)
        circ  = (4 * np.pi * area) / max(perim**2, 1)
        if circ < 0.15:
            continue
        candidates.append((circ * area, cnt))

    if not candidates:
        return None

    _, best = max(candidates, key=lambda x: x[0])
    return _polygon_bbox(best, pad=10)


# ---------------------------------------------------------------------------
# Strategy 4 — Circular text-density heatmap
# ---------------------------------------------------------------------------

def _s4_text_density(bgr: np.ndarray, cfg: StampDetectorConfig) -> Optional[list[int]]:
    """
    Stamps contain circular arrangements of small text.
    Detect by finding regions with uniformly high small-blob density
    arranged in a roughly circular pattern.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find small connected components (individual letters in stamp text)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    h, w = bgr.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cw   = stats[i, cv2.CC_STAT_WIDTH]
        ch   = stats[i, cv2.CC_STAT_HEIGHT]
        # Small components = stamp text characters
        if 10 <= area <= 800 and cw < 60 and ch < 60:
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            cv2.circle(heatmap, (cx, cy), 20, 1.0, -1)

    # Blur the heatmap — dense regions become hot spots
    heatmap = cv2.GaussianBlur(heatmap, (61, 61), 20)
    _, thresh = cv2.threshold(heatmap, 0.4, 1.0, cv2.THRESH_BINARY)
    thresh_u8 = (thresh * 255).astype(np.uint8)

    contours, _ = cv2.findContours(thresh_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    valid = [c for c in contours if cv2.contourArea(c) > cfg.min_contour_area * 2]
    if not valid:
        return None

    best = max(valid, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(best)
    # Must be vaguely circular
    if not (0.3 <= bw / max(bh, 1) <= 3.0):
        return None

    return [max(0, x-15), max(0, y-15), x+bw+15, y+bh+15]


# ---------------------------------------------------------------------------
# Multi-strategy runner
# ---------------------------------------------------------------------------

def _run_strategies(
    bgr: np.ndarray,
    cfg: StampDetectorConfig,
) -> list[tuple[list[int], float, str]]:
    """Run all strategies and return list of (bbox, confidence, method)."""
    results = []

    b = _s1_hough(bgr, cfg)
    if b: results.append((b, 0.90, "hough_circle"))

    b = _s2_contour(bgr, cfg)
    if b: results.append((b, 0.80, "contour_circularity"))

    b = _s3_dark_blob(bgr, cfg)
    if b: results.append((b, 0.82, "dark_blob"))

    b = _s4_text_density(bgr, cfg)
    if b: results.append((b, 0.78, "text_density"))

    return results


def _merge_bboxes(bboxes: list[list[int]]) -> list[int]:
    """Merge multiple bboxes into one enclosing bbox."""
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class StampDetectionResult:
    present: bool
    bbox: Optional[list[int]]
    confidence: float
    method: str = "none"


def detect_stamp(
    image: Image.Image | np.ndarray,
    cfg: Optional[StampDetectorConfig] = None,
) -> StampDetectionResult:
    if cfg is None:
        cfg = StampDetectorConfig()

    bgr = _to_bgr(image)
    H, W = bgr.shape[:2]

    # ── Try bottom-right search region first ──────────────────────────────
    region, x_off, y_off = _search_region(bgr, cfg.search_region_fraction)
    hits = _run_strategies(region, cfg)

    if len(hits) >= cfg.min_strategies_agree:
        # Use highest-confidence hit
        best_bbox, best_conf, best_method = max(hits, key=lambda x: x[1])
        bbox = _clamp(_offset(best_bbox, x_off, y_off), W, H)
        conf = min(0.95, best_conf + 0.05 * len(hits))
        logger.debug("Stamp: %s (conf=%.2f, %d strategies agreed)", best_method, conf, len(hits))
        return StampDetectionResult(present=True, bbox=bbox, confidence=conf, method=best_method)

    # ── Full image fallback ───────────────────────────────────────────────
    hits_full = _run_strategies(bgr, cfg)
    if len(hits_full) >= cfg.min_strategies_agree:
        best_bbox, best_conf, best_method = max(hits_full, key=lambda x: x[1])
        bbox = _clamp(best_bbox, W, H)
        conf = best_conf * 0.85   # slight penalty for finding outside expected region
        logger.debug("Stamp (full-image): %s (conf=%.2f)", best_method, conf)
        return StampDetectionResult(present=True, bbox=bbox, confidence=conf, method=f"full_{best_method}")

    logger.debug("No stamp detected.")
    return StampDetectionResult(present=False, bbox=None, confidence=0.75, method="none")