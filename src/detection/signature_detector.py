"""
detection/signature_detector.py
---------------------------------
Handwritten signature detector for invoice images.

Strategy:
  1. Find anchor text  – search OCR blocks for "Authorised Signatory",
     "Authorized Signatory", "हस्ताक्षर", "खरीददार का हस्ताक्षर", 
     "For [DealerName]", "Customer's Signature" etc.
  2. Define a search zone above/left of the anchor text block.
  3. Within that zone, find ink contours (dark, non-text-like shapes).
  4. Filter contours by aspect ratio and stroke continuity to identify
     cursive handwriting vs printed text.
  5. Fallback: scan the bottom half of the document for similar contours
     if no anchor text is found.

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
# Anchor keyword sets
# ---------------------------------------------------------------------------

_SIGNATURE_ANCHOR_KEYWORDS = [
    # English variants
    "authorised signatory",
    "authorized signatory",
    "authorised signatory.",
    "authorisedSignatory",
    "prop.",
    "proprietor",
    "for ",               # "For AMS TRACTORS"
    "customer's signature",
    "customer signature",
    "buyer's signature",
    "खरीददार का हस्ताक्षर",    # Buyer's signature (Hindi)
    "हस्ताक्षर",                 # Signature (Hindi)
    "ग्राहक के हस्ताक्षर",       # Customer's signature (Hindi)
    "dealer's signature",
    "manager",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SignatureDetectorConfig:
    # Search zone: how many pixels ABOVE the anchor to look for signature
    search_above_px: int = 180
    # Search zone: horizontal expansion around anchor bbox
    search_h_padding: int = 60
    # Minimum contour area (px²) — filters noise
    min_contour_area: int = 150
    # Maximum contour area (px²) — filters large printed text blocks
    max_contour_area: int = 80000
    # Aspect ratio limits for signature bounding box
    min_aspect: float = 0.3    # Not too tall and thin
    max_aspect: float = 12.0   # Not too wide and flat
    # Fallback: search bottom fraction of image
    fallback_region_fraction: float = 0.45
    # Minimum number of contours to declare a signature present
    min_contour_count: int = 2


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


def _get_binary(bgr: np.ndarray) -> np.ndarray:
    """Convert to inverted binary — dark ink on white background → white on black."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _find_ink_contours(
    binary: np.ndarray,
    cfg: SignatureDetectorConfig,
) -> list[np.ndarray]:
    """Find contours that could be handwriting strokes."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg.min_contour_area or area > cfg.max_contour_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect = w / h
        if cfg.min_aspect <= aspect <= cfg.max_aspect:
            filtered.append(cnt)
    return filtered


def _contours_to_merged_bbox(
    contours: list[np.ndarray],
) -> list[int]:
    """Merge a list of contours into one enclosing bounding box."""
    all_pts = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_pts)
    pad = 12
    return [max(0, x - pad), max(0, y - pad), x + w + pad, y + h + pad]


def _find_anchor_blocks(
    ocr_result: object,   # OCRResult — avoid circular import
    keywords: list[str],
) -> list[object]:
    """
    Find OCR text blocks matching any of the anchor keywords.
    Returns list of matching TextBlock objects.
    """
    matched = []
    if ocr_result is None or not hasattr(ocr_result, "blocks"):
        return matched
    for blk in ocr_result.blocks:
        text_lower = blk.text.lower().strip()
        for kw in keywords:
            if kw in text_lower:
                matched.append(blk)
                break
    return matched


def _define_search_zone(
    anchor_blocks: list[object],
    img_w: int,
    img_h: int,
    cfg: SignatureDetectorConfig,
) -> Optional[tuple[int, int, int, int]]:
    """
    Define a rectangular search zone based on anchor block positions.

    Returns (x1, y1, x2, y2) or None.
    """
    if not anchor_blocks:
        return None

    # Use the topmost anchor block as reference
    anchor = min(anchor_blocks, key=lambda b: b.y_min)

    x1 = max(0, anchor.x_min - cfg.search_h_padding)
    x2 = min(img_w, anchor.x_max + cfg.search_h_padding)
    y1 = max(0, anchor.y_min - cfg.search_above_px)
    y2 = anchor.y_min  # Search only ABOVE the anchor text

    if y2 <= y1:
        return None

    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Detection strategies
# ---------------------------------------------------------------------------

def _detect_in_zone(
    bgr: np.ndarray,
    zone: tuple[int, int, int, int],
    cfg: SignatureDetectorConfig,
) -> Optional[list[int]]:
    """
    Search for a signature in a defined zone.
    Returns full-image bbox or None.
    """
    x1, y1, x2, y2 = zone
    region = bgr[y1:y2, x1:x2]

    if region.size == 0:
        return None

    binary = _get_binary(region)
    contours = _find_ink_contours(binary, cfg)

    if len(contours) < cfg.min_contour_count:
        return None

    # Additional filter: remove contours that look like printed text
    # (very regular spacing, very uniform height)
    signature_contours = _filter_signature_contours(contours)

    if len(signature_contours) < cfg.min_contour_count:
        return None

    # Build merged bbox in region coordinates, then offset to full image
    local_bbox = _contours_to_merged_bbox(signature_contours)
    return [
        local_bbox[0] + x1,
        local_bbox[1] + y1,
        local_bbox[2] + x1,
        local_bbox[3] + y1,
    ]


def _filter_signature_contours(
    contours: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Keep contours that are more likely cursive handwriting than printed text.

    Heuristic: signatures have irregular stroke widths and non-uniform heights.
    We simply remove extremely thin horizontal lines (ruled lines) and
    very small dot-like artefacts.
    """
    result = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Skip very thin horizontal lines (table borders, underlines)
        if h < 4 and w > 80:
            continue
        # Skip tiny dots
        if w < 8 and h < 8:
            continue
        result.append(cnt)
    return result


def _detect_fallback(
    bgr: np.ndarray,
    cfg: SignatureDetectorConfig,
) -> Optional[list[int]]:
    """
    Fallback: scan the bottom portion of the image for signature-like marks.
    Used when no anchor text is found.
    """
    h, w = bgr.shape[:2]
    y_start = int(h * (1 - cfg.fallback_region_fraction))
    region = bgr[y_start:h, 0:w]

    if region.size == 0:
        return None

    binary = _get_binary(region)
    contours = _find_ink_contours(binary, cfg)
    contours = _filter_signature_contours(contours)

    if len(contours) < cfg.min_contour_count:
        return None

    # Among all candidate contours, look for a cluster that forms a
    # plausible signature region (not just scattered dots)
    clustered = _cluster_contours(contours, max_gap=60)
    if not clustered:
        return None

    # Pick the largest cluster
    best = max(clustered, key=lambda grp: sum(cv2.contourArea(c) for c in grp))
    if len(best) < cfg.min_contour_count:
        return None

    local_bbox = _contours_to_merged_bbox(best)
    full_bbox = [
        local_bbox[0],
        local_bbox[1] + y_start,
        local_bbox[2],
        local_bbox[3] + y_start,
    ]
    return full_bbox


def _cluster_contours(
    contours: list[np.ndarray],
    max_gap: int,
) -> list[list[np.ndarray]]:
    """
    Simple spatial clustering: group contours whose bounding boxes are
    within `max_gap` pixels of each other.
    """
    if not contours:
        return []

    # Get centre points
    centres = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        centres.append((x + w / 2, y + h / 2))

    clusters: list[list[int]] = []
    assigned = [False] * len(contours)

    for i in range(len(contours)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, len(contours)):
            if assigned[j]:
                continue
            dx = abs(centres[i][0] - centres[j][0])
            dy = abs(centres[i][1] - centres[j][1])
            if dx <= max_gap and dy <= max_gap:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    return [[contours[i] for i in grp] for grp in clusters]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class SignatureDetectionResult:
    present: bool
    bbox: Optional[list[int]]    # [x_min, y_min, x_max, y_max]
    confidence: float
    method: str = "none"


def detect_signature(
    image: Image.Image | np.ndarray,
    ocr_result: Optional[object] = None,
    cfg: Optional[SignatureDetectorConfig] = None,
) -> SignatureDetectionResult:
    """
    Detect a handwritten signature in an invoice image.

    Args:
        image:      PIL Image or OpenCV BGR ndarray.
        ocr_result: OCRResult from PaddleOCR (used to find anchor keywords).
                    If None, falls back to spatial search only.
        cfg:        Optional config override.

    Returns:
        SignatureDetectionResult with present flag, bbox, and confidence.
    """
    if cfg is None:
        cfg = SignatureDetectorConfig()

    bgr = _pil_to_bgr(image)
    img_h, img_w = bgr.shape[:2]

    # ── Strategy 1: anchor-based search ──────────────────────────────────
    if ocr_result is not None:
        anchor_blocks = _find_anchor_blocks(ocr_result, _SIGNATURE_ANCHOR_KEYWORDS)
        if anchor_blocks:
            zone = _define_search_zone(anchor_blocks, img_w, img_h, cfg)
            if zone is not None:
                bbox = _detect_in_zone(bgr, zone, cfg)
                if bbox is not None:
                    bbox = [
                        max(0, bbox[0]), max(0, bbox[1]),
                        min(img_w, bbox[2]), min(img_h, bbox[3]),
                    ]
                    logger.debug("Signature detected via anchor zone: %s", bbox)
                    return SignatureDetectionResult(
                        present=True, bbox=bbox, confidence=0.85, method="anchor"
                    )

    # ── Strategy 2: full bottom-half scan ────────────────────────────────
    bbox_fallback = _detect_fallback(bgr, cfg)
    if bbox_fallback is not None:
        bbox_fallback = [
            max(0, bbox_fallback[0]), max(0, bbox_fallback[1]),
            min(img_w, bbox_fallback[2]), min(img_h, bbox_fallback[3]),
        ]
        logger.debug("Signature detected via fallback scan: %s", bbox_fallback)
        return SignatureDetectionResult(
            present=True, bbox=bbox_fallback, confidence=0.65, method="fallback"
        )

    logger.debug("No signature detected.")
    return SignatureDetectionResult(present=False, bbox=None, confidence=0.75, method="none")
