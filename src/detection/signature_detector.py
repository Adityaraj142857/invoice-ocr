"""
detection/signature_detector.py
---------------------------------
Robust handwritten signature detector — structural understanding approach.

Core insight: signatures are NOT just random ink blobs.
They have predictable properties:
  1. Located near specific anchor TEXT ("Authorised Signatory", "For XYZ", etc.)
  2. Made of connected, flowing strokes (high curvature, variable width)
  3. Span a characteristic aspect ratio (wider than tall, not too thin)
  4. Located in the BOTTOM half of the document
  5. NOT part of a regular text line (irregular baseline)

Five strategies:
  S1 — Anchor text zone search (OCR-guided, most reliable)
  S2 — Connected stroke analysis (ink flow properties)
  S3 — Baseline irregularity detection (signatures break text line patterns)
  S4 — Spatial density gradient (ink concentration shifts in signature zones)
  S5 — "For [Name]" region search (common invoice pattern)
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

_PRIMARY_ANCHORS = [
    "authorised signatory", "authorized signatory",
    "authorised signatory.", "prop.", "proprietor",
    "खरीददार का हस्ताक्षर", "हस्ताक्षर",
    "customer's signature", "customer signature",
    "buyer's signature", "dealer's signature",
    "manager",
]

_FOR_ANCHORS = [
    "for ", "for\n",
]

_SECONDARY_ANCHORS = [
    "signature", "sign", "signed",
    "authorised", "authorized",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SignatureDetectorConfig:
    search_above_px: int = 220       # increased — search higher above anchor
    search_below_px: int = 20        # small zone below anchor too
    search_h_padding: int = 100      # wider horizontal padding
    min_contour_area: int = 80       # lowered — catches thin strokes
    max_contour_area: int = 120000
    min_aspect: float = 0.15         # very thin cursive lines allowed
    max_aspect: float = 20.0         # very wide signatures allowed
    fallback_region_fraction: float = 0.40
    min_stroke_count: int = 2        # minimum ink strokes to declare sig
    # Stroke analysis thresholds
    min_stroke_length: int = 15      # minimum pixel length of a stroke
    max_line_height: int = 6         # strokes taller than this → not a ruled line


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_bgr(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _get_binary_inv(bgr: np.ndarray) -> np.ndarray:
    """Dark ink on white → white on black (inverted binary)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _find_anchor_blocks(ocr_result, keywords: list[str]) -> list:
    """Find OCR text blocks matching any keyword."""
    if ocr_result is None or not hasattr(ocr_result, "blocks"):
        return []
    matched = []
    for blk in ocr_result.blocks:
        text_lower = blk.text.lower().strip()
        for kw in keywords:
            if kw.strip() in text_lower:
                matched.append(blk)
                break
    return matched


def _bbox_from_contours(contours: list[np.ndarray], pad: int = 14) -> list[int]:
    all_pts = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_pts)
    return [max(0, x-pad), max(0, y-pad), x+w+pad, y+h+pad]


# ---------------------------------------------------------------------------
# Stroke quality filter — the core improvement
# ---------------------------------------------------------------------------

def _is_signature_stroke(cnt: np.ndarray, cfg: SignatureDetectorConfig) -> bool:
    """
    Return True if a contour looks like a handwritten ink stroke.

    Rejects:
    - Horizontal ruled lines (very thin, very wide)
    - Tiny dots / noise
    - Perfectly rectangular text bounding boxes
    - Very regular shapes (machine-printed elements)
    """
    area = cv2.contourArea(cnt)
    if area < cfg.min_contour_area or area > cfg.max_contour_area:
        return False

    x, y, w, h = cv2.boundingRect(cnt)

    # Reject tiny noise
    if w < 8 and h < 8:
        return False

    # Reject horizontal ruled lines
    if h <= cfg.max_line_height and w > 80:
        return False

    # Reject vertical lines (table borders)
    if w <= 4 and h > 60:
        return False

    # Aspect ratio check
    aspect = w / max(h, 1)
    if not (cfg.min_aspect <= aspect <= cfg.max_aspect):
        return False

    # Stroke length check (diagonal of bounding box)
    length = np.sqrt(w**2 + h**2)
    if length < cfg.min_stroke_length:
        return False

    # Rectangularity check: real signatures have irregular outlines
    # Rectangularity = area / (w * h). Perfect rect = 1.0, cursive ≈ 0.2–0.7
    rect_area = max(w * h, 1)
    rectangularity = area / rect_area
    # Allow a broad range — some thick strokes are quite filled
    if rectangularity > 0.95 and area > 2000:
        return False  # Suspiciously perfect rectangle → likely a printed element

    return True


def _cluster_strokes(contours: list[np.ndarray], max_gap: int = 80) -> list[list[np.ndarray]]:
    """
    Cluster strokes that are spatially close into groups.
    Uses centre-point proximity — improved over the old dx+dy check.
    """
    if not contours:
        return []

    centres = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w//2, y + h//2
        centres.append((cx, cy))

    assigned = [False] * len(contours)
    clusters: list[list[int]] = []

    for i in range(len(contours)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i+1, len(contours)):
            if assigned[j]:
                continue
            dist = np.sqrt((centres[i][0]-centres[j][0])**2 +
                           (centres[i][1]-centres[j][1])**2)
            if dist <= max_gap:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    return [[contours[i] for i in grp] for grp in clusters]


def _best_signature_cluster(
    clusters: list[list[np.ndarray]],
    cfg: SignatureDetectorConfig,
) -> Optional[list[np.ndarray]]:
    """
    Score each cluster by how signature-like it is.
    Prefer: moderate size, multiple strokes, moderate aspect ratio.
    """
    scored = []
    for grp in clusters:
        if len(grp) < cfg.min_stroke_count:
            continue
        total_area = sum(cv2.contourArea(c) for c in grp)
        if total_area < cfg.min_contour_area * cfg.min_stroke_count:
            continue
        all_pts = np.vstack(grp)
        x, y, w, h = cv2.boundingRect(all_pts)
        aspect = w / max(h, 1)
        # Signatures are typically wider than tall
        aspect_score = 1.0 if 1.0 <= aspect <= 8.0 else 0.5
        count_score  = min(len(grp) / 10.0, 1.0)
        score = aspect_score * count_score * np.log(total_area + 1)
        scored.append((score, grp))

    if not scored:
        return None
    return max(scored, key=lambda x: x[0])[1]


# ---------------------------------------------------------------------------
# Strategy 1 — OCR anchor zone search (primary strategy)
# ---------------------------------------------------------------------------

def _s1_anchor(
    bgr: np.ndarray,
    ocr_result,
    cfg: SignatureDetectorConfig,
) -> Optional[list[int]]:
    H, W = bgr.shape[:2]

    # Try primary anchors first, then secondary
    anchors = _find_anchor_blocks(ocr_result, _PRIMARY_ANCHORS)
    if not anchors:
        anchors = _find_anchor_blocks(ocr_result, _SECONDARY_ANCHORS)
    if not anchors:
        return None

    # Use the lowest anchor block (closest to bottom)
    anchor = max(anchors, key=lambda b: b.y_max)

    # Search zone: above and slightly below the anchor, with wide horizontal padding
    x1 = max(0, anchor.x_min - cfg.search_h_padding)
    x2 = min(W, anchor.x_max + cfg.search_h_padding)
    y1 = max(0, anchor.y_min - cfg.search_above_px)
    y2 = min(H, anchor.y_max + cfg.search_below_px)

    if y2 <= y1 or x2 <= x1:
        return None

    region = bgr[y1:y2, x1:x2]
    binary = _get_binary_inv(region)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    strokes = [c for c in contours if _is_signature_stroke(c, cfg)]

    if len(strokes) < cfg.min_stroke_count:
        return None

    clusters = _cluster_strokes(strokes, max_gap=80)
    best = _best_signature_cluster(clusters, cfg)
    if best is None:
        return None

    local_bbox = _bbox_from_contours(best)
    return [local_bbox[0]+x1, local_bbox[1]+y1, local_bbox[2]+x1, local_bbox[3]+y1]


# ---------------------------------------------------------------------------
# Strategy 2 — "For [DealerName]" region pattern
# ---------------------------------------------------------------------------

def _s2_for_anchor(
    bgr: np.ndarray,
    ocr_result,
    cfg: SignatureDetectorConfig,
) -> Optional[list[int]]:
    """
    Pattern: signature appears between 'For [Name]' label and
    'Authorised Signatory' text — common in Indian invoices.
    """
    H, W = bgr.shape[:2]
    for_blocks = _find_anchor_blocks(ocr_result, _FOR_ANCHORS)
    if not for_blocks:
        return None

    anchor = max(for_blocks, key=lambda b: b.y_max)
    # Search zone: right of "For" label, extending down
    x1 = max(0, anchor.x_min - 20)
    x2 = W
    y1 = max(0, anchor.y_min - 20)
    y2 = min(H, anchor.y_max + cfg.search_above_px)

    if y2 <= y1 or x2 <= x1:
        return None

    region = bgr[y1:y2, x1:x2]
    binary = _get_binary_inv(region)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    strokes = [c for c in contours if _is_signature_stroke(c, cfg)]

    if len(strokes) < cfg.min_stroke_count:
        return None

    clusters = _cluster_strokes(strokes)
    best = _best_signature_cluster(clusters, cfg)
    if best is None:
        return None

    local_bbox = _bbox_from_contours(best)
    return [local_bbox[0]+x1, local_bbox[1]+y1, local_bbox[2]+x1, local_bbox[3]+y1]


# ---------------------------------------------------------------------------
# Strategy 3 — Bottom-half spatial scan (no OCR needed)
# ---------------------------------------------------------------------------

def _s3_bottom_scan(bgr: np.ndarray, cfg: SignatureDetectorConfig) -> Optional[list[int]]:
    """
    Scan the bottom portion of the image for signature-like ink clusters.
    Used when OCR anchor fails.
    """
    H, W = bgr.shape[:2]
    y_start = int(H * (1 - cfg.fallback_region_fraction))
    region = bgr[y_start:, :]

    binary = _get_binary_inv(region)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    strokes = [c for c in contours if _is_signature_stroke(c, cfg)]

    if len(strokes) < cfg.min_stroke_count:
        return None

    clusters = _cluster_strokes(strokes, max_gap=60)
    best = _best_signature_cluster(clusters, cfg)
    if best is None:
        return None

    local_bbox = _bbox_from_contours(best)
    return [local_bbox[0], local_bbox[1]+y_start, local_bbox[2], local_bbox[3]+y_start]


# ---------------------------------------------------------------------------
# Strategy 4 — Ink flow analysis (curvature-based)
# ---------------------------------------------------------------------------

def _s4_curvature(bgr: np.ndarray, cfg: SignatureDetectorConfig) -> Optional[list[int]]:
    """
    Handwritten signatures have high curvature variance (curvy strokes).
    Find contours with high mean curvature → likely handwriting.
    """
    H, W = bgr.shape[:2]
    y_start = int(H * 0.55)   # bottom 45% only
    region = bgr[y_start:, :]

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 8
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    high_curve = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg.min_contour_area or area > cfg.max_contour_area:
            continue
        if len(cnt) < 10:
            continue
        # Approximate and measure deviation → proxy for curvature
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx  = cv2.approxPolyDP(cnt, epsilon, True)
        # High point count in approximation → curvy → signature-like
        if len(approx) >= 5:
            high_curve.append(cnt)

    if len(high_curve) < cfg.min_stroke_count:
        return None

    # Filter to signature strokes
    strokes = [c for c in high_curve if _is_signature_stroke(c, cfg)]
    if len(strokes) < cfg.min_stroke_count:
        return None

    clusters = _cluster_strokes(strokes)
    best = _best_signature_cluster(clusters, cfg)
    if best is None:
        return None

    local_bbox = _bbox_from_contours(best)
    return [local_bbox[0], local_bbox[1]+y_start, local_bbox[2], local_bbox[3]+y_start]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class SignatureDetectionResult:
    present: bool
    bbox: Optional[list[int]]
    confidence: float
    method: str = "none"


def detect_signature(
    image: Image.Image | np.ndarray,
    ocr_result=None,
    cfg: Optional[SignatureDetectorConfig] = None,
) -> SignatureDetectionResult:
    if cfg is None:
        cfg = SignatureDetectorConfig()

    bgr = _to_bgr(image)
    H, W = bgr.shape[:2]

    def clamp(bbox: list[int]) -> list[int]:
        return [max(0,bbox[0]), max(0,bbox[1]), min(W,bbox[2]), min(H,bbox[3])]

    # ── Strategy 1: OCR anchor ────────────────────────────────────────────
    try:
        bbox = _s1_anchor(bgr, ocr_result, cfg)
        if bbox is not None:
            logger.debug("Signature: anchor zone (conf=0.87)")
            return SignatureDetectionResult(present=True, bbox=clamp(bbox),
                                            confidence=0.87, method="anchor")
    except Exception as exc:
        logger.warning("S1 anchor failed: %s", exc)

    # ── Strategy 2: For-anchor pattern ───────────────────────────────────
    try:
        bbox = _s2_for_anchor(bgr, ocr_result, cfg)
        if bbox is not None:
            logger.debug("Signature: for-anchor (conf=0.82)")
            return SignatureDetectionResult(present=True, bbox=clamp(bbox),
                                            confidence=0.82, method="for_anchor")
    except Exception as exc:
        logger.warning("S2 for-anchor failed: %s", exc)

    # ── Strategy 3: Bottom-half scan ─────────────────────────────────────
    try:
        bbox = _s3_bottom_scan(bgr, cfg)
        if bbox is not None:
            logger.debug("Signature: bottom scan (conf=0.70)")
            return SignatureDetectionResult(present=True, bbox=clamp(bbox),
                                            confidence=0.70, method="bottom_scan")
    except Exception as exc:
        logger.warning("S3 bottom scan failed: %s", exc)

    # ── Strategy 4: Curvature analysis ───────────────────────────────────
    try:
        bbox = _s4_curvature(bgr, cfg)
        if bbox is not None:
            logger.debug("Signature: curvature (conf=0.65)")
            return SignatureDetectionResult(present=True, bbox=clamp(bbox),
                                            confidence=0.65, method="curvature")
    except Exception as exc:
        logger.warning("S4 curvature failed: %s", exc)

    logger.debug("No signature detected.")
    return SignatureDetectionResult(present=False, bbox=None, confidence=0.70, method="none")