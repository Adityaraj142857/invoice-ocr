"""
preprocessor.py
---------------
Image pre-processing pipeline for invoice / quotation images before OCR.

Steps applied in order:
  1. Resize         – cap long-side at MAX_DIM to avoid slow OCR on huge images
  2. Grayscale      – convert to single channel for analysis
  3. Rotation fix   – detect and correct physical rotation (0 / 90 / 180 / 270°)
  4. Deskew         – correct small skew angles (< ±45°) using Hough lines
  5. Denoise        – remove salt-and-pepper / scanner noise (fastNlMeans)
  6. Binarise       – adaptive thresholding (Sauvola-style via THRESH_GAUSSIAN)
  7. Contrast boost – CLAHE on the grayscale before final return

The pipeline returns BOTH the cleaned RGB image (for VLM use) and the
binarised grayscale (for PaddleOCR).

Dependencies:
    pip install opencv-python-headless numpy Pillow
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """All knobs in one place — pass a custom instance to override defaults."""

    # ── Resize ───────────────────────────────────────────────────────────
    max_dim: int = 2048          # Long-side pixel cap before processing
    min_dim: int = 800           # Upscale if image is smaller than this

    # ── Rotation detection ───────────────────────────────────────────────
    rotation_confidence_threshold: float = 0.6
    # Minimum fraction of text lines that must agree on an orientation

    # ── Deskew ───────────────────────────────────────────────────────────
    deskew_max_angle: float = 45.0   # Ignore angles beyond this (not skew)
    deskew_enabled: bool = True

    # ── Denoise ──────────────────────────────────────────────────────────
    denoise_enabled: bool = True
    denoise_h: int = 10              # Filter strength (higher = more blur)
    denoise_template_window: int = 7
    denoise_search_window: int = 21

    # ── Binarise ─────────────────────────────────────────────────────────
    binarise_enabled: bool = True
    adaptive_block_size: int = 31    # Must be odd
    adaptive_c: int = 10             # Subtracted constant

    # ── CLAHE ────────────────────────────────────────────────────────────
    clahe_enabled: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = field(default_factory=lambda: (8, 8))


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR ndarray."""
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR or grayscale ndarray to PIL RGB Image."""
    if len(cv2_img.shape) == 2:
        # Grayscale → convert to RGB PIL
        return Image.fromarray(cv2_img).convert("RGB")
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def _resize(img: np.ndarray, max_dim: int, min_dim: int) -> np.ndarray:
    """
    Resize image so:
      • long side ≤ max_dim (downscale if too large)
      • short side ≥ min_dim (upscale if too small)
    Aspect ratio is always preserved.
    """
    h, w = img.shape[:2]
    long_side = max(h, w)
    short_side = min(h, w)

    scale = 1.0
    if long_side > max_dim:
        scale = max_dim / long_side
    elif short_side < min_dim:
        scale = min_dim / short_side

    if abs(scale - 1.0) < 0.01:
        return img  # Already fine

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale, handling already-gray inputs."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Step 3 — Rotation fix (0 / 90 / 180 / 270°)
# ---------------------------------------------------------------------------

def _detect_rotation_angle(gray: np.ndarray) -> int:
    """
    Detect the closest 90° rotation needed using the Projection Profile method.

    Strategy:
      - Try 0, 90, 180, 270 degree rotations.
      - For each, compute the horizontal projection profile (row-wise sum of
        binary pixels).
      - The correct orientation produces the sharpest profile (highest variance
        between rows), because text lines create strong horizontal bands.

    Returns:
        0, 90, 180, or 270 (degrees to rotate counter-clockwise to fix).
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_angle = 0
    best_score = -1.0

    for angle in (0, 90, 180, 270):
        rotated = _rotate_exact(binary, angle)
        profile = np.sum(rotated, axis=1).astype(np.float64)
        score = float(np.var(profile))
        if score > best_score:
            best_score = score
            best_angle = angle

    logger.debug("Rotation detection: best angle=%d°  score=%.1f", best_angle, best_score)
    return best_angle


def _rotate_exact(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image by an exact multiple of 90° without cropping."""
    if angle == 0:
        return img
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    raise ValueError(f"angle must be 0/90/180/270, got {angle}")


def fix_rotation(bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Detect and correct 90° / 180° / 270° rotation.

    Args:
        bgr: OpenCV BGR image.

    Returns:
        (corrected_bgr, angle_applied)
    """
    gray = _to_gray(bgr)
    angle = _detect_rotation_angle(gray)
    if angle == 0:
        return bgr, 0
    corrected = _rotate_exact(bgr, angle)
    logger.info("Rotation corrected: %d°", angle)
    return corrected, angle


# ---------------------------------------------------------------------------
# Step 4 — Deskew (fine angle correction)
# ---------------------------------------------------------------------------

def _estimate_skew_angle(gray: np.ndarray, max_angle: float) -> float:
    """
    Estimate small skew angle using Hough line transform on edge image.

    Returns angle in degrees (positive = clockwise tilt).
    Returns 0.0 if no reliable angle found.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=gray.shape[1] // 4,
        maxLineGap=20,
    )

    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue  # Vertical line — skip
        angle_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Keep only near-horizontal lines (text baselines)
        if abs(angle_deg) <= max_angle:
            angles.append(angle_deg)

    if not angles:
        return 0.0

    # Use median to be robust against outliers
    skew = float(np.median(angles))
    logger.debug("Skew estimate: %.2f° (from %d lines)", skew, len(angles))
    return skew


def deskew(bgr: np.ndarray, max_angle: float = 45.0) -> tuple[np.ndarray, float]:
    """
    Correct small skew (non-right-angle tilt) in the image.

    Args:
        bgr:       OpenCV BGR image (already rotation-corrected).
        max_angle: Ignore angles beyond this threshold.

    Returns:
        (deskewed_bgr, skew_angle_corrected)
    """
    gray = _to_gray(bgr)
    skew = _estimate_skew_angle(gray, max_angle)

    if abs(skew) < 0.5:
        # Not worth touching — avoid unnecessary interpolation
        return bgr, 0.0

    h, w = bgr.shape[:2]
    centre = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(centre, skew, 1.0)

    # Compute new bounding box size to avoid cropping corners
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    deskewed = cv2.warpAffine(
        bgr, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logger.info("Deskewed: %.2f°", skew)
    return deskewed, skew


# ---------------------------------------------------------------------------
# Step 5 — Denoise
# ---------------------------------------------------------------------------

def denoise(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Apply Non-local Means denoising to the grayscale image.
    Effective for scanner noise and compression artefacts.
    """
    return cv2.fastNlMeansDenoising(
        gray,
        h=cfg.denoise_h,
        templateWindowSize=cfg.denoise_template_window,
        searchWindowSize=cfg.denoise_search_window,
    )


# ---------------------------------------------------------------------------
# Step 6 — Binarise
# ---------------------------------------------------------------------------

def binarise(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Adaptive Gaussian thresholding — handles uneven illumination across
    the document (common in phone photos and scans through plastic sleeves).

    block_size must be odd; enforce that here.
    """
    block_size = cfg.adaptive_block_size
    if block_size % 2 == 0:
        block_size += 1

    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=cfg.adaptive_c,
    )
    return binary


# ---------------------------------------------------------------------------
# Step 7 — CLAHE contrast enhancement
# ---------------------------------------------------------------------------

def apply_clahe(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalisation.
    Improves readability in dark or low-contrast document regions.
    """
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=cfg.clahe_tile_grid,
    )
    return clahe.apply(gray)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@dataclass
class PreprocessResult:
    """Output bundle returned by the preprocessing pipeline."""

    rgb_clean: Image.Image        # RGB PIL — for VLM / display
    gray_enhanced: np.ndarray     # Grayscale uint8 — for PaddleOCR
    binary: np.ndarray            # Binarised uint8 — optional OCR fallback
    rotation_applied: int         # Degrees (0 / 90 / 180 / 270)
    skew_applied: float           # Fine deskew angle in degrees
    original_size: tuple[int, int]  # (W, H) before any processing
    final_size: tuple[int, int]     # (W, H) after processing


def preprocess(
    image: Image.Image | np.ndarray,
    cfg: Optional[PreprocessConfig] = None,
) -> PreprocessResult:
    """
    Full preprocessing pipeline for a single invoice image.

    Args:
        image: PIL Image or OpenCV BGR ndarray.
        cfg:   Optional config override. Uses defaults if None.

    Returns:
        PreprocessResult with all output variants.
    """
    if cfg is None:
        cfg = PreprocessConfig()

    # ── Normalise input to OpenCV BGR ─────────────────────────────────────
    if isinstance(image, Image.Image):
        bgr = _pil_to_cv2(image)
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    original_h, original_w = bgr.shape[:2]
    original_size = (original_w, original_h)

    # ── Step 1: Resize ────────────────────────────────────────────────────
    bgr = _resize(bgr, cfg.max_dim, cfg.min_dim)

    # ── Step 2: Grayscale ─────────────────────────────────────────────────
    gray = _to_gray(bgr)

    # ── Step 3: Rotation fix ──────────────────────────────────────────────
    bgr, rotation_applied = fix_rotation(bgr)
    gray = _to_gray(bgr)  # Re-derive after rotation

    # ── Step 4: Deskew ────────────────────────────────────────────────────
    skew_applied = 0.0
    if cfg.deskew_enabled:
        bgr, skew_applied = deskew(bgr, cfg.deskew_max_angle)
        gray = _to_gray(bgr)

    # ── Step 5: Denoise ───────────────────────────────────────────────────
    if cfg.denoise_enabled:
        gray = denoise(gray, cfg)

    # ── Step 6: CLAHE ─────────────────────────────────────────────────────
    if cfg.clahe_enabled:
        gray = apply_clahe(gray, cfg)

    # ── Step 7: Binarise ──────────────────────────────────────────────────
    binary = binarise(gray, cfg) if cfg.binarise_enabled else gray.copy()

    # ── Rebuild colour image with same geometric corrections ──────────────
    # Apply CLAHE in LAB L-channel on the colour version for VLM
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe_obj = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=cfg.clahe_tile_grid,
    )
    l_chan = clahe_obj.apply(l_chan)
    bgr_enhanced = cv2.cvtColor(cv2.merge([l_chan, a_chan, b_chan]), cv2.COLOR_LAB2BGR)

    final_h, final_w = bgr_enhanced.shape[:2]

    return PreprocessResult(
        rgb_clean=_cv2_to_pil(bgr_enhanced),
        gray_enhanced=gray,
        binary=binary,
        rotation_applied=rotation_applied,
        skew_applied=skew_applied,
        original_size=original_size,
        final_size=(final_w, final_h),
    )


def preprocess_batch(
    images: list[Image.Image],
    cfg: Optional[PreprocessConfig] = None,
    fail_silently: bool = True,
) -> list[PreprocessResult | None]:
    """
    Preprocess a list of images (e.g. all pages of a PDF).

    Args:
        images:        List of PIL Images.
        cfg:           Shared config for all images.
        fail_silently: If True, failed pages return None instead of raising.

    Returns:
        List of PreprocessResult (or None for failed pages).
    """
    results: list[PreprocessResult | None] = []
    for idx, img in enumerate(images):
        try:
            results.append(preprocess(img, cfg))
        except Exception as exc:
            logger.error("Preprocessing failed on page %d: %s", idx + 1, exc)
            if fail_silently:
                results.append(None)
            else:
                raise
    return results


# ---------------------------------------------------------------------------
# Save helper (useful for debugging / EDA notebook)
# ---------------------------------------------------------------------------

def save_debug_outputs(
    result: PreprocessResult,
    output_dir: str | Path,
    stem: str = "doc",
) -> None:
    """
    Save all variants from a PreprocessResult to disk for inspection.

    Creates:
      <stem>_clean.png       – enhanced RGB
      <stem>_gray.png        – enhanced grayscale
      <stem>_binary.png      – binarised
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.rgb_clean.save(str(output_dir / f"{stem}_clean.png"))
    Image.fromarray(result.gray_enhanced).save(str(output_dir / f"{stem}_gray.png"))
    Image.fromarray(result.binary).save(str(output_dir / f"{stem}_binary.png"))

    logger.info(
        "Debug outputs saved to '%s' (rotation=%d° skew=%.1f°)",
        output_dir, result.rotation_applied, result.skew_applied,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python preprocessor.py <image_path> [output_dir]")
        sys.exit(1)

    from PIL import Image as PILImage

    img_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else img_path.parent / "debug"

    pil_img = PILImage.open(str(img_path)).convert("RGB")
    result = preprocess(pil_img)

    print(f"\nOriginal size : {result.original_size[0]}×{result.original_size[1]} px")
    print(f"Final size    : {result.final_size[0]}×{result.final_size[1]} px")
    print(f"Rotation fixed: {result.rotation_applied}°")
    print(f"Skew corrected: {result.skew_applied:.2f}°")

    save_debug_outputs(result, out_dir, stem=img_path.stem)
    print(f"\nDebug images saved to: {out_dir}")
