"""
extraction/field_parser.py
---------------------------
Rule-based / regex field extractor that operates on OCRResult.

Complements the VLM extractor — used for consensus and as a standalone
fallback when VLM is not available or returns null.

Fields extracted:
  - dealer_name
  - model_name
  - horse_power
  - asset_cost
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Import OCR types lazily to avoid circular imports ────────────────────
try:
    from ocr.paddle_ocr import OCRResult, TextBlock
    from ocr.ocr_utils import (
        clean_text,
        extract_amount,
        extract_all_amounts,
        extract_hp,
        is_likely_total_row,
        normalise_dealer_name,
    )
except ImportError:
    # Allow standalone use
    OCRResult = None
    TextBlock = None
    from ocr.ocr_utils import (
        clean_text,
        extract_amount,
        extract_all_amounts,
        extract_hp,
        is_likely_total_row,
        normalise_dealer_name,
    )


# ---------------------------------------------------------------------------
# Known tractor brand anchors — used to find model lines
# ---------------------------------------------------------------------------

_TRACTOR_BRANDS = [
    "mahindra", "swaraj", "eicher", "powertrac", "sonalika",
    "john deere", "new holland", "kubota", "escorts", "farmtrac",
    "force", "indo farm", "captain", "vst", "preet", "standard",
    "tafe", "massey", "massey ferguson",
]

# Patterns that precede a model name on the same line
_MODEL_LABEL_PATTERNS = [
    r"model\s*[:\-–.]+\s*",
    r"model\s+name\s*[:\-–.]+\s*",
    r"tractor\s+model\s*[:\-–.]+\s*",
    r"vehicle\s+model\s*[:\-–.]+\s*",
]

# Patterns that indicate an amount row (for asset cost extraction)
_AMOUNT_ROW_LABELS = [
    "total", "grand total", "amount", "net amount",
    "ex showroom", "ex-showroom", "exshowroom",
    "cost of one", "cost of tractor",
    "योग", "कुल", "जमा",
]

# Dealer section indicators in letterhead
_DEALER_SECTION_INDICATORS = [
    "gstin", "gst no", "gst in", "dealer code",
    "authorised dealer", "authorized dealer",
    "sales", "service", "spare",
]


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class FieldParserResult:
    dealer_name: Optional[str] = None
    model_name: Optional[str] = None
    horse_power: Optional[int] = None
    asset_cost: Optional[int] = None
    # Confidence per field (simple heuristic, 0–1)
    dealer_conf: float = 0.0
    model_conf: float = 0.0
    hp_conf: float = 0.0
    cost_conf: float = 0.0

    def to_dict(self) -> dict:
        return {
            "dealer_name": self.dealer_name,
            "model_name": self.model_name,
            "horse_power": self.horse_power,
            "asset_cost": self.asset_cost,
        }


# ---------------------------------------------------------------------------
# Individual field extractors
# ---------------------------------------------------------------------------

def _extract_dealer_name(ocr: "OCRResult") -> tuple[Optional[str], float]:
    """
    Heuristic: the dealer name is in the top ~25% of the document,
    usually the largest / most prominent text.

    Strategy:
    1. Look for blocks in the top quarter of the image.
    2. Among those, find the one that is NOT a GSTIN, date, or phone number.
    3. Return the longest such string.
    """
    if not ocr or not ocr.blocks:
        return None, 0.0

    img_h = ocr.image_shape[0] if ocr.image_shape[0] > 0 else 9999
    top_threshold = img_h * 0.3

    top_blocks = [b for b in ocr.blocks if b.y_max <= top_threshold]
    if not top_blocks:
        top_blocks = sorted(ocr.blocks, key=lambda b: b.y_min)[:8]

    # Filter out obvious non-name blocks
    candidates = []
    for blk in top_blocks:
        t = blk.text.strip()
        if len(t) < 3:
            continue
        if re.search(r"\b\d{15}\b", t):   # GSTIN (15 digits)
            continue
        if re.search(r"\d{10}", t):        # Phone number
            continue
        if re.search(r"\d{2}[/-]\d{2}[/-]\d{2,4}", t):  # Date
            continue
        if re.match(r"^\d+$", t):          # Pure number
            continue
        candidates.append(blk)

    if not candidates:
        return None, 0.0

    # Prefer the block with the highest y (furthest up) and longest text
    # as a proxy for the main header
    best = max(candidates, key=lambda b: (len(b.text), -b.y_min))
    name = normalise_dealer_name(best.text)

    if len(name) < 3:
        return None, 0.0

    conf = min(0.9, 0.5 + (best.confidence * 0.4))
    return name, conf


def _extract_model_name(ocr: "OCRResult") -> tuple[Optional[str], float]:
    """
    Find the tractor model name.

    Strategy:
    1. Look for lines containing known brand names.
    2. Look for lines matching "Model: ..." label pattern.
    3. If multiple candidates, prefer lines that also contain an HP value
       or an amount (they're more likely to be the active model).
    """
    if not ocr or not ocr.blocks:
        return None, 0.0

    full_text = ocr.full_text
    candidates: list[tuple[str, float]] = []

    # ── Pattern 1: "Model: ..." label ────────────────────────────────────
    for pattern in _MODEL_LABEL_PATTERNS:
        match = re.search(pattern + r"(.{3,60})", full_text, re.IGNORECASE)
        if match:
            raw = match.group(1).strip()
            raw = re.split(r"[\n\r]", raw)[0].strip()
            if raw:
                candidates.append((clean_text(raw), 0.85))

    # ── Pattern 2: known brand on a line ─────────────────────────────────
    lines = ocr.get_line_groups()
    for line_blocks in lines:
        line_text = " ".join(b.text for b in line_blocks).strip()
        line_lower = line_text.lower()
        for brand in _TRACTOR_BRANDS:
            if brand in line_lower:
                cleaned = clean_text(line_text)
                # Exclude lines that are clearly address or long descriptions
                if len(cleaned.split()) <= 8:
                    candidates.append((cleaned, 0.75))
                break

    if not candidates:
        return None, 0.0

    # Pick the one with the highest confidence, or shortest (most model-like)
    best_text, best_conf = max(candidates, key=lambda x: x[1])
    return best_text, best_conf


def _extract_hp(ocr: "OCRResult") -> tuple[Optional[int], float]:
    """
    Extract Horse Power from OCR text.

    Tries both the full text and individual blocks.
    When multiple values are found, picks the most common one.
    """
    if not ocr:
        return None, 0.0

    found_values: list[int] = []

    # Try full text first
    hp = extract_hp(ocr.full_text)
    if hp is not None:
        found_values.append(hp)

    # Try individual blocks (catches cases where HP is on its own line)
    for blk in ocr.blocks:
        hp_blk = extract_hp(blk.text)
        if hp_blk is not None:
            found_values.append(hp_blk)

    if not found_values:
        return None, 0.0

    # Most common value
    from collections import Counter
    most_common = Counter(found_values).most_common(1)[0][0]
    count = Counter(found_values).most_common(1)[0][1]
    conf = min(0.95, 0.6 + count * 0.15)
    return most_common, conf


def _extract_asset_cost(ocr: "OCRResult") -> tuple[Optional[int], float]:
    """
    Extract the total asset cost.

    Strategy:
    1. Find the 'Total' / 'Grand Total' row in the table.
    2. Extract the largest amount on that line.
    3. Fallback: take the largest amount in the bottom half of the document.
    4. Sanity check: amount must be between 50,000 and 50,000,000 (₹0.5L–₹5Cr).
    """
    if not ocr or not ocr.blocks:
        return None, 0.0

    img_h = ocr.image_shape[0] if ocr.image_shape[0] > 0 else 9999

    # ── Strategy 1: Total row ─────────────────────────────────────────────
    lines = ocr.get_line_groups()
    for line_blocks in lines:
        line_text = " ".join(b.text for b in line_blocks)
        if is_likely_total_row(line_text):
            amounts = extract_all_amounts(line_text)
            for amt in amounts:
                if 50_000 <= amt <= 50_000_000:
                    return amt, 0.90

    # ── Strategy 2: Largest amount in bottom half ─────────────────────────
    bottom_threshold = img_h * 0.5
    bottom_text = " ".join(
        b.text for b in ocr.blocks if b.y_min >= bottom_threshold
    )
    amounts = extract_all_amounts(bottom_text)
    plausible = [a for a in amounts if 50_000 <= a <= 50_000_000]

    if plausible:
        # Take the largest (most likely to be the total, not a sub-item)
        return max(plausible), 0.70

    # ── Strategy 3: Largest amount anywhere ──────────────────────────────
    all_amounts = extract_all_amounts(ocr.full_text)
    plausible_all = [a for a in all_amounts if 50_000 <= a <= 50_000_000]
    if plausible_all:
        return max(plausible_all), 0.55

    return None, 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_fields(ocr_result: "OCRResult") -> FieldParserResult:
    """
    Run all field parsers on an OCRResult.

    Args:
        ocr_result: OCRResult from PaddleOCR.

    Returns:
        FieldParserResult with extracted values and per-field confidences.
    """
    result = FieldParserResult()

    try:
        result.dealer_name, result.dealer_conf = _extract_dealer_name(ocr_result)
    except Exception as exc:
        logger.warning("Dealer name extraction failed: %s", exc)

    try:
        result.model_name, result.model_conf = _extract_model_name(ocr_result)
    except Exception as exc:
        logger.warning("Model name extraction failed: %s", exc)

    try:
        result.horse_power, result.hp_conf = _extract_hp(ocr_result)
    except Exception as exc:
        logger.warning("HP extraction failed: %s", exc)

    try:
        result.asset_cost, result.cost_conf = _extract_asset_cost(ocr_result)
    except Exception as exc:
        logger.warning("Asset cost extraction failed: %s", exc)

    logger.debug(
        "FieldParser: dealer=%s model=%s hp=%s cost=%s",
        result.dealer_name, result.model_name,
        result.horse_power, result.asset_cost,
    )
    return result
