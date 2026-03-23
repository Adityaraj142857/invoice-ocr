"""
extraction/consensus.py
------------------------
Cross-validates and merges outputs from the OCR rule-based parser
and the Qwen2.5-VL extractor to produce a final high-confidence result.

Logic per field:
  - If both agree → use that value, boost confidence.
  - If only one has a value → use it with its confidence.
  - If both disagree → use the one with higher confidence, flag for review.
  - For numeric fields (HP, cost) → use fuzzy equality (±5% tolerance).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FieldConsensus:
    value: Any
    confidence: float
    source: str       # "ocr", "vlm", "both", "none"
    conflicted: bool = False


@dataclass
class ConsensusResult:
    dealer_name: FieldConsensus = field(
        default_factory=lambda: FieldConsensus(None, 0.0, "none"))
    model_name: FieldConsensus = field(
        default_factory=lambda: FieldConsensus(None, 0.0, "none"))
    horse_power: FieldConsensus = field(
        default_factory=lambda: FieldConsensus(None, 0.0, "none"))
    asset_cost: FieldConsensus = field(
        default_factory=lambda: FieldConsensus(None, 0.0, "none"))
    signature_present: FieldConsensus = field(
        default_factory=lambda: FieldConsensus(None, 0.0, "none"))
    stamp_present: FieldConsensus = field(
        default_factory=lambda: FieldConsensus(None, 0.0, "none"))

    @property
    def overall_confidence(self) -> float:
        """Mean confidence across all non-null fields."""
        fields_list = [
            self.dealer_name, self.model_name, self.horse_power,
            self.asset_cost, self.signature_present, self.stamp_present,
        ]
        valid = [f.confidence for f in fields_list if f.value is not None]
        return round(sum(valid) / max(len(valid), 1), 4)

    def to_dict(self) -> dict:
        return {
            "dealer_name": self.dealer_name.value,
            "model_name": self.model_name.value,
            "horse_power": self.horse_power.value,
            "asset_cost": self.asset_cost.value,
            "signature_present": self.signature_present.value,
            "stamp_present": self.stamp_present.value,
        }

    def confidence_dict(self) -> dict:
        return {
            "dealer_name": self.dealer_name.confidence,
            "model_name": self.model_name.confidence,
            "horse_power": self.horse_power.confidence,
            "asset_cost": self.asset_cost.confidence,
            "signature_present": self.signature_present.confidence,
            "stamp_present": self.stamp_present.confidence,
            "overall": self.overall_confidence,
        }

    def has_conflicts(self) -> bool:
        return any([
            self.dealer_name.conflicted, self.model_name.conflicted,
            self.horse_power.conflicted, self.asset_cost.conflicted,
        ])


# ---------------------------------------------------------------------------
# Agreement helpers
# ---------------------------------------------------------------------------

def _numeric_agree(a: Optional[int], b: Optional[int], tolerance: float = 0.05) -> bool:
    """True if two numbers are within `tolerance` fraction of each other."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    diff = abs(a - b) / max(abs(a), abs(b), 1)
    return diff <= tolerance


def _text_agree(a: Optional[str], b: Optional[str]) -> bool:
    """Rough text agreement check (case-insensitive, strip whitespace)."""
    if a is None or b is None:
        return False
    return a.strip().lower() == b.strip().lower()


def _text_partial_agree(
    a: Optional[str],
    b: Optional[str],
    threshold: float = 0.8,
) -> bool:
    """
    Check if two strings share enough tokens to be considered the same.
    Used for model name matching (e.g. "Swaraj 742 XT" vs "742 XT Swaraj").
    """
    if a is None or b is None:
        return False
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return False
    overlap = len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))
    return overlap >= threshold


# ---------------------------------------------------------------------------
# Per-field consensus logic
# ---------------------------------------------------------------------------

def _merge_text_field(
    ocr_val: Optional[str],
    ocr_conf: float,
    vlm_val: Optional[str],
    vlm_conf: float,
    field_name: str,
) -> FieldConsensus:
    """Merge two optional text values with their confidences."""
    both_present = ocr_val is not None and vlm_val is not None

    if not both_present:
        if vlm_val is not None:
            return FieldConsensus(vlm_val, vlm_conf, "vlm")
        if ocr_val is not None:
            return FieldConsensus(ocr_val, ocr_conf, "ocr")
        return FieldConsensus(None, 0.0, "none")

    # Both present — check agreement
    agree = _text_agree(ocr_val, vlm_val) or _text_partial_agree(ocr_val, vlm_val)

    if agree:
        # Use VLM value (typically cleaner formatting), boost confidence
        combined_conf = min(1.0, (ocr_conf + vlm_conf) / 2 + 0.1)
        logger.debug("%s: both agree → '%s' (conf=%.2f)", field_name, vlm_val, combined_conf)
        return FieldConsensus(vlm_val, combined_conf, "both")
    else:
        # Disagreement — use higher-confidence source
        if vlm_conf >= ocr_conf:
            winner, winner_conf = vlm_val, vlm_conf * 0.9  # Slight penalty for conflict
            source = "vlm"
        else:
            winner, winner_conf = ocr_val, ocr_conf * 0.9
            source = "ocr"
        logger.info(
            "%s conflict: ocr='%s' (%.2f) vs vlm='%s' (%.2f) → '%s'",
            field_name, ocr_val, ocr_conf, vlm_val, vlm_conf, winner,
        )
        return FieldConsensus(winner, winner_conf, source, conflicted=True)


def _merge_numeric_field(
    ocr_val: Optional[int],
    ocr_conf: float,
    vlm_val: Optional[int],
    vlm_conf: float,
    field_name: str,
    tolerance: float = 0.05,
) -> FieldConsensus:
    """Merge two optional numeric values."""
    both_present = ocr_val is not None and vlm_val is not None

    if not both_present:
        if vlm_val is not None:
            return FieldConsensus(vlm_val, vlm_conf, "vlm")
        if ocr_val is not None:
            return FieldConsensus(ocr_val, ocr_conf, "ocr")
        return FieldConsensus(None, 0.0, "none")

    if _numeric_agree(ocr_val, vlm_val, tolerance):
        avg = round((ocr_val + vlm_val) / 2)
        combined_conf = min(1.0, (ocr_conf + vlm_conf) / 2 + 0.1)
        logger.debug("%s: both agree → %d (conf=%.2f)", field_name, avg, combined_conf)
        return FieldConsensus(avg, combined_conf, "both")
    else:
        if vlm_conf >= ocr_conf:
            winner, winner_conf, source = vlm_val, vlm_conf * 0.85, "vlm"
        else:
            winner, winner_conf, source = ocr_val, ocr_conf * 0.85, "ocr"
        logger.info(
            "%s conflict: ocr=%s (%.2f) vs vlm=%s (%.2f) → %s",
            field_name, ocr_val, ocr_conf, vlm_val, vlm_conf, winner,
        )
        return FieldConsensus(winner, winner_conf, source, conflicted=True)


def _merge_bool_field(
    det_val: Optional[bool],
    det_conf: float,
    vlm_val: Optional[bool],
    vlm_conf: float,
    field_name: str,
) -> FieldConsensus:
    """Merge detection result (CV) with VLM result for stamp/signature."""
    both_present = det_val is not None and vlm_val is not None

    if not both_present:
        if det_val is not None:
            return FieldConsensus(det_val, det_conf, "detection")
        if vlm_val is not None:
            return FieldConsensus(vlm_val, vlm_conf, "vlm")
        return FieldConsensus(False, 0.5, "none")

    if det_val == vlm_val:
        combined_conf = min(1.0, (det_conf + vlm_conf) / 2 + 0.1)
        return FieldConsensus(det_val, combined_conf, "both")

    # Disagreement: detection (CV) generally more reliable for binary presence
    if det_conf >= vlm_conf:
        return FieldConsensus(det_val, det_conf * 0.9, "detection", conflicted=True)
    else:
        return FieldConsensus(vlm_val, vlm_conf * 0.9, "vlm", conflicted=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_consensus(
    ocr_fields: dict,
    ocr_confidences: dict,
    vlm_fields: dict,
    vlm_confidences: dict,
    detection_fields: dict,
    detection_confidences: dict,
) -> ConsensusResult:
    """
    Build a consensus result from three sources:
      - OCR rule-based parser
      - VLM (Qwen2.5-VL) extractor
      - CV detection (stamp + signature bounding boxes)

    Args:
        ocr_fields:              dict with keys: dealer_name, model_name, horse_power, asset_cost
        ocr_confidences:         per-field confidence from OCR parser
        vlm_fields:              dict with same keys + signature_present, stamp_present
        vlm_confidences:         per-field confidence from VLM
        detection_fields:        dict with keys: signature_present, stamp_present
        detection_confidences:   per-field confidence from CV detectors

    Returns:
        ConsensusResult
    """
    result = ConsensusResult()

    result.dealer_name = _merge_text_field(
        ocr_fields.get("dealer_name"), ocr_confidences.get("dealer_name", 0.5),
        vlm_fields.get("dealer_name"), vlm_confidences.get("dealer_name", 0.5),
        "dealer_name",
    )

    result.model_name = _merge_text_field(
        ocr_fields.get("model_name"), ocr_confidences.get("model_name", 0.5),
        vlm_fields.get("model_name"), vlm_confidences.get("model_name", 0.5),
        "model_name",
    )

    result.horse_power = _merge_numeric_field(
        ocr_fields.get("horse_power"), ocr_confidences.get("horse_power", 0.5),
        vlm_fields.get("horse_power"), vlm_confidences.get("horse_power", 0.5),
        "horse_power",
        tolerance=0.0,   # HP must match exactly
    )

    result.asset_cost = _merge_numeric_field(
        ocr_fields.get("asset_cost"), ocr_confidences.get("asset_cost", 0.5),
        vlm_fields.get("asset_cost"), vlm_confidences.get("asset_cost", 0.5),
        "asset_cost",
        tolerance=0.05,  # ±5% tolerance for cost
    )

    result.signature_present = _merge_bool_field(
        detection_fields.get("signature_present"),
        detection_confidences.get("signature_present", 0.5),
        vlm_fields.get("signature_present"),
        vlm_confidences.get("signature_present", 0.5),
        "signature_present",
    )

    result.stamp_present = _merge_bool_field(
        detection_fields.get("stamp_present"),
        detection_confidences.get("stamp_present", 0.5),
        vlm_fields.get("stamp_present"),
        vlm_confidences.get("stamp_present", 0.5),
        "stamp_present",
    )

    return result
