"""
utils/json_utils.py
--------------------
Output JSON builder matching the hackathon submission format exactly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional


_REQUIRED_KEYS = {
    "doc_id", "fields", "confidence", "processing_time_sec", "cost_estimate_usd"
}

_FIELD_KEYS = {
    "dealer_name", "model_name", "horse_power",
    "asset_cost", "signature", "stamp",
}


def build_output_json(
    doc_id: str,
    dealer_name: Optional[str],
    model_name: Optional[str],
    horse_power: Optional[int],
    asset_cost: Optional[int],
    signature_present: Optional[bool],
    signature_bbox: Optional[list[int]],
    stamp_present: Optional[bool],
    stamp_bbox: Optional[list[int]],
    overall_confidence: float,
    processing_time_sec: float,
    cost_estimate_usd: float = 0.0,
) -> dict[str, Any]:
    """
    Build the output JSON object for one document.

    Args:
        doc_id:               Document identifier (filename without extension).
        dealer_name:          Extracted dealer name string or None.
        model_name:           Extracted model name string or None.
        horse_power:          Extracted HP integer or None.
        asset_cost:           Extracted cost integer or None.
        signature_present:    True/False/None.
        signature_bbox:       [x_min, y_min, x_max, y_max] or None.
        stamp_present:        True/False/None.
        stamp_bbox:           [x_min, y_min, x_max, y_max] or None.
        overall_confidence:   Aggregate confidence score 0.0–1.0.
        processing_time_sec:  Wall-clock time to process this document.
        cost_estimate_usd:    Estimated inference cost (0 for local models).

    Returns:
        Dict matching the hackathon output spec.
    """
    return {
        "doc_id": str(doc_id),
        "fields": {
            "dealer_name": dealer_name,
            "model_name": model_name,
            "horse_power": horse_power,
            "asset_cost": asset_cost,
            "signature": {
                "present": bool(signature_present) if signature_present is not None else False,
                "bbox": signature_bbox,
            },
            "stamp": {
                "present": bool(stamp_present) if stamp_present is not None else False,
                "bbox": stamp_bbox,
            },
        },
        "confidence": round(float(overall_confidence), 4),
        "processing_time_sec": round(float(processing_time_sec), 3),
        "cost_estimate_usd": round(float(cost_estimate_usd), 6),
    }


def validate_output(record: dict) -> tuple[bool, list[str]]:
    """
    Validate a single output record against the submission spec.

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: list[str] = []

    missing_top = _REQUIRED_KEYS - set(record.keys())
    if missing_top:
        errors.append(f"Missing top-level keys: {missing_top}")
        return False, errors

    fields = record.get("fields", {})
    missing_fields = _FIELD_KEYS - set(fields.keys())
    if missing_fields:
        errors.append(f"Missing field keys: {missing_fields}")

    # HP must be int or null
    hp = fields.get("horse_power")
    if hp is not None and not isinstance(hp, int):
        errors.append(f"horse_power must be int, got {type(hp).__name__}")

    # asset_cost must be int or null
    cost = fields.get("asset_cost")
    if cost is not None and not isinstance(cost, int):
        errors.append(f"asset_cost must be int, got {type(cost).__name__}")

    # Signature bbox must be list of 4 ints or null
    sig = fields.get("signature", {})
    if isinstance(sig, dict):
        sig_bbox = sig.get("bbox")
        if sig_bbox is not None:
            if not (isinstance(sig_bbox, list) and len(sig_bbox) == 4):
                errors.append("signature.bbox must be [x1,y1,x2,y2] or null")

    # Confidence must be 0–1
    conf = record.get("confidence", -1)
    if not (0.0 <= float(conf) <= 1.0):
        errors.append(f"confidence must be 0–1, got {conf}")

    return len(errors) == 0, errors


def save_results(
    results: list[dict],
    output_path: str | Path,
    pretty: bool = True,
) -> None:
    """
    Save a list of output records to a JSON file.

    Args:
        results:     List of output dicts from build_output_json().
        output_path: Path to write the result.json file.
        pretty:      If True, indent the JSON for readability.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            json.dump(results, f, ensure_ascii=False)


def doc_id_from_path(file_path: str | Path) -> str:
    """Extract a clean doc_id from a file path."""
    return Path(file_path).stem


def load_results(json_path: str | Path) -> list[dict]:
    """Load and return a previously saved results JSON file."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return data
