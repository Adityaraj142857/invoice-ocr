"""
extraction/confidence.py
------------------------
Per-field and document-level confidence scoring.

Combines:
  - Source agreement signals from consensus
  - OCR block confidence scores
  - Field sanity checks (HP range, cost range, name plausibility)
  - Presence of expected document structure
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ConfidenceReport:
    dealer_name: float = 0.0
    model_name: float = 0.0
    horse_power: float = 0.0
    asset_cost: float = 0.0
    signature_present: float = 0.0
    stamp_present: float = 0.0
    overall: float = 0.0

    def to_dict(self) -> dict:
        return {
            "dealer_name": round(self.dealer_name, 3),
            "model_name": round(self.model_name, 3),
            "horse_power": round(self.horse_power, 3),
            "asset_cost": round(self.asset_cost, 3),
            "signature_present": round(self.signature_present, 3),
            "stamp_present": round(self.stamp_present, 3),
            "overall": round(self.overall, 3),
        }


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _hp_plausible(hp: Any) -> float:
    """Returns confidence bonus if HP is in a realistic tractor range."""
    if hp is None:
        return 0.0
    try:
        hp_int = int(hp)
    except (TypeError, ValueError):
        return 0.0
    if 15 <= hp_int <= 100:
        return 0.1   # Bonus for plausible range
    return -0.2      # Penalty for implausible


def _cost_plausible(cost: Any) -> float:
    """Returns confidence bonus if cost is in a realistic tractor price range."""
    if cost is None:
        return 0.0
    try:
        cost_int = int(cost)
    except (TypeError, ValueError):
        return 0.0
    if 100_000 <= cost_int <= 30_000_000:
        return 0.1
    return -0.2


def _name_plausible(name: Optional[str]) -> float:
    """Returns confidence bonus if name looks like a real business name."""
    if not name:
        return 0.0
    if len(name) < 3 or len(name) > 120:
        return -0.1
    # Penalise if it's all digits
    if re.match(r"^\d+$", name.strip()):
        return -0.3
    return 0.05


def _model_plausible(model: Optional[str]) -> float:
    """Returns confidence bonus if model name contains alphanumeric content."""
    if not model:
        return 0.0
    has_alpha = bool(re.search(r"[A-Za-z]", model))
    has_digit = bool(re.search(r"\d", model))
    if has_alpha and has_digit:
        return 0.1   # Most model names have both letters and numbers
    return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_confidence(
    consensus_result: Any,   # ConsensusResult — avoid circular import
) -> ConfidenceReport:
    """
    Compute final confidence scores for all fields.

    Args:
        consensus_result: ConsensusResult from consensus.build_consensus()

    Returns:
        ConfidenceReport with per-field and overall scores.
    """
    report = ConfidenceReport()

    # Dealer name
    base = consensus_result.dealer_name.confidence
    report.dealer_name = max(0.0, min(1.0,
        base + _name_plausible(consensus_result.dealer_name.value)
    ))

    # Model name
    base = consensus_result.model_name.confidence
    report.model_name = max(0.0, min(1.0,
        base + _model_plausible(consensus_result.model_name.value)
    ))

    # HP
    base = consensus_result.horse_power.confidence
    report.horse_power = max(0.0, min(1.0,
        base + _hp_plausible(consensus_result.horse_power.value)
    ))

    # Asset cost
    base = consensus_result.asset_cost.confidence
    report.asset_cost = max(0.0, min(1.0,
        base + _cost_plausible(consensus_result.asset_cost.value)
    ))

    # Signature and stamp (direct from consensus)
    report.signature_present = max(0.0, min(1.0,
        consensus_result.signature_present.confidence
    ))
    report.stamp_present = max(0.0, min(1.0,
        consensus_result.stamp_present.confidence
    ))

    # Overall: mean of all fields, penalise if any is null
    scores = [
        report.dealer_name, report.model_name, report.horse_power,
        report.asset_cost, report.signature_present, report.stamp_present,
    ]
    values = [
        consensus_result.dealer_name.value, consensus_result.model_name.value,
        consensus_result.horse_power.value, consensus_result.asset_cost.value,
        consensus_result.signature_present.value, consensus_result.stamp_present.value,
    ]
    null_penalty = sum(0.05 for v in values if v is None)
    report.overall = max(0.0, min(1.0,
        sum(scores) / len(scores) - null_penalty
    ))

    return report
