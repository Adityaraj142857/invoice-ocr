"""
matching/fuzzy_matcher.py
--------------------------
Fuzzy matching of extracted dealer/model names against master lists.

Uses RapidFuzz (C-extension, very fast) with multiple scorer strategies.

Dependencies:
    pip install rapidfuzz
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    original: Optional[str]       # Extracted value (before matching)
    matched: Optional[str]        # Best match from master list
    score: float                  # Match score 0–100
    matched_above_threshold: bool

    @property
    def normalised_score(self) -> float:
        """Score as 0.0–1.0."""
        return self.score / 100.0


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

def _import_rapidfuzz():
    try:
        from rapidfuzz import fuzz, process
        return fuzz, process
    except ImportError as exc:
        raise ImportError(
            "RapidFuzz not installed. Run: pip install rapidfuzz"
        ) from exc


def _preprocess(text: Optional[str]) -> str:
    """Lowercase, strip, collapse whitespace."""
    if not text:
        return ""
    import re
    text = re.sub(r"\s+", " ", str(text).lower().strip())
    # Remove common legal suffixes for matching
    for suffix in [
        r"\bpvt\.?\s*ltd\.?\b", r"\bltd\.?\b", r"\bco\.?\b",
        r"\bprivate\s+limited\b", r"\blimited\b", r"\btraders?\b",
        r"\bagenc(y|ies)\b", r"\benterprises?\b",
    ]:
        text = re.sub(suffix, "", text, flags=re.IGNORECASE).strip()
    return re.sub(r"\s+", " ", text).strip()


def match_dealer_name(
    extracted: Optional[str],
    master_list: list[str],
    threshold: float = 80.0,
) -> MatchResult:
    """
    Match an extracted dealer name against the master list.

    Uses a combination of:
      - WRatio (handles abbreviations and partial matches)
      - token_sort_ratio (handles word-order differences)
      - token_set_ratio (handles subset matching)

    Args:
        extracted:   Raw extracted dealer name.
        master_list: List of canonical dealer names.
        threshold:   Minimum score (0–100) to declare a match.

    Returns:
        MatchResult
    """
    if not extracted or not master_list:
        return MatchResult(extracted, None, 0.0, False)

    fuzz, process = _import_rapidfuzz()
    query = _preprocess(extracted)
    choices = {name: _preprocess(name) for name in master_list}

    best_score = 0.0
    best_name = None

    for canonical, preprocessed in choices.items():
        scores = [
            fuzz.WRatio(query, preprocessed),
            fuzz.token_sort_ratio(query, preprocessed),
            fuzz.token_set_ratio(query, preprocessed),
        ]
        score = max(scores)
        if score > best_score:
            best_score = score
            best_name = canonical

    matched = best_name if best_score >= threshold else None
    logger.debug(
        "Dealer match: '%s' → '%s' (score=%.1f, threshold=%.1f)",
        extracted, matched, best_score, threshold,
    )
    return MatchResult(
        original=extracted,
        matched=matched,
        score=best_score,
        matched_above_threshold=best_score >= threshold,
    )


def match_model_name(
    extracted: Optional[str],
    master_list: list[str],
    threshold: float = 85.0,
) -> MatchResult:
    """
    Match an extracted model name against the master list.

    Model names require a higher threshold and token-based matching
    because word order often differs (e.g. "742 XT Swaraj" vs "Swaraj 742 XT").

    Args:
        extracted:   Raw extracted model name.
        master_list: List of canonical model names.
        threshold:   Minimum score (0–100). Model matching is stricter.

    Returns:
        MatchResult
    """
    if not extracted or not master_list:
        return MatchResult(extracted, None, 0.0, False)

    fuzz, process = _import_rapidfuzz()
    query = _preprocess(extracted)
    choices = {name: _preprocess(name) for name in master_list}

    best_score = 0.0
    best_name = None

    for canonical, preprocessed in choices.items():
        scores = [
            fuzz.ratio(query, preprocessed),
            fuzz.token_sort_ratio(query, preprocessed),
            fuzz.partial_ratio(query, preprocessed),
        ]
        score = max(scores)
        if score > best_score:
            best_score = score
            best_name = canonical

    matched = best_name if best_score >= threshold else None
    logger.debug(
        "Model match: '%s' → '%s' (score=%.1f, threshold=%.1f)",
        extracted, matched, best_score, threshold,
    )
    return MatchResult(
        original=extracted,
        matched=matched,
        score=best_score,
        matched_above_threshold=best_score >= threshold,
    )


def match_all_fields(
    dealer_name: Optional[str],
    model_name: Optional[str],
    master_data: object,    # MasterData — avoids circular import
    dealer_threshold: float = 80.0,
    model_threshold: float = 85.0,
) -> dict[str, MatchResult]:
    """
    Run fuzzy matching on both dealer and model names.

    Args:
        dealer_name:       Extracted dealer name.
        model_name:        Extracted model name.
        master_data:       MasterData instance.
        dealer_threshold:  Score threshold for dealer (0–100).
        model_threshold:   Score threshold for model (0–100).

    Returns:
        Dict with keys 'dealer' and 'model', values are MatchResult.
    """
    dealer_result = match_dealer_name(
        dealer_name, master_data.dealer_names, dealer_threshold
    )
    model_result = match_model_name(
        model_name, master_data.model_names, model_threshold
    )
    return {"dealer": dealer_result, "model": model_result}
