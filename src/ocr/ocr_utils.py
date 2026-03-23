"""
ocr/ocr_utils.py
----------------
Text cleaning, normalisation and post-processing utilities for raw OCR output.

Handles:
  - Hindi/Gujarati numeral → Arabic numeral mapping
  - Common OCR error corrections (O↔0, I↔1, etc.)
  - Currency / amount string normalisation
  - HP string extraction (English + Hindi variants)
  - Whitespace / punctuation cleanup
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# ---------------------------------------------------------------------------
# Numeral mappings
# ---------------------------------------------------------------------------

# Devanagari (Hindi) digits: ०१२३४५६७८९
_DEVANAGARI_DIGITS = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}

# Gujarati digits: ૦૧૨૩૪૫૬૭૮૯
_GUJARATI_DIGITS = {
    "૦": "0", "૧": "1", "૨": "2", "૩": "3", "૪": "4",
    "૫": "5", "૬": "6", "૭": "7", "૮": "8", "૯": "9",
}

_ALL_NON_ASCII_DIGITS = {**_DEVANAGARI_DIGITS, **_GUJARATI_DIGITS}

# ---------------------------------------------------------------------------
# OCR character confusion corrections
# ---------------------------------------------------------------------------

# Applied only when extracting purely numeric fields (HP, cost)
_NUMERIC_OCR_FIXES = {
    "O": "0", "o": "0",
    "I": "1", "l": "1",   # lowercase L
    "S": "5",
    "B": "8",
    "G": "6",
    "Z": "2",
    "?": "7",
    " ": "",
}

# ---------------------------------------------------------------------------
# HP keyword patterns  (English + Hindi transliterations)
# ---------------------------------------------------------------------------

# Matches: "50 HP", "50HP", "50 H.P.", "50H.P", "हा.पा. 50", "हा.पा.45",
#          "H.P....50", "HP....40", "45 hp", "45-HP"
_HP_PATTERNS = [
    # Number BEFORE the HP keyword
    r"(\d{1,3})\s*[-–]?\s*[Hh]\.?\s*[Pp]\.?",
    # HP keyword BEFORE the number (Hindi style or English label)
    r"[Hh]\.?\s*[Pp]\.?\s*[-:.\s]*\s*(\d{1,3})",
    # Hindi: हा.पा. or हा पा
    r"हा\.?\s*पा\.?\s*[-:.\s]*\s*(\d{1,3})",
    # Explicit label "Horse Power" followed by number
    r"[Hh]orse\s+[Pp]ower\s*[-:.\s]*\s*(\d{1,3})",
]

# ---------------------------------------------------------------------------
# Amount / cost patterns
# ---------------------------------------------------------------------------

# Matches Indian number formats: 7,00,000 / 7,00,000/- / 700000 / 8,01,815.00
_AMOUNT_CLEANUP_RE = re.compile(r"[^\d.,]")
_TRAILING_JUNK_RE = re.compile(r"[,.\-/\\]+$")


# ===========================================================================
# Public utilities
# ===========================================================================

def normalise_digits(text: str) -> str:
    """
    Replace Devanagari and Gujarati digits with ASCII equivalents.
    Leaves all other characters unchanged.
    """
    for native, ascii_digit in _ALL_NON_ASCII_DIGITS.items():
        text = text.replace(native, ascii_digit)
    return text


def clean_text(text: str) -> str:
    """
    General-purpose text cleanup:
      1. Normalise unicode (NFC)
      2. Replace native digits
      3. Strip leading/trailing whitespace
      4. Collapse multiple spaces
    """
    text = unicodedata.normalize("NFC", text)
    text = normalise_digits(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_hp(text: str) -> Optional[int]:
    """
    Extract a horse-power value from an OCR text string.

    Handles English and Hindi variants, common OCR confusions, and
    different orderings (number before/after 'HP').

    Returns:
        Integer HP value, or None if not found.

    Examples:
        "Powertrac Euro G28  HP....28" → 28
        "हा.पा. 45"                   → 45
        "55 HP tractor"               → 55
        "40HP 3CYL"                   → 40
    """
    text = clean_text(text)

    for pattern in _HP_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE | re.UNICODE)
        if match:
            raw = match.group(1)
            # Apply numeric OCR fixes
            for bad, good in _NUMERIC_OCR_FIXES.items():
                raw = raw.replace(bad, good)
            try:
                val = int(raw)
                if 5 <= val <= 200:   # Sanity range for tractor HP
                    return val
            except ValueError:
                continue
    return None


def extract_amount(text: str) -> Optional[int]:
    """
    Extract a numeric rupee amount from a text string.

    Strips:
      - Currency symbols (₹, Rs, Rs., रु, रु.)
      - Indian comma separators (7,00,000 → 700000)
      - Trailing /- or decimals (.00)
      - Leading/trailing junk

    Returns:
        Integer amount, or None if extraction fails.

    Examples:
        "Rs. 7,00,000/-"   → 700000
        "₹8,01,815.00"    → 801815
        "550,000.00"       → 550000
        "रु.6,50,000/-"    → 650000
    """
    text = clean_text(text)

    # Remove currency symbols
    text = re.sub(r"[₹Rr][sS]\.?\s*", "", text)
    text = re.sub(r"रु\.?\s*", "", text)

    # Keep only digits, commas, dots, minus
    text = _AMOUNT_CLEANUP_RE.sub("", text)

    # Remove trailing junk (/-  ,  .)
    text = _TRAILING_JUNK_RE.sub("", text)

    if not text:
        return None

    # Remove commas (Indian grouping)
    text = text.replace(",", "")

    # Remove decimal part (e.g. .00)
    if "." in text:
        text = text.split(".")[0]

    # Apply OCR character corrections
    for bad, good in _NUMERIC_OCR_FIXES.items():
        text = text.replace(bad, good)

    try:
        val = int(text)
        if val > 0:
            return val
    except ValueError:
        pass
    return None


def extract_all_amounts(text: str) -> list[int]:
    """
    Find ALL numeric amounts in a text string.
    Useful when a quotation lists multiple prices — caller picks the total.

    Returns list of integers sorted descending (largest/total amount first).
    """
    # Pattern: 1–8 digit groups separated by Indian commas
    pattern = re.compile(
        r"\b\d{1,3}(?:[,\s]\d{2,3})*(?:\.\d{1,2})?/?\-?\b"
    )
    amounts = []
    for match in pattern.finditer(clean_text(text)):
        val = extract_amount(match.group())
        if val is not None and val >= 1000:   # Minimum plausible invoice amount
            amounts.append(val)
    return sorted(set(amounts), reverse=True)


def normalise_model_name(text: str) -> str:
    """
    Normalise a tractor model name string:
      - Collapse whitespace
      - Uppercase
      - Remove OCR noise characters
      - Keep alphanumerics, spaces, hyphens, dots

    Examples:
        "Powertrac  Euro  G28"    → "POWERTRAC EURO G28"
        "Swaraj 742 X T"          → "SWARAJ 742 XT"
        "John  Deere   5042D"     → "JOHN DEERE 5042D"
    """
    text = clean_text(text)
    # Remove characters that are not alphanumeric, space, hyphen, or dot
    text = re.sub(r"[^\w\s\-.]", "", text)
    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip().upper()
    return text


def normalise_dealer_name(text: str) -> str:
    """
    Normalise a dealer name for fuzzy matching:
      - Strip common suffixes that vary (Pvt Ltd, Co., etc.)
      - Collapse whitespace
      - Title case

    Examples:
        "MAAN TRACTOR CO."         → "Maan Tractor"
        "AMS TRACTORS PVT. LTD."   → "Ams Tractors"
        "MK.MOTORS"                → "Mk Motors"
    """
    text = clean_text(text)
    # Replace dots that act as word separators
    text = re.sub(r"\.(?=[A-Za-z])", ". ", text)
    # Remove common legal suffixes
    text = re.sub(
        r"\b(pvt\.?\s*ltd\.?|ltd\.?|private\s+limited|limited|co\.?|"
        r"company|enterprises?|traders?|agency|agencies|prop\.?)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip()
    # Remove trailing punctuation
    text = text.rstrip(".,/\\-")
    return text.title()


def remove_pii_noise(text: str) -> str:
    """
    Remove patterns that are likely PII placeholders (grey boxes in images
    appear as garbled OCR text) or clearly not field content.

    In the provided dataset, PII is redacted with grey rectangles.
    OCR on these produces junk strings — this removes them.
    """
    # Long runs of same character (grey box artefact: "████" → "■■■■")
    text = re.sub(r"(.)\1{4,}", "", text)
    # Remove standalone single characters that are not digits or letters
    text = re.sub(r"(?<!\w)[^A-Za-z0-9\u0900-\u097F\u0A80-\u0AFF](?!\w)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_likely_total_row(text: str) -> bool:
    """
    Heuristic: does this text look like it's from a 'Total' row?
    Used when selecting which amount in a multi-row table is the asset cost.
    """
    keywords = [
        "total", "grand total", "योग", "कुल", "जमा", "amount", "net amount",
        "grand", "sum",
    ]
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def split_lines(full_text: str) -> list[str]:
    """Split OCR full_text into non-empty lines."""
    return [ln.strip() for ln in re.split(r"[\n\r]+", full_text) if ln.strip()]


def contains_idfc(text: str) -> bool:
    """Check if text contains IDFC First Bank reference (hypothecation line)."""
    text_lower = text.lower()
    return (
        "idfc" in text_lower
        or "idfcfirst" in text_lower
        or ("first" in text_lower and "bank" in text_lower)
    )
