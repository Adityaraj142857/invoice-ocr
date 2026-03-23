"""
vlm/prompts.py
--------------
All prompt templates for Qwen2.5-VL invoice field extraction.

Keeping prompts in one file makes iteration easy without touching model code.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Primary extraction prompt  (JSON output)
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are a document AI specialising in extracting structured fields from Indian tractor loan quotation documents. These documents may be in English, Hindi, Gujarati, or a mix of scripts.

Your task is to extract exactly six fields and return them as a JSON object with no other text, no markdown, no explanation.

FIELD RULES:
1. dealer_name: The company/shop name at the TOP of the document (letterhead). Do not include customer name. Return exactly as printed.
2. model_name: The tractor model identifier (e.g. "Mahindra 575 DI", "Swaraj 742 XT", "John Deere 5042D"). If multiple models are listed, return only the one that has a price/amount filled in next to it.
3. horse_power: Integer only. Extract the HP / H.P. / हा.पा. value. Return as a number, not a string.
4. asset_cost: The TOTAL amount / Grand Total in Indian Rupees. Return as integer with no commas, no symbols, no decimals. E.g. 7,00,000/- → 700000.
5. signature_present: true if there is a handwritten signature (cursive ink mark) near "Authorised Signatory" or "For [Dealer Name]". false otherwise.
6. stamp_present: true if there is a circular or oval rubber ink stamp on the document. Printed logos do NOT count as stamps. false otherwise.

IMPORTANT:
- If a field cannot be determined, use null.
- Return ONLY the JSON object. No markdown fences. No explanation.
- Numbers must be JSON numbers (not strings): horse_power: 45, not "45".

OUTPUT FORMAT:
{
  "dealer_name": "...",
  "model_name": "...",
  "horse_power": <integer or null>,
  "asset_cost": <integer or null>,
  "signature_present": <true or false>,
  "stamp_present": <true or false>
}"""


EXTRACTION_USER_PROMPT = """Extract the six fields from this invoice/quotation image and return only the JSON object as described."""


# ---------------------------------------------------------------------------
# Self-check / verification prompt
# ---------------------------------------------------------------------------

VERIFICATION_SYSTEM_PROMPT = """You are a quality-checking AI. You will be given:
1. An invoice image
2. A previously extracted JSON

Your job is to verify each field and return a corrected JSON if any field is wrong, or the same JSON if everything is correct.

Rules:
- Only change a field if you are confident it is wrong.
- Return ONLY the corrected JSON. No explanation. No markdown.
- Keep the same format as the input JSON."""

VERIFICATION_USER_TEMPLATE = """Previously extracted JSON:
{extracted_json}

Please verify and correct if needed. Return only the JSON."""


# ---------------------------------------------------------------------------
# Fallback prompt for hard documents (low quality / heavily rotated)
# ---------------------------------------------------------------------------

FALLBACK_SYSTEM_PROMPT = """You are extracting data from a low-quality or difficult invoice scan. The image may be blurry, rotated, or have mixed languages. Do your best.

Extract what you can. If a field is truly unreadable, use null.
Return ONLY a JSON object with these keys:
dealer_name, model_name, horse_power, asset_cost, signature_present, stamp_present

No explanation. No markdown. JSON only."""

FALLBACK_USER_PROMPT = """Extract fields from this difficult invoice image. Return only JSON."""


# ---------------------------------------------------------------------------
# Confidence estimation prompt
# ---------------------------------------------------------------------------

CONFIDENCE_SYSTEM_PROMPT = """You are a confidence scoring AI. Given an invoice image and an extracted JSON, rate your confidence in each field from 0.0 to 1.0.

Return ONLY a JSON object with the same keys but confidence scores as values.
Example: {"dealer_name": 0.95, "model_name": 0.88, "horse_power": 1.0, "asset_cost": 0.92, "signature_present": 0.85, "stamp_present": 0.70}

No explanation. No markdown. JSON only."""

CONFIDENCE_USER_TEMPLATE = """Extracted JSON:
{extracted_json}

Rate your confidence in each field (0.0 to 1.0). Return only JSON."""


def build_extraction_messages(include_system: bool = True) -> list[dict]:
    """
    Build the messages list for the primary extraction call.
    The image is injected by the caller.
    """
    messages = []
    if include_system:
        messages.append({"role": "system", "content": EXTRACTION_SYSTEM_PROMPT})
    return messages


def build_verification_messages(extracted_json: str) -> list[dict]:
    """Build messages for the self-verification pass."""
    return [
        {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": VERIFICATION_USER_TEMPLATE.format(
            extracted_json=extracted_json
        )},
    ]


def build_confidence_messages(extracted_json: str) -> list[dict]:
    """Build messages for the confidence scoring pass."""
    return [
        {"role": "system", "content": CONFIDENCE_SYSTEM_PROMPT},
        {"role": "user", "content": CONFIDENCE_USER_TEMPLATE.format(
            extracted_json=extracted_json
        )},
    ]
