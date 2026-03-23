"""
vlm/qwen_extractor.py
---------------------
Qwen2-VL vision-language model wrapper.
Cross-platform: auto-detects CUDA / MPS / CPU.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*top_k.*")
warnings.filterwarnings("ignore", message=".*generation flags.*")
warnings.filterwarnings("ignore", message=".*fast processor.*")
warnings.filterwarnings("ignore", message=".*image processor.*")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from dataclasses import dataclass
from typing import Any, Optional

from PIL import Image

from .prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT,
    FALLBACK_SYSTEM_PROMPT,
    FALLBACK_USER_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
    VERIFICATION_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class VLMExtractionResult:
    dealer_name: Optional[str] = None
    model_name: Optional[str] = None
    horse_power: Optional[int] = None
    asset_cost: Optional[int] = None
    signature_present: Optional[bool] = None
    stamp_present: Optional[bool] = None
    raw_response: str = ""
    latency_sec: float = 0.0
    parse_success: bool = False
    error: str = ""
    fields_found: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dealer_name": self.dealer_name,
            "model_name": self.model_name,
            "horse_power": self.horse_power,
            "asset_cost": self.asset_cost,
            "signature_present": self.signature_present,
            "stamp_present": self.stamp_present,
        }


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

_model_cache: dict[str, tuple[Any, Any]] = {}


def _load_model(model_id: str, device: str, dtype_str: str) -> tuple[Any, Any]:
    if model_id in _model_cache:
        return _model_cache[model_id]

    try:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Missing packages. Run:\n  pip install transformers accelerate torch torchvision"
        ) from exc

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)

    logger.info("Loading model '%s' on %s (%s)...", model_id, device, dtype_str)
    t0 = time.time()

    # MPS does not support device_map — must load then move
    if device == "mps":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to("mps")
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
        )

    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model.eval()

    logger.info("Model loaded in %.1f s", time.time() - t0)
    _model_cache[model_id] = (model, processor)
    return model, processor


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    return re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()


def _parse_json(raw: str) -> dict[str, Any]:
    raw = _strip_markdown(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _coerce_str(val: Any) -> Optional[str]:
    if val is None or str(val).lower() in ("null", "none", "n/a", ""):
        return None
    return str(val).strip()


def _coerce_int(val: Any) -> Optional[int]:
    if val is None or str(val).lower() in ("null", "none", "n/a", ""):
        return None
    cleaned = re.sub(r"[,₹Rs.\s\-/]", "", str(val))
    try:
        return int(float(cleaned))
    except (ValueError, TypeError):
        return None


def _coerce_bool(val: Any) -> Optional[bool]:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    sv = str(val).lower().strip()
    if sv in ("true", "yes", "1", "present"):
        return True
    if sv in ("false", "no", "0", "absent", "not present"):
        return False
    return None


def _coerce_fields(data: dict) -> dict:
    return {
        "dealer_name":      _coerce_str(data.get("dealer_name")),
        "model_name":       _coerce_str(data.get("model_name")),
        "horse_power":      _coerce_int(data.get("horse_power")),
        "asset_cost":       _coerce_int(data.get("asset_cost")),
        "signature_present":_coerce_bool(data.get("signature_present")),
        "stamp_present":    _coerce_bool(data.get("stamp_present")),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(
    model: Any,
    processor: Any,
    image: Image.Image,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    device: str,
) -> tuple[str, float]:
    import torch

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": f"{system_prompt}\n\n{user_prompt}"},
        ],
    }]

    try:
        from qwen_vl_utils import process_vision_info
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
    except ImportError:
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_input], images=[image], padding=True, return_tensors="pt"
        )

    # Move inputs to device — MPS needs careful dtype handling
    if device == "mps":
        inputs = {
            k: v.to(device) if v.is_floating_point() else v.to(device)
            for k, v in inputs.items()
        }
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    latency = time.time() - t0

    input_len = inputs["input_ids"].shape[1]
    response = processor.batch_decode(
        output_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return response.strip(), latency


# ---------------------------------------------------------------------------
# Public extractor
# ---------------------------------------------------------------------------

@dataclass
class QwenExtractorConfig:
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 256
    run_verification: bool = False
    fallback_on_failure: bool = True


class QwenExtractor:
    def __init__(self, cfg: Optional[QwenExtractorConfig] = None):
        self.cfg = cfg or QwenExtractorConfig()
        self._model = None
        self._processor = None
        self._resolved_device: Optional[str] = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            # Resolve auto device/dtype
            from utils.device_utils import resolve_device, resolve_dtype
            device = resolve_device(self.cfg.device)
            dtype  = resolve_dtype(self.cfg.dtype, device)
            self._resolved_device = device
            self._model, self._processor = _load_model(
                self.cfg.model_id, device, dtype
            )

    def _pass(self, image: Image.Image, system: str, user: str) -> tuple[dict, str, float]:
        self._ensure_loaded()
        raw, latency = _run_inference(
            self._model, self._processor, image,
            system, user,
            self.cfg.max_new_tokens,
            self._resolved_device,
        )
        return _coerce_fields(_parse_json(raw)), raw, latency

    def extract(self, image: Image.Image) -> VLMExtractionResult:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")

        total_latency = 0.0
        data, raw, parse_success = {}, "", False

        try:
            data, raw, latency = self._pass(image, EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_PROMPT)
            total_latency += latency
            parse_success = any(v is not None for v in data.values())
        except Exception as exc:
            logger.error("Primary VLM pass failed: %s", exc)

        if not parse_success and self.cfg.fallback_on_failure:
            try:
                data, raw, latency = self._pass(image, FALLBACK_SYSTEM_PROMPT, FALLBACK_USER_PROMPT)
                total_latency += latency
                parse_success = any(v is not None for v in data.values())
            except Exception as exc:
                logger.error("Fallback VLM pass failed: %s", exc)

        if parse_success and self.cfg.run_verification:
            try:
                verified, _, lv = self._pass(
                    image,
                    VERIFICATION_SYSTEM_PROMPT,
                    VERIFICATION_USER_TEMPLATE.format(
                        extracted_json=json.dumps(data, ensure_ascii=False)
                    ),
                )
                total_latency += lv
                for k in data:
                    if verified.get(k) is not None:
                        data[k] = verified[k]
            except Exception as exc:
                logger.warning("Verification pass failed (non-fatal): %s", exc)

        fields_found = sum(1 for v in data.values() if v is not None)
        return VLMExtractionResult(
            dealer_name=data.get("dealer_name"),
            model_name=data.get("model_name"),
            horse_power=data.get("horse_power"),
            asset_cost=data.get("asset_cost"),
            signature_present=data.get("signature_present"),
            stamp_present=data.get("stamp_present"),
            raw_response=raw,
            latency_sec=total_latency,
            parse_success=parse_success,
            fields_found=fields_found,
        )
