"""
ocr/paddle_ocr.py
-----------------
PaddleOCR wrapper — compatible with PaddleOCR 2.x and 3.x APIs.
Suppresses all PaddleOCR internal logs and warnings.
"""

from __future__ import annotations

import logging
import os
import warnings

# Suppress all paddle / urllib3 / requests warnings before anything loads
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_call_stack_level"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*chardet.*")
warnings.filterwarnings("ignore", message=".*charset_normalizer.*")
warnings.filterwarnings("ignore", message=".*RequestsDependency.*")

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Silence paddleocr internal loggers
for _noisy in ("ppocr", "paddle", "paddleocr", "ppstructure"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    text: str
    confidence: float
    bbox: list[int]
    raw_bbox: list[list[int]]

    @property
    def x_min(self) -> int: return self.bbox[0]
    @property
    def y_min(self) -> int: return self.bbox[1]
    @property
    def x_max(self) -> int: return self.bbox[2]
    @property
    def y_max(self) -> int: return self.bbox[3]
    @property
    def centre_y(self) -> float: return (self.y_min + self.y_max) / 2


@dataclass
class OCRResult:
    blocks: list[TextBlock] = field(default_factory=list)
    full_text: str = ""
    language_hint: str = "unknown"
    image_shape: tuple[int, int] = (0, 0)

    def get_text_near(self, keyword: str, search_radius_px: int = 200, case_sensitive: bool = False) -> list[TextBlock]:
        kw = keyword if case_sensitive else keyword.lower()
        anchors = [b for b in self.blocks if kw in (b.text if case_sensitive else b.text.lower())]
        if not anchors:
            return []
        nearby = []
        for anchor in anchors:
            for blk in self.blocks:
                if blk is anchor:
                    continue
                dx = abs((blk.x_min + blk.x_max) / 2 - (anchor.x_min + anchor.x_max) / 2)
                dy = abs(blk.centre_y - anchor.centre_y)
                if dx <= search_radius_px and dy <= search_radius_px:
                    nearby.append(blk)
        return nearby

    def get_line_groups(self, line_tolerance_px: int = 12) -> list[list[TextBlock]]:
        if not self.blocks:
            return []
        sorted_blocks = sorted(self.blocks, key=lambda b: b.centre_y)
        lines, current = [], [sorted_blocks[0]]
        for blk in sorted_blocks[1:]:
            if abs(blk.centre_y - current[-1].centre_y) <= line_tolerance_px:
                current.append(blk)
            else:
                lines.append(sorted(current, key=lambda b: b.x_min))
                current = [blk]
        if current:
            lines.append(sorted(current, key=lambda b: b.x_min))
        return lines

    def get_region_text(self, x_min: int, y_min: int, x_max: int, y_max: int) -> str:
        return " ".join(
            b.text for b in self.blocks
            if x_min <= (b.x_min + b.x_max)/2 <= x_max
            and y_min <= (b.y_min + b.y_max)/2 <= y_max
        )


# ---------------------------------------------------------------------------
# Engine — handles PaddleOCR 2.x and 3.x APIs
# ---------------------------------------------------------------------------

_ocr_engines: dict[str, object] = {}


def _build_engine(lang: str, use_angle_cls: bool) -> object:
    try:
        #from paddleocr import PaddleOCR
        from paddleocr import PaddleOCRVL
        pipeline = PaddleOCRVL(pipeline_version="v1")
    except ImportError as exc:
        raise ImportError("Run: pip install paddlepaddle paddleocr") from exc

    import inspect
    sig = inspect.signature(PaddleOCRVL.__init__)
    accepted = set(sig.parameters.keys())

    kwargs: dict = {"lang": lang, "use_angle_cls": use_angle_cls}

    if "show_log" in accepted:
        kwargs["show_log"] = False
    if "use_gpu" in accepted:
        kwargs["use_gpu"] = False
    if "enable_mkldnn" in accepted:
        kwargs["enable_mkldnn"] = False
    if "device" in accepted and "use_gpu" not in accepted:
        kwargs["device"] = "cpu"

    try:
        return PaddleOCRVL(**kwargs)
    except TypeError:
        logger.warning("PaddleOCR full-kwargs init failed, retrying minimal.")
        return PaddleOCRVL(lang=lang)


def _get_engine(lang: str = "en", use_angle_cls: bool = True) -> object:
    key = f"{lang}_{use_angle_cls}"
    if key not in _ocr_engines:
        logger.info("Initialising PaddleOCR engine (lang=%s) ...", lang)
        _ocr_engines[key] = _build_engine(lang, use_angle_cls)
        logger.info("PaddleOCR engine ready (lang=%s)", lang)
    return _ocr_engines[key]


# ---------------------------------------------------------------------------
# Runner — handles 2.x .ocr() and 3.x .predict() APIs
# ---------------------------------------------------------------------------

def _run_engine(engine: object, image_arr: np.ndarray, cls: bool = True) -> list:
    # Try 2.x API
    try:
        result = engine.ocr(image_arr, cls=cls)
        if result and isinstance(result[0], dict):
            return _convert_3x(result)
        return result if result else [[]]
    except TypeError:
        pass
    # Try 3.x predict API
    try:
        result = engine.predict(image_arr)
        return _convert_3x(result)
    except Exception as exc:
        logger.error("Both OCR APIs failed: %s", exc)
        return [[]]


def _convert_3x(result: list) -> list:
    if not result:
        return [[]]
    converted = []
    items = result[0] if (result and isinstance(result[0], list)) else result
    for item in items:
        if not isinstance(item, dict):
            converted.append(item)
            continue
        text = item.get("rec_text", item.get("text", ""))
        score = item.get("rec_score", item.get("score", 0.0))
        bbox = item.get("det_bbox", item.get("bbox", [[0,0],[0,0],[0,0],[0,0]]))
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and isinstance(bbox[0], (int, float)):
            x1, y1, x2, y2 = bbox
            bbox = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        converted.append([bbox, (text, score)])
    return [converted]


# ---------------------------------------------------------------------------
# Parsing + Public API
# ---------------------------------------------------------------------------

def _pil_to_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        return np.stack([image]*3, axis=-1) if image.ndim == 2 else image
    raise TypeError(f"Unsupported type: {type(image)}")


def _parse_output(raw: list, image_shape: tuple) -> OCRResult:
    blocks = []
    if not raw or not raw[0]:
        return OCRResult(image_shape=image_shape)
    for line in raw[0]:
        if not line or len(line) < 2:
            continue
        polygon, text_conf = line[0], line[1]
        if not polygon or not text_conf:
            continue
        text = str(text_conf[0]).strip()
        try:
            conf = float(text_conf[1])
        except (ValueError, TypeError):
            conf = 0.0
        if not text:
            continue
        try:
            pts = [[int(p[0]), int(p[1])] for p in polygon]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
        except Exception:
            continue
        blocks.append(TextBlock(text=text, confidence=conf, bbox=bbox, raw_bbox=pts))
    return OCRResult(blocks=blocks, full_text=" ".join(b.text for b in blocks), image_shape=image_shape)


def _detect_script(arr: np.ndarray) -> str:
    try:
        engine = _get_engine("en", use_angle_cls=False)
        raw = _run_engine(engine, arr, cls=False)
        if not raw or not raw[0]:
            return "en"
        texts = [line[1][0] for line in raw[0] if line and len(line) >= 2 and line[1]]
        combined = " ".join(texts)
        ratio = sum(1 for c in combined if ord(c) > 127) / max(len(combined), 1)
        return "hi" if ratio > 0.3 else "en"
    except Exception:
        return "en"


def run_ocr(
    image: Image.Image | np.ndarray,
    lang: Optional[str] = None,
    auto_detect_lang: bool = True,
    confidence_threshold: float = 0.3,
) -> OCRResult:
    arr = _pil_to_array(image)
    h, w = arr.shape[:2]
    if lang is None:
        lang = _detect_script(arr) if auto_detect_lang else "en"
    engine = _get_engine(lang, use_angle_cls=True)
    try:
        raw = _run_engine(engine, arr, cls=True)
    except Exception as exc:
        logger.error("OCR failed: %s", exc)
        return OCRResult(image_shape=(h, w))
    result = _parse_output(raw, (h, w))
    result.language_hint = lang
    result.blocks = [b for b in result.blocks if b.confidence >= confidence_threshold]
    result.full_text = " ".join(b.text for b in result.blocks)
    return result


def _bbox_iou(a: list, b: list) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    return inter / (max(1,(a[2]-a[0])*(a[3]-a[1])) + max(1,(b[2]-b[0])*(b[3]-b[1])) - inter)


def _deduplicate_blocks(blocks: list[TextBlock], iou_threshold: float = 0.5) -> list[TextBlock]:
    kept = []
    for blk in sorted(blocks, key=lambda b: -b.confidence):
        if not any(_bbox_iou(blk.bbox, k.bbox) >= iou_threshold for k in kept):
            kept.append(blk)
    return sorted(kept, key=lambda b: (b.y_min, b.x_min))


def run_ocr_multi_lang(
    image: Image.Image | np.ndarray,
    langs: tuple[str, ...] = ("en", "hi"),
    confidence_threshold: float = 0.3,
) -> OCRResult:
    all_blocks = []
    for lang in langs:
        try:
            r = run_ocr(image, lang=lang, auto_detect_lang=False,
                        confidence_threshold=confidence_threshold)
            all_blocks.extend(r.blocks)
        except Exception as exc:
            logger.warning("OCR lang=%s failed: %s", lang, exc)
    deduped = _deduplicate_blocks(all_blocks)
    arr = _pil_to_array(image)
    h, w = arr.shape[:2]
    return OCRResult(
        blocks=deduped,
        full_text=" ".join(b.text for b in deduped),
        language_hint="+".join(langs),
        image_shape=(h, w),
    )