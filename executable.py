"""
executable.py
-------------
Main entry point for the Invoice OCR pipeline.
Cross-platform: Mac (Apple Silicon + Intel) and Windows.

Usage:
    python executable.py --input data/train_docs/
    python executable.py --input data/train_docs/ --output sample_output/result.json
    python executable.py --input invoice.pdf --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# ── Suppress noisy warnings before any heavy imports ──────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*chardet.*")
warnings.filterwarnings("ignore", message=".*charset_normalizer.*")
warnings.filterwarnings("ignore", message=".*RequestsDependency.*")
warnings.filterwarnings("ignore", message=".*top_k.*")
warnings.filterwarnings("ignore", message=".*generation flags.*")
warnings.filterwarnings("ignore", message=".*fast processor.*")
warnings.filterwarnings("ignore", message=".*image processor.*")

# ── Logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
# Silence noisy third-party loggers
for _lib in ("httpx", "httpcore", "transformers", "huggingface_hub"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

logger = logging.getLogger("executable")

# ── Add src/ to path ──────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))

import yaml

# ── Pipeline imports ──────────────────────────────────────────────────────
from ingestion.pdf_converter import ingest_document
from ingestion.preprocessor import PreprocessConfig, preprocess

from vlm.qwen_extractor import QwenExtractor, QwenExtractorConfig

from detection.stamp_detector import StampDetectorConfig, detect_stamp
from detection.signature_detector import SignatureDetectorConfig, detect_signature

from extraction.consensus import build_consensus
from extraction.confidence import compute_confidence

from matching.master_loader import MasterData
from matching.fuzzy_matcher import match_all_fields

from utils.image_utils import resize_for_model
from utils.device_utils import (
    resolve_device, resolve_dtype, print_system_info
)
from utils.json_utils import (
    build_output_json, doc_id_from_path,
    save_results, validate_output,
)


# ===========================================================================
# Config loader
# ===========================================================================

def load_config(config_path: Path) -> dict:
    try:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("Config not found: %s — using defaults", config_path)
        return {}
    except Exception as exc:
        logger.error("Config load error: %s — using defaults", exc)
        return {}


def _get(cfg: dict, *keys, default=None):
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
    return node


# ===========================================================================
# Build pipeline components
# ===========================================================================

def build_pipeline_components(cfg: dict) -> dict:

    # ── Preprocessing ─────────────────────────────────────────────────────
    pre_cfg = PreprocessConfig(
        max_dim=_get(cfg, "preprocessing", "max_dim", default=2048),
        min_dim=_get(cfg, "preprocessing", "min_dim", default=800),
        deskew_enabled=_get(cfg, "preprocessing", "deskew_enabled", default=True),
        denoise_enabled=_get(cfg, "preprocessing", "denoise_enabled", default=True),
        clahe_enabled=_get(cfg, "preprocessing", "clahe_enabled", default=True),
        binarise_enabled=_get(cfg, "preprocessing", "binarise_enabled", default=True),
    )

    # ── VLM — auto device/dtype ────────────────────────────────────────────
    vlm_cfg = QwenExtractorConfig(
        model_id=_get(cfg, "vlm", "model_id", default="Qwen/Qwen2-VL-2B-Instruct"),
        device=_get(cfg, "vlm", "device", default="auto"),
        dtype=_get(cfg, "vlm", "dtype", default="auto"),
        max_new_tokens=_get(cfg, "vlm", "max_new_tokens", default=256),
        run_verification=_get(cfg, "vlm", "run_verification", default=False),
        fallback_on_failure=_get(cfg, "vlm", "fallback_on_failure", default=True),
    )
    vlm_extractor = QwenExtractor(vlm_cfg)

    # ── Detection ─────────────────────────────────────────────────────────
    stamp_cfg = StampDetectorConfig(
        search_region_fraction=_get(cfg, "detection", "stamp",
                                    "search_region_fraction", default=0.50),
        min_contour_area=_get(cfg, "detection", "stamp",
                              "min_contour_area", default=800),
        min_ink_density=_get(cfg, "detection", "stamp",
                             "min_ink_density", default=0.04),
        hough_param2=_get(cfg, "detection", "stamp",
                          "hough_param2", default=20),
        min_strategies_agree=_get(cfg, "detection", "stamp",
                                  "min_strategies_agree", default=1),
    )
    sig_cfg = SignatureDetectorConfig(
        search_above_px=_get(cfg, "detection", "signature",
                             "search_above_px", default=220),
        search_h_padding=_get(cfg, "detection", "signature",
                              "search_h_padding", default=100),
        min_contour_area=_get(cfg, "detection", "signature",
                              "min_contour_area", default=80),
        min_stroke_count=_get(cfg, "detection", "signature",
                              "min_stroke_count", default=2),
        fallback_region_fraction=_get(cfg, "detection", "signature",
                                      "fallback_region_fraction", default=0.40),
    )

    # ── Master data ───────────────────────────────────────────────────────
    master_dir = _ROOT / _get(cfg, "matching", "master_dir", default="master_data")
    master_data = MasterData(master_dir=master_dir)

    return {
        "pre_cfg":          pre_cfg,
        "vlm_extractor":    vlm_extractor,
        "stamp_cfg":        stamp_cfg,
        "sig_cfg":          sig_cfg,
        "master_data":      master_data,
        "dealer_threshold": _get(cfg, "matching", "dealer_threshold", default=80.0),
        "model_threshold":  _get(cfg, "matching", "model_threshold", default=85.0),
    }


# ===========================================================================
# Single document processor
# ===========================================================================

def process_document(file_path: Path, components: dict) -> dict:
    doc_id = doc_id_from_path(file_path)
    t_start = time.time()
    logger.info("── Processing: %s ──", file_path.name)

    # ── 1. Ingest ─────────────────────────────────────────────────────────
    try:
        pages = ingest_document(file_path, dpi=200)
        logger.info("  Ingested %d page(s)", len(pages))
    except Exception as exc:
        logger.error("  Ingestion failed: %s", exc)
        return _empty_result(doc_id, time.time() - t_start)

    page = pages[0]

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    try:
        pre = preprocess(page, components["pre_cfg"])
        rgb_image = pre.rgb_clean
        gray_image = pre.gray_enhanced
        logger.info("  Preprocessed: rotation=%d° skew=%.1f°",
                    pre.rotation_applied, pre.skew_applied)
    except Exception as exc:
        logger.warning("  Preprocessing failed (%s) — using raw image", exc)
        rgb_image = page
        gray_image = None

    # ── 3. OCR: skipped — VLM-only mode ──────────────────────────────────
    logger.info("  OCR: skipped (VLM-only mode)")
    ocr_result = None

    # ── 4. VLM extraction ─────────────────────────────────────────────────
    vlm_image = resize_for_model(rgb_image, max_dim=560)
    vlm_result = None
    try:
        vlm_result = components["vlm_extractor"].extract(vlm_image)
        logger.info("  VLM: %d/6 fields found (%.1fs)",
                    vlm_result.fields_found, vlm_result.latency_sec)
    except Exception as exc:
        logger.error("  VLM failed: %s", exc)

    # ── 5. Detection (use original page — not CLAHE-altered) ──────────────
    stamp_result = sig_result = None
    try:
        stamp_result = detect_stamp(page, components["stamp_cfg"])
        sig_result   = detect_signature(page, ocr_result, components["sig_cfg"])
        logger.info("  Detection: stamp=%s sig=%s",
                    stamp_result.present, sig_result.present)
    except Exception as exc:
        logger.error("  Detection failed: %s", exc)

    # ── 6. Field parsing from OCR — skipped (VLM-only mode) ──────────────

    # ── 7. Consensus — VLM fields only ───────────────────────────────────
    vlm_fields = vlm_result.to_dict() if vlm_result else {}
    vlm_conf   = {k: 0.75 for k in vlm_fields}

    det_fields = {}
    det_conf   = {}
    if stamp_result:
        det_fields["stamp_present"]     = stamp_result.present
        det_conf["stamp_present"]       = stamp_result.confidence
    if sig_result:
        det_fields["signature_present"] = sig_result.present
        det_conf["signature_present"]   = sig_result.confidence

    consensus = None
    try:
        consensus = build_consensus(
            ocr_fields={}, ocr_confidences={},
            vlm_fields=vlm_fields, vlm_confidences=vlm_conf,
            detection_fields=det_fields, detection_confidences=det_conf,
        )
    except Exception as exc:
        logger.error("  Consensus failed: %s", exc)

    # ── 8. Fuzzy matching ─────────────────────────────────────────────────
    final_dealer = final_model = None
    if consensus:
        try:
            matches = match_all_fields(
                dealer_name=consensus.dealer_name.value,
                model_name=consensus.model_name.value,
                master_data=components["master_data"],
                dealer_threshold=components["dealer_threshold"],
                model_threshold=components["model_threshold"],
            )
            final_dealer = (matches["dealer"].matched
                            if matches["dealer"].matched_above_threshold
                            else consensus.dealer_name.value)
            final_model  = (matches["model"].matched
                            if matches["model"].matched_above_threshold
                            else consensus.model_name.value)
        except Exception as exc:
            logger.warning("  Fuzzy matching failed: %s", exc)
            final_dealer = consensus.dealer_name.value if consensus else None
            final_model  = consensus.model_name.value  if consensus else None

    # ── 9. Confidence ─────────────────────────────────────────────────────
    overall_conf = 0.0
    if consensus:
        try:
            overall_conf = compute_confidence(consensus).overall
        except Exception:
            overall_conf = consensus.overall_confidence

    # ── 10. Build output ──────────────────────────────────────────────────
    processing_time = time.time() - t_start
    output = build_output_json(
        doc_id=doc_id,
        dealer_name=final_dealer,
        model_name=final_model,
        horse_power=consensus.horse_power.value  if consensus else None,
        asset_cost=consensus.asset_cost.value    if consensus else None,
        signature_present=consensus.signature_present.value if consensus else None,
        signature_bbox=sig_result.bbox   if sig_result   and sig_result.present   else None,
        stamp_present=consensus.stamp_present.value     if consensus else None,
        stamp_bbox=stamp_result.bbox     if stamp_result and stamp_result.present  else None,
        overall_confidence=overall_conf,
        processing_time_sec=processing_time,
        cost_estimate_usd=0.0,
    )

    valid, errors = validate_output(output)
    if not valid:
        logger.warning("  Output issues: %s", errors)

    logger.info(
        "  ✓ Done in %.1fs | conf=%.3f | dealer='%s' model='%s' hp=%s cost=%s sig=%s stamp=%s",
        processing_time, overall_conf,
        output["fields"]["dealer_name"],
        output["fields"]["model_name"],
        output["fields"]["horse_power"],
        output["fields"]["asset_cost"],
        output["fields"]["signature"]["present"],
        output["fields"]["stamp"]["present"],
    )
    return output


def _empty_result(doc_id: str, processing_time: float) -> dict:
    return build_output_json(
        doc_id=doc_id, dealer_name=None, model_name=None,
        horse_power=None, asset_cost=None,
        signature_present=False, signature_bbox=None,
        stamp_present=False, stamp_bbox=None,
        overall_confidence=0.0, processing_time_sec=processing_time,
        cost_estimate_usd=0.0,
    )


# ===========================================================================
# Batch runner
# ===========================================================================

def run_batch(input_path: Path, output_path: Path, config_path: Path) -> list[dict]:
    cfg = load_config(config_path)
    print_system_info()

    logger.info("Initialising pipeline...")
    components = build_pipeline_components(cfg)
    logger.info("Pipeline ready.")

    supported = tuple(_get(cfg, "ingestion", "supported_extensions",
                           default=[".pdf", ".png", ".jpg", ".jpeg", ".tiff"]))

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(f for f in input_path.iterdir()
                       if f.suffix.lower() in supported)
    else:
        logger.error("Input path not found: %s", input_path)
        sys.exit(1)

    if not files:
        logger.error("No supported files found in: %s", input_path)
        sys.exit(1)

    logger.info("Found %d document(s).", len(files))
    results, failed = [], 0

    for i, fp in enumerate(files, 1):
        logger.info("\n[%d/%d] %s", i, len(files), fp.name)
        try:
            results.append(process_document(fp, components))
        except Exception as exc:
            logger.error("Unhandled error on '%s': %s", fp.name, exc)
            results.append(_empty_result(doc_id_from_path(fp), 0.0))
            failed += 1

    save_results(results, output_path)

    confs  = [r["confidence"]          for r in results]
    times  = [r["processing_time_sec"] for r in results]
    logger.info(
        "\n✓ Done: %d docs | avg conf=%.3f | avg latency=%.1fs | failed=%d",
        len(results),
        sum(confs)/max(len(confs),1),
        sum(times)/max(len(times),1),
        failed,
    )
    logger.info("Output saved → %s", output_path)
    return results


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Invoice OCR — cross-platform field extraction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="PDF/image file or folder of documents")
    parser.add_argument("--output", "-o", default="sample_output/result.json",
                        help="Output JSON path")
    parser.add_argument("--config", "-c", default="configs/config.yaml",
                        help="YAML config path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_batch(Path(args.input), Path(args.output), Path(args.config))


if __name__ == "__main__":
    main()