"""
utils/device_utils.py
---------------------
Cross-platform device and dtype detection.
Automatically selects the best available compute device:
  CUDA (Windows/Linux GPU) → MPS (Mac Apple Silicon) → CPU (fallback)
"""

from __future__ import annotations

import platform
import sys
import logging

logger = logging.getLogger(__name__)


def get_best_device() -> str:
    """
    Detect and return the best available device string.
    Returns: 'cuda', 'mps', or 'cpu'
    """
    try:
        import torch

        # 1. NVIDIA GPU (Windows/Linux)
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("Device: CUDA (%s)", name)
            return "cuda"

        # 2. Apple Silicon MPS (Mac M1/M2/M3)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Device: MPS (Apple Silicon)")
            return "mps"

    except ImportError:
        pass

    logger.info("Device: CPU")
    return "cpu"


def get_best_dtype(device: str) -> str:
    """
    Return the best dtype for the given device.
    - CUDA: float16 (fast, memory efficient)
    - MPS:  float16 (Apple Silicon native)
    - CPU:  float32 (float16 is slow on CPU)
    """
    if device in ("cuda", "mps"):
        return "float16"
    return "float32"


def resolve_device(config_device: str) -> str:
    """Resolve 'auto' or a literal device string."""
    if config_device == "auto":
        return get_best_device()
    return config_device


def resolve_dtype(config_dtype: str, device: str) -> str:
    """Resolve 'auto' or a literal dtype string."""
    if config_dtype == "auto":
        return get_best_dtype(device)
    return config_dtype


def is_apple_silicon() -> bool:
    """True on Mac with Apple Silicon (M1/M2/M3)."""
    return (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
    )


def is_windows() -> bool:
    return platform.system() == "Windows"


def is_mac() -> bool:
    return platform.system() == "Darwin"


def ocr_should_run(config_enabled: str) -> bool:
    """
    Decide whether to run PaddleOCR based on platform.

    PaddleOCR on Mac Apple Silicon (arm64) has a known hanging bug
    with the connectivity check. It returns 0 blocks and stalls for
    30+ minutes. Disable automatically on that platform.

    On Windows and Linux it works fine.

    config_enabled: 'auto', 'true', or 'false'
    """
    if config_enabled == "false" or config_enabled is False:
        return False
    if config_enabled == "true" or config_enabled is True:
        return True
    # auto
    if is_apple_silicon():
        logger.info("OCR: auto-disabled on Apple Silicon (known PaddleOCR hang bug)")
        return False
    logger.info("OCR: auto-enabled on %s", platform.system())
    return True


def print_system_info() -> None:
    """Print system info for debugging."""
    device = get_best_device()
    dtype = get_best_dtype(device)
    logger.info(
        "System: %s %s | Python %s | Device: %s | Dtype: %s | OCR: %s",
        platform.system(), platform.machine(),
        sys.version.split()[0],
        device, dtype,
        "enabled" if ocr_should_run("auto") else "disabled (Apple Silicon)",
    )
