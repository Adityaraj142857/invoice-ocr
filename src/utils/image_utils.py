"""
utils/image_utils.py
---------------------
Shared image helpers used across the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def pil_to_cv2_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL RGB Image → OpenCV BGR ndarray."""
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv2_bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR ndarray → PIL RGB Image."""
    if len(bgr.shape) == 2:
        return Image.fromarray(bgr).convert("RGB")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def crop_region(
    image: Image.Image | np.ndarray,
    bbox: list[int],
) -> Image.Image:
    """
    Crop image to bounding box [x_min, y_min, x_max, y_max].
    Returns PIL Image.
    """
    if isinstance(image, np.ndarray):
        image = cv2_bgr_to_pil(image)
    x1, y1, x2, y2 = bbox
    return image.crop((max(0, x1), max(0, y1), x2, y2))


def resize_for_model(
    image: Image.Image,
    max_dim: int = 1024,
) -> Image.Image:
    """
    Resize image so its longest side ≤ max_dim, preserving aspect ratio.
    Used before sending to VLM to control token count.
    """
    w, h = image.size
    long_side = max(w, h)
    if long_side <= max_dim:
        return image
    scale = max_dim / long_side
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), Image.LANCZOS)


def draw_bbox(
    image: Image.Image,
    bbox: list[int],
    label: str = "",
    colour: tuple[int, int, int] = (0, 200, 0),
    thickness: int = 2,
) -> Image.Image:
    """
    Draw a bounding box on a PIL Image. Returns a new image with the box drawn.
    Useful for debug visualisations and EDA notebooks.
    """
    bgr = pil_to_cv2_bgr(image)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(bgr, (x1, y1), (x2, y2), colour, thickness)
    if label:
        cv2.putText(
            bgr, label, (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
        )
    return cv2_bgr_to_pil(bgr)


def compute_iou(bbox_a: list[int], bbox_b: list[int]) -> float:
    """
    Compute Intersection over Union (IoU) of two bboxes [x1, y1, x2, y2].
    Returns float 0.0–1.0.
    """
    ix1 = max(bbox_a[0], bbox_b[0])
    iy1 = max(bbox_a[1], bbox_b[1])
    ix2 = min(bbox_a[2], bbox_b[2])
    iy2 = min(bbox_a[3], bbox_b[3])

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h

    if inter == 0:
        return 0.0

    area_a = max(1, (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]))
    area_b = max(1, (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]))
    union = area_a + area_b - inter
    return inter / union


def save_image(image: Image.Image, path: str | Path) -> None:
    """Save a PIL Image to disk, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path))
