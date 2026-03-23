"""
pdf_converter.py
----------------
Converts PDF files to a list of PIL Images or saves them as PNG files.
Handles single-page and multi-page PDFs. Falls back gracefully if
pdf2image / poppler is unavailable (e.g. already-image inputs).

Dependencies:
    pip install pdf2image Pillow
    System: poppler-utils  (sudo apt install poppler-utils  OR
                            conda install -c conda-forge poppler)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper – lazy import so the module loads even without pdf2image
# ---------------------------------------------------------------------------
def _import_pdf2image():
    try:
        from pdf2image import convert_from_path
        from pdf2image.exceptions import (
            PDFInfoNotInstalledError,
            PDFPageCountError,
            PDFSyntaxError,
        )
        return convert_from_path, (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError)
    except ImportError as exc:
        raise ImportError(
            "pdf2image is not installed. Run: pip install pdf2image\n"
            "Also install poppler: sudo apt install poppler-utils"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_pdf_to_images(
    pdf_path: str | Path,
    dpi: int = 200,
    output_dir: Optional[str | Path] = None,
    fmt: str = "PNG",
    first_page: Optional[int] = None,
    last_page: Optional[int] = None,
) -> list[Image.Image]:
    """
    Convert a PDF file to a list of PIL Images, one per page.

    Args:
        pdf_path:   Path to the PDF file.
        dpi:        Rendering resolution. 200 is a good balance between
                    quality and speed for A4 invoices. Use 300 for small text.
        output_dir: If provided, images are also saved here as
                    <stem>_pg<N>.png. Directory is created if absent.
        fmt:        Image format when saving ('PNG' or 'JPEG').
        first_page: 1-indexed first page to convert (None = start).
        last_page:  1-indexed last page to convert (None = end).

    Returns:
        List of PIL Image objects (RGB).

    Raises:
        FileNotFoundError: If pdf_path does not exist.
        RuntimeError:      If conversion fails (corrupt PDF, bad poppler, …).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    convert_from_path, pdf_errors = _import_pdf2image()

    # ── Build kwargs only with values that are not None ──────────────────
    kwargs: dict = {"dpi": dpi, "fmt": fmt}
    if first_page is not None:
        kwargs["first_page"] = first_page
    if last_page is not None:
        kwargs["last_page"] = last_page

    try:
        images: list[Image.Image] = convert_from_path(str(pdf_path), **kwargs)
    except pdf_errors as exc:
        raise RuntimeError(f"pdf2image failed on '{pdf_path}': {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Unexpected error converting '{pdf_path}': {exc}") from exc

    if not images:
        raise RuntimeError(f"No pages extracted from '{pdf_path}'.")

    # Ensure all pages are RGB (some PDFs render as RGBA or P mode)
    images = [img.convert("RGB") for img in images]

    # ── Optionally save to disk ───────────────────────────────────────────
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = pdf_path.stem
        for idx, img in enumerate(images, start=1):
            save_path = output_dir / f"{stem}_pg{idx}.{fmt.lower()}"
            img.save(str(save_path))
            logger.debug("Saved page %d → %s", idx, save_path)

    logger.info(
        "Converted '%s' → %d page(s) at %d dpi", pdf_path.name, len(images), dpi
    )
    return images


def load_image(image_path: str | Path) -> Image.Image:
    """
    Load a single image file (PNG / JPG / TIFF / BMP) as a PIL RGB Image.
    Use this when the input is already an image rather than a PDF.

    Args:
        image_path: Path to the image file.

    Returns:
        PIL Image in RGB mode.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file is not a supported image format.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    supported = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
    if image_path.suffix.lower() not in supported:
        raise ValueError(
            f"Unsupported image format '{image_path.suffix}'. "
            f"Supported: {supported}"
        )

    try:
        img = Image.open(str(image_path)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc

    logger.debug("Loaded image '%s' (%dx%d)", image_path.name, img.width, img.height)
    return img


def ingest_document(
    file_path: str | Path,
    dpi: int = 200,
    output_dir: Optional[str | Path] = None,
) -> list[Image.Image]:
    """
    Universal entry point — accepts both PDFs and image files.

    For PDFs  → converts all pages to PIL Images.
    For images → wraps the single image in a list for uniform handling.

    Args:
        file_path:  Path to a PDF or image file.
        dpi:        Used only for PDF conversion.
        output_dir: If provided, saved images go here (PDF mode only).

    Returns:
        List of PIL Images (always a list, even for single-image inputs).
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return convert_pdf_to_images(file_path, dpi=dpi, output_dir=output_dir)

    # Treat everything else as a direct image
    return [load_image(file_path)]


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def batch_ingest(
    input_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    dpi: int = 200,
    extensions: tuple[str, ...] = (".pdf", ".png", ".jpg", ".jpeg", ".tiff"),
) -> dict[str, list[Image.Image]]:
    """
    Ingest all documents in a directory.

    Args:
        input_dir:  Folder containing PDFs / images.
        output_dir: If provided, converted pages are saved here.
        dpi:        DPI for PDF rendering.
        extensions: File suffixes to process.

    Returns:
        Dict mapping filename → list of PIL Images.
        Files that fail are logged and skipped (never raise).
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    results: dict[str, list[Image.Image]] = {}
    files = [f for f in sorted(input_dir.iterdir()) if f.suffix.lower() in extensions]

    if not files:
        logger.warning("No matching files found in '%s'.", input_dir)
        return results

    logger.info("Found %d file(s) in '%s'.", len(files), input_dir)

    for file in files:
        try:
            pages = ingest_document(file, dpi=dpi, output_dir=output_dir)
            results[file.name] = pages
            logger.info("  ✓ %s → %d page(s)", file.name, len(pages))
        except Exception as exc:
            logger.error("  ✗ %s — skipped: %s", file.name, exc)

    return results


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python pdf_converter.py <path/to/file_or_dir>")
        sys.exit(1)

    target = Path(sys.argv[1])

    if target.is_dir():
        docs = batch_ingest(target, output_dir=target / "converted")
        print(f"\nIngested {len(docs)} document(s).")
        for name, pages in docs.items():
            print(f"  {name}: {len(pages)} page(s)")
    else:
        pages = ingest_document(target)
        print(f"\nLoaded {len(pages)} page(s) from '{target.name}'.")
        for i, p in enumerate(pages, 1):
            print(f"  Page {i}: {p.size[0]}×{p.size[1]} px")
