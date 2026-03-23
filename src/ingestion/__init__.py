"""
ingestion/
----------
Document ingestion layer for the Invoice OCR pipeline.

Public API
~~~~~~~~~~
    from ingestion import ingest_document, batch_ingest
    from ingestion import preprocess, preprocess_batch, PreprocessConfig
    from ingestion import save_debug_outputs
"""

from .pdf_converter import (
    batch_ingest,
    convert_pdf_to_images,
    ingest_document,
    load_image,
)
from .preprocessor import (
    PreprocessConfig,
    PreprocessResult,
    deskew,
    fix_rotation,
    preprocess,
    preprocess_batch,
    save_debug_outputs,
)

__all__ = [
    # pdf_converter
    "ingest_document",
    "batch_ingest",
    "convert_pdf_to_images",
    "load_image",
    # preprocessor
    "preprocess",
    "preprocess_batch",
    "PreprocessConfig",
    "PreprocessResult",
    "fix_rotation",
    "deskew",
    "save_debug_outputs",
]