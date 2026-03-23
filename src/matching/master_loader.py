"""
matching/master_loader.py
--------------------------
Loads dealer and model master lists from CSV files.
Creates normalised lookup lists for fuzzy matching.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Fallback built-in lists if CSV files are not found
_FALLBACK_DEALER_NAMES: list[str] = [
    "Maan Tractor Co",
    "AMS Tractors",
    "MK Motors",
    "Krishna Traders",
    "Sai Wheels",
    "Nayak Auto Care",
    "Shiv Shakti Agro",
    "S D Trading Corporation",
    "Odisha Agro Industries Corporation",
]

_FALLBACK_MODEL_NAMES: list[str] = [
    "Escorts Kubota FT",
    "Powertrac Euro G28",
    "Swaraj 742 XT",
    "Swaraj 744 FE",
    "Swaraj 855 FE",
    "John Deere 5042D",
    "Sonalika DI-7451i Powerplus",
    "Mahindra 575 DI",
    "Eicher 380",
    "PT 434 DS PLUS HR",
]


def load_master_list(
    csv_path: str | Path,
    column_name: Optional[str] = None,
    column_index: int = 0,
) -> list[str]:
    """
    Load a list of names from a CSV file.

    Args:
        csv_path:     Path to the CSV file.
        column_name:  Header name to read from (if CSV has headers).
        column_index: Column index to read from (used if column_name not found).

    Returns:
        List of non-empty strings from the specified column.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.warning("Master CSV not found: %s — using built-in fallback", csv_path)
        return []

    names: list[str] = []
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            # Determine which column to use
            col_idx = column_index
            if header and column_name:
                header_lower = [h.strip().lower() for h in header]
                if column_name.lower() in header_lower:
                    col_idx = header_lower.index(column_name.lower())
                else:
                    # Header exists but column not found — still read first row as data
                    names.append(header[col_idx].strip())

            for row in reader:
                if len(row) > col_idx:
                    val = row[col_idx].strip()
                    if val:
                        names.append(val)
    except Exception as exc:
        logger.error("Error reading '%s': %s", csv_path, exc)
        return []

    logger.info("Loaded %d entries from '%s'", len(names), csv_path.name)
    return names


class MasterData:
    """
    Holds dealer and model master lists.
    Automatically falls back to built-in lists if CSVs are missing.
    """

    def __init__(
        self,
        dealer_csv: Optional[str | Path] = None,
        model_csv: Optional[str | Path] = None,
        master_dir: Optional[str | Path] = None,
    ):
        """
        Args:
            dealer_csv:  Path to dealer_master.csv
            model_csv:   Path to model_master.csv
            master_dir:  Directory containing both CSVs (alternative to explicit paths)
        """
        if master_dir:
            master_dir = Path(master_dir)
            dealer_csv = dealer_csv or master_dir / "dealer_master.csv"
            model_csv = model_csv or master_dir / "model_master.csv"

        self.dealer_names: list[str] = []
        self.model_names: list[str] = []

        if dealer_csv:
            self.dealer_names = load_master_list(dealer_csv, column_name="dealer_name")
        if not self.dealer_names:
            logger.info("Using built-in dealer fallback list (%d entries)", len(_FALLBACK_DEALER_NAMES))
            self.dealer_names = _FALLBACK_DEALER_NAMES.copy()

        if model_csv:
            self.model_names = load_master_list(model_csv, column_name="model_name")
        if not self.model_names:
            logger.info("Using built-in model fallback list (%d entries)", len(_FALLBACK_MODEL_NAMES))
            self.model_names = _FALLBACK_MODEL_NAMES.copy()

    def __repr__(self) -> str:
        return (
            f"MasterData(dealers={len(self.dealer_names)}, "
            f"models={len(self.model_names)})"
        )
