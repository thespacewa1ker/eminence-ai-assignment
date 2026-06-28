"""Utilities for extracting and loading the assignment taxonomy."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from docx import Document

try:
    from .config import ASSIGNMENT_DOC_PATH, TAXONOMY_PATH
    from .preprocessing import configure_logging
except ImportError:
    from config import ASSIGNMENT_DOC_PATH, TAXONOMY_PATH  # type: ignore
    from preprocessing import configure_logging  # type: ignore


logger = logging.getLogger(__name__)


def extract_taxonomy_from_docx(docx_path: Path = ASSIGNMENT_DOC_PATH) -> dict[str, list[str]]:
    """Extract the Driver -> Sub Driver hierarchy from the assignment DOCX."""
    docx_path = Path(docx_path)
    if not docx_path.exists():
        raise FileNotFoundError(f"Assignment document not found: {docx_path}")

    logger.info("Extracting taxonomy from %s", docx_path)
    document = Document(docx_path)

    for table in document.tables:
        headers = [cell.text.strip() for cell in table.rows[0].cells]
        if headers[:2] != ["Reputation Driver", "Sub-Parameter"]:
            continue

        taxonomy: dict[str, list[str]] = {}
        current_driver: str | None = None

        for row in table.rows[1:]:
            cells = [cell.text.strip() for cell in row.cells]
            driver = cells[0]
            sub_driver = cells[1]

            if driver:
                current_driver = driver
                taxonomy.setdefault(current_driver, [])

            if current_driver and sub_driver:
                taxonomy[current_driver].append(sub_driver)

        if taxonomy:
            return taxonomy

    raise ValueError("Classification framework table was not found in the assignment doc")


def save_taxonomy(
    taxonomy: dict[str, list[str]],
    output_path: Path = TAXONOMY_PATH,
) -> None:
    """Persist taxonomy as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving taxonomy to %s", output_path)
    output_path.write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")


def load_taxonomy(taxonomy_path: Path = TAXONOMY_PATH) -> dict[str, list[str]]:
    """Load taxonomy from JSON."""
    taxonomy_path = Path(taxonomy_path)
    if not taxonomy_path.exists():
        raise FileNotFoundError(
            f"Taxonomy file not found: {taxonomy_path}. "
            "Run `python -m src.taxonomy` first."
        )

    return json.loads(taxonomy_path.read_text(encoding="utf-8"))


def ensure_taxonomy(
    docx_path: Path = ASSIGNMENT_DOC_PATH,
    taxonomy_path: Path = TAXONOMY_PATH,
) -> dict[str, list[str]]:
    """Load taxonomy if it exists; otherwise extract and save it once."""
    taxonomy_path = Path(taxonomy_path)
    if taxonomy_path.exists():
        logger.info("Loading existing taxonomy from %s", taxonomy_path)
        return load_taxonomy(taxonomy_path)

    taxonomy = extract_taxonomy_from_docx(docx_path)
    save_taxonomy(taxonomy, taxonomy_path)
    return taxonomy


if __name__ == "__main__":
    configure_logging()
    extracted_taxonomy = ensure_taxonomy()
    print(json.dumps(extracted_taxonomy, indent=2))
