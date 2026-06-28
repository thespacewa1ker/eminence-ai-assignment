"""Configuration constants for the preprocessing pipeline."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

ASSIGNMENT_DOC_PATH = PROJECT_ROOT / "docs" / "Assignment.docx"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset.xlsx"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_DATASET_PATH = PROCESSED_DATA_DIR / "clean_dataset.csv"
PREPROCESSING_SUMMARY_PATH = PROCESSED_DATA_DIR / "preprocessing_summary.json"
TAXONOMY_PATH = PROCESSED_DATA_DIR / "taxonomy.json"
CLASSIFIED_DATASET_PATH = PROCESSED_DATA_DIR / "classified_dataset.csv"

REQUIRED_COLUMNS = [
    "Date",
    "URL",
    "Source Name",
    "Title",
    "Opening Text",
    "Hit Sentence",
    "Driver",
    "Sub driver",
    "Sentiment",
    "Reach",
]

TEXT_COLUMNS = ["Title", "Opening Text", "Hit Sentence"]
SENTIMENT_VALUES = {"Positive", "Neutral", "Negative"}
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TEST_MODE = True
TEST_ROWS = 5
BATCH_SIZE = 20
