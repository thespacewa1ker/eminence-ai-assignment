"""Configuration constants for the preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Dataset.xlsx"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_DATASET_PATH = PROCESSED_DATA_DIR / "clean_dataset.csv"
PREPROCESSING_SUMMARY_PATH = PROCESSED_DATA_DIR / "preprocessing_summary.json"

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
