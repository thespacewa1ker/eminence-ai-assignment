"""Reusable data loading, EDA, and preprocessing helper functions.

This module intentionally avoids classification, dashboard logic, LLM calls, or
business insight generation. It provides deterministic utilities for profiling
and preparing the raw dataset for a later AI classification phase.
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

try:
    from .config import (
        CLEAN_DATASET_PATH,
        PREPROCESSING_SUMMARY_PATH,
        RAW_DATA_PATH,
        REQUIRED_COLUMNS,
        SENTIMENT_VALUES,
        TEXT_COLUMNS,
    )
except ImportError:
    from config import (  # type: ignore
        CLEAN_DATASET_PATH,
        PREPROCESSING_SUMMARY_PATH,
        RAW_DATA_PATH,
        REQUIRED_COLUMNS,
        SENTIMENT_VALUES,
        TEXT_COLUMNS,
    )

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a simple application logger if logging is not configured."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


def load_excel_dataset(file_path: Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Load an Excel worksheet into a pandas DataFrame.

    Args:
        file_path: Path to the Excel workbook.
        sheet_name: Worksheet name or index to load. Defaults to the first sheet.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the workbook does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    logger.info("Loading dataset from %s", file_path)
    return pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")


def load_dataset(
    file_path: Path = RAW_DATA_PATH,
    sheet_name: str | int | None = 0,
) -> pd.DataFrame:
    """Load the raw assignment dataset from disk."""
    return load_excel_dataset(file_path=file_path, sheet_name=sheet_name)


def validate_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str] = REQUIRED_COLUMNS,
) -> None:
    """Validate that all required raw columns exist in the dataset."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column labels by trimming outer whitespace.

    The original business-friendly column names are preserved because notebooks,
    reports, and assignment language refer to those names directly.
    """
    result = df.copy()
    result.columns = [str(column).strip() for column in result.columns]
    return result


def _clean_text_value(value: Any) -> str | pd.NA:
    """Clean one text value while preserving punctuation and case."""
    if pd.isna(value):
        return pd.NA

    text = html.unescape(str(value))
    text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text if text else pd.NA


def clean_text_columns(
    df: pd.DataFrame,
    text_columns: Sequence[str] = TEXT_COLUMNS,
) -> pd.DataFrame:
    """Clean assignment text fields without lowercasing or removing words."""
    validate_columns(df, text_columns)
    result = df.copy()

    for column in text_columns:
        logger.info("Cleaning text column: %s", column)
        result[column] = result[column].map(_clean_text_value)

    return result


def normalize_sentiment(df: pd.DataFrame, column: str = "Sentiment") -> pd.DataFrame:
    """Normalize sentiment labels to Positive, Neutral, or Negative."""
    validate_columns(df, [column])
    result = df.copy()
    normalized = result[column].astype("string").str.strip().str.casefold()
    sentiment_map = {
        "positive": "Positive",
        "neutral": "Neutral",
        "negative": "Negative",
    }
    result[column] = normalized.map(sentiment_map)

    invalid_count = int(result[column].isna().sum())
    if invalid_count:
        invalid_values = sorted(
            df.loc[result[column].isna(), column].dropna().astype(str).unique().tolist()
        )
        raise ValueError(
            "Sentiment contains unsupported or missing values after normalization: "
            f"{invalid_values or ['<missing>']}"
        )

    return result


def create_combined_text(
    df: pd.DataFrame,
    text_columns: Sequence[str] = TEXT_COLUMNS,
    output_column: str = "combined_text",
) -> pd.DataFrame:
    """Create the combined text field used by later classification steps."""
    return add_combined_text_column(
        df=df,
        text_columns=text_columns,
        output_column=output_column,
    )


def remove_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate rows, keeping the first occurrence."""
    before = len(df)
    result = df.drop_duplicates(keep="first").copy()
    removed = before - len(result)
    logger.info("Removed %s exact duplicate rows", removed)
    return result, removed


def remove_duplicate_mentions(
    df: pd.DataFrame,
    key_columns: Sequence[str] = ("Date", "Source Name", "Title"),
) -> tuple[pd.DataFrame, int]:
    """Remove duplicate digital mentions using the business mention key."""
    validate_columns(df, key_columns)
    before = len(df)

    # URL alone can over-count app store, Reddit, syndicated, or tracking-link
    # records, while Title alone can collapse distinct mentions. Date, source,
    # and title together better represent one published digital mention.
    result = df.drop_duplicates(subset=list(key_columns), keep="first").copy()
    removed = before - len(result)
    logger.info("Removed %s duplicate digital mention rows", removed)
    return result, removed


def remove_irrelevant_records(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove records that cannot support downstream text classification.

    Phase 2 does not perform semantic relevance filtering. This deterministic
    safeguard removes only records whose combined text is blank.
    """
    validate_columns(df, ["combined_text"])
    before = len(df)
    result = df[df["combined_text"].fillna("").astype(str).str.strip().ne("")].copy()
    removed = before - len(result)
    logger.info("Removed %s records with blank combined_text", removed)
    return result, removed


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate the clean dataset before saving."""
    validate_columns(df, [*REQUIRED_COLUMNS, "combined_text"])

    mention_key = ["Date", "Source Name", "Title"]
    if df.duplicated(subset=mention_key).any():
        duplicate_count = int(df.duplicated(subset=mention_key).sum())
        raise ValueError(
            "Validation failed: "
            f"{duplicate_count} duplicate business-key mentions remain"
        )

    sentiments = set(df["Sentiment"].dropna().astype(str).unique())
    unsupported_sentiments = sentiments - SENTIMENT_VALUES
    if unsupported_sentiments:
        raise ValueError(
            "Validation failed: unsupported sentiment values remain: "
            f"{sorted(unsupported_sentiments)}"
        )

    if df["Sentiment"].isna().any():
        raise ValueError("Validation failed: missing sentiment values remain")

    if df["combined_text"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Validation failed: blank combined_text values remain")


def save_outputs(
    df: pd.DataFrame,
    summary: dict[str, Any],
    clean_dataset_path: Path = CLEAN_DATASET_PATH,
    summary_path: Path = PREPROCESSING_SUMMARY_PATH,
) -> None:
    """Save the clean dataset and preprocessing summary."""
    clean_dataset_path = Path(clean_dataset_path)
    summary_path = Path(summary_path)
    clean_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving clean dataset to %s", clean_dataset_path)
    df.to_csv(clean_dataset_path, index=False)

    logger.info("Saving preprocessing summary to %s", summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def process_dataset(
    input_path: Path = RAW_DATA_PATH,
    clean_dataset_path: Path = CLEAN_DATASET_PATH,
    summary_path: Path = PREPROCESSING_SUMMARY_PATH,
    sheet_name: str | int | None = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full deterministic preprocessing pipeline."""
    configure_logging()
    start_time = time.perf_counter()

    logger.info("Starting preprocessing pipeline")
    df = load_dataset(file_path=input_path, sheet_name=sheet_name)
    rows_before = int(len(df))

    logger.info("Standardizing column names")
    df = standardize_column_names(df)
    validate_columns(df)

    logger.info("Cleaning text fields")
    df = clean_text_columns(df)

    logger.info("Normalizing sentiment labels")
    df = normalize_sentiment(df)

    logger.info("Creating combined_text")
    df = create_combined_text(df)

    logger.info("Removing duplicates")
    df, duplicate_rows_removed = remove_duplicate_rows(df)
    df, duplicate_mentions_removed = remove_duplicate_mentions(df)

    logger.info("Removing deterministic irrelevant records")
    df, irrelevant_records_removed = remove_irrelevant_records(df)

    logger.info("Validating clean dataset")
    validate_dataset(df)

    execution_time = round(time.perf_counter() - start_time, 4)
    summary = {
        "rows_before": rows_before,
        "rows_after": int(len(df)),
        "duplicates_removed": int(duplicate_rows_removed),
        "duplicate_mentions_removed": int(duplicate_mentions_removed),
        "irrelevant_records_removed": int(irrelevant_records_removed),
        "missing_values": {
            column: int(value) for column, value in df.isna().sum().to_dict().items()
        },
        "execution_time": execution_time,
    }

    save_outputs(
        df=df,
        summary=summary,
        clean_dataset_path=clean_dataset_path,
        summary_path=summary_path,
    )

    logger.info("Preprocessing pipeline completed in %.4f seconds", execution_time)
    return df, summary

def get_dataset_overview(df: pd.DataFrame) -> dict[str, object]:
    """Return high-level dataset metadata."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "data_types": {column: str(dtype) for column, dtype in df.dtypes.items()},
    }


def get_missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Create a missing-value report with counts and percentages."""
    missing_count = df.isna().sum()
    missing_percentage = (missing_count / len(df) * 100).round(2)

    return (
        pd.DataFrame(
            {
                "missing_count": missing_count,
                "missing_percentage": missing_percentage,
            }
        )
        .sort_values(by=["missing_count", "missing_percentage"], ascending=False)
        .reset_index(names="column")
    )


def get_null_percentage(df: pd.DataFrame) -> pd.Series:
    """Return null percentage per column."""
    return (df.isna().mean() * 100).round(2)


def get_duplicate_row_count(df: pd.DataFrame) -> int:
    """Return the number of fully duplicated rows."""
    return int(df.duplicated().sum())


def get_duplicate_value_report(
    df: pd.DataFrame,
    column: str,
    include_nulls: bool = False,
) -> pd.DataFrame:
    """Return duplicated values and their frequencies for a column.

    Args:
        df: Source DataFrame.
        column: Column to analyze.
        include_nulls: Whether null values should be included in the analysis.

    Returns:
        DataFrame with duplicated values and occurrence counts.
    """
    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")

    values = df[column] if include_nulls else df[column].dropna()
    counts = values.value_counts(dropna=not include_nulls)
    duplicate_counts = counts[counts > 1]

    return duplicate_counts.rename_axis(column).reset_index(name="count")


def add_combined_text_column(
    df: pd.DataFrame,
    text_columns: Sequence[str],
    output_column: str = "combined_text",
) -> pd.DataFrame:
    """Return a copy of the DataFrame with text fields concatenated.

    Null values are ignored so the combined text does not contain placeholder
    strings such as "nan" or "None".
    """
    missing_columns = [column for column in text_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing text columns: {missing_columns}")

    result = df.copy()
    result[output_column] = (
        result.loc[:, text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return result


def get_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric, categorical, and datetime columns."""
    try:
        return df.describe(include="all", datetime_is_numeric=True).transpose()
    except TypeError:
        return df.describe(include="all").transpose()


def get_text_quality_summary(
    df: pd.DataFrame,
    text_columns: Iterable[str],
) -> pd.DataFrame:
    """Summarize completeness and length characteristics for text columns."""
    rows: list[dict[str, object]] = []

    for column in text_columns:
        if column not in df.columns:
            raise KeyError(f"Column not found: {column}")

        text_series = df[column].fillna("").astype(str).str.strip()
        lengths = text_series.str.len()
        non_empty = text_series.ne("")

        rows.append(
            {
                "column": column,
                "non_empty_count": int(non_empty.sum()),
                "empty_or_null_count": int((~non_empty).sum()),
                "empty_or_null_percentage": round(float((~non_empty).mean() * 100), 2),
                "avg_length": round(float(lengths[non_empty].mean()), 2)
                if non_empty.any()
                else 0.0,
                "min_length": int(lengths[non_empty].min()) if non_empty.any() else 0,
                "max_length": int(lengths[non_empty].max()) if non_empty.any() else 0,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    process_dataset()
