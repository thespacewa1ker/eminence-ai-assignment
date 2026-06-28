"""Reusable data loading and EDA helper functions.

This module intentionally avoids assignment-specific cleaning, classification,
sentiment analysis, or dashboard logic. It provides small utilities that can be
reused by notebooks, scripts, tests, and later pipeline stages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

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
