# Exploratory Data Analysis Report

## Dataset Overview

- Source workbook: `data/raw/Dataset.xlsx`
- Sheet analyzed: first worksheet
- Number of records: 100
- Number of columns: 10
- Columns: Date, URL, Source Name, Title, Opening Text, Hit Sentence, Driver, Sub driver, Sentiment, Reach

## Missing Values

| column | missing_count | missing_percentage |
| --- | --- | --- |
| Driver | 100 | 100.0 |
| Sub driver | 100 | 100.0 |
| Hit Sentence | 46 | 46.0 |
| Reach | 35 | 35.0 |
| Title | 19 | 19.0 |
| Date | 16 | 16.0 |
| Source Name | 9 | 9.0 |
| Opening Text | 5 | 5.0 |
| URL | 0 | 0.0 |
| Sentiment | 0 | 0.0 |

## Duplicate Analysis

- Fully duplicated rows: 1
- Rows belonging to duplicated URLs: 17
- Unique duplicated URL values: 2
- Rows belonging to duplicated titles: 27
- Unique duplicated Title values: 2

## Sentiment Label Profile

| sentiment | count |
| --- | --- |
| neutral | 52 |
| positive | 33 |
| Negative | 12 |
| Positive | 3 |

## Text Quality Summary

| column | non_empty_count | empty_or_null_count | empty_or_null_percentage | avg_length | min_length | max_length |
| --- | --- | --- | --- | --- | --- | --- |
| Title | 81 | 19 | 19.0 | 108.17 | 16 | 859 |
| Opening Text | 95 | 5 | 5.0 | 190.62 | 36 | 1303 |
| Hit Sentence | 54 | 46 | 46.0 | 406.22 | 79 | 6510 |
| combined_text | 100 | 0 | 0.0 | 488.15 | 53 | 8091 |

## Data Quality Observations

- Driver and Sub driver are fully empty, which is expected before the classification phase but must be populated later.
- Hit Sentence has substantial missingness, so downstream text construction should safely fall back to Title and Opening Text.
- Reach is missing for a meaningful share of records, so dashboard reach aggregations will need explicit null handling.
- Date, Source Name, Title, Opening Text, and Hit Sentence contain missing values that should be reviewed before preprocessing.
- Duplicate URLs and duplicate titles indicate syndicated, repeated, or overlapping media mentions that require deduplication rules in the next phase.
- Sentiment labels are populated, but capitalization is inconsistent across values such as positive, Positive, neutral, and Negative.
- Text fields include snippets with ellipses and partial context, which may affect classification confidence if not handled carefully.

## Recommendations Before Preprocessing

- Define deterministic deduplication rules using URL, normalized title, source, and date before classification.
- Preserve the original Title, Opening Text, and Hit Sentence columns while creating derived text fields for modeling.
- Normalize whitespace and text encoding in a later preprocessing phase, but avoid changing raw evidence fields in Phase 1 outputs.
- Standardize categorical labels such as Sentiment only after documenting the raw distribution.
- Decide how missing Reach should be represented before building dashboards or weighted insights.
- Validate whether missing Date and Source Name records are acceptable or should be excluded as low-quality mentions.

## Phase 1 Output

- Exploration dataset saved to `data/processed/exploration.csv`
- `combined_text` was created from Title, Opening Text, and Hit Sentence without cleaning or classification.
