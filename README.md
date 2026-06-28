# Reputation Intelligence EDA

Phase 1 project setup and exploratory data analysis for an AI & Data Solutions
Engineer interview assignment.

## Objective

Build a scalable reputation intelligence workflow for BFSI digital mentions.
This phase initializes the repository, profiles the provided dataset, and
documents data quality observations before any preprocessing, classification,
sentiment analysis, dashboarding, or LLM integration.

## Project Structure

```text
data/
  raw/
  processed/
docs/
notebooks/
src/
tests/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Phase 1 Usage

Open and run `notebooks/eda.ipynb` to reproduce the exploratory analysis.
The notebook reads `data/raw/Dataset.xlsx` and writes:

- `docs/EDA.md`
- `data/processed/exploration.csv`

## Phase 2 Preprocessing Pipeline

The preprocessing pipeline prepares the raw dataset for later AI
classification without using LLMs, classification logic, dashboards, or APIs.
For deduplication, a digital mention is identified by the business key:
`Date + Source Name + Title`. This avoids treating repeated platform URLs as
the only identity signal and avoids collapsing distinct mentions that happen to
share a title.

Run it from the project root:

```bash
python -m src.preprocessing
```

Pipeline flow:

1. Load `data/raw/Dataset.xlsx`
2. Validate required columns
3. Standardize column name whitespace
4. Clean text fields: `Title`, `Opening Text`, `Hit Sentence`
5. Normalize `Sentiment` to `Positive`, `Neutral`, or `Negative`
6. Create `combined_text` from `Title`, `Opening Text`, and `Hit Sentence`
7. Remove exact duplicate rows
8. Remove duplicate digital mentions using `Date + Source Name + Title`,
   keeping the first occurrence
9. Remove records with blank `combined_text`
10. Validate the clean dataset
11. Save outputs

Generated files:

- `data/processed/clean_dataset.csv`
- `data/processed/preprocessing_summary.json`

## Current Scope

Included:

- Project initialization
- Reusable EDA helper functions
- Exploratory profiling
- Data quality report
- Deterministic preprocessing pipeline

Excluded from the current phases:

- Classification
- Sentiment modeling
- Dashboard
- FastAPI
- LLM integration
- Deployment
