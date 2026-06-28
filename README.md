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

## Current Scope

Included:

- Project initialization
- Reusable EDA helper functions
- Exploratory profiling
- Data quality report

Excluded from Phase 1:

- Cleaning
- Deduplication decisions
- Classification
- Sentiment modeling
- Dashboard
- FastAPI
- LLM integration
- Deployment
