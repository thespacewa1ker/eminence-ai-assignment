# Reputation Intelligence EDA

End-to-end AI-powered Reputation Intelligence solution for monitoring BFSI digital mentions. The project automates data preprocessing, taxonomy generation, AI-based classification, and interactive dashboard analytics.

---

## Objective

Build a scalable reputation intelligence workflow that collects, cleans, classifies, and analyzes BFSI digital mentions using Google Gemini and visualizes insights through an interactive Streamlit dashboard.

---

## Project Structure

```text
data/
  raw/
  processed/
dashboard/
docs/
notebooks/
src/
tests/
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

# Phase 1 – Exploratory Data Analysis

Run the notebook:

```bash
jupyter notebook notebooks/eda.ipynb
```

Outputs:

* docs/EDA.md
* data/processed/exploration.csv

---

# Phase 2 – Data Preprocessing

Run:

```bash
python -m src.preprocessing
```

Pipeline:

1. Load raw dataset
2. Validate required columns
3. Clean and standardize text
4. Normalize sentiment values
5. Create combined article text
6. Remove duplicates
7. Remove empty records
8. Validate processed dataset
9. Save cleaned dataset

Outputs:

* data/processed/clean_dataset.csv
* data/processed/preprocessing_summary.json

---

# Phase 3 – Taxonomy Generation

Generate or load the reputation taxonomy.

```bash
python -m src.taxonomy
```

Output:

* data/processed/taxonomy.json

The taxonomy is generated dynamically using Gemini and stored as JSON for reuse.

---

# Phase 4 – AI Classification

Set the Gemini API key:

```bash
set GEMINI_API_KEY=your_api_key
```

Run:

```bash
python -m src.classifier
```

The classifier:

* Loads the cleaned dataset.
* Reads taxonomy.json.
* Classifies articles in configurable batches.
* Predicts:

  * Driver
  * Sub-driver
  * Sentiment
  * Reason
* Validates every Gemini response.
* Retries failed requests once.
* Saves the classified dataset.

Output:

* data/processed/classified_dataset.csv

---

# Phase 5 – Streamlit Dashboard

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

Dashboard Features

* Executive KPI Summary
* Driver Distribution
* Sub-driver Distribution
* Sentiment Distribution
* Driver vs Sentiment Analysis
* Top Discussion Themes
* Interactive Filters
* Search Functionality
* Content Explorer
* CSV Export

---

# Testing

Run classification in test mode using the configuration in `src/config.py`.

Test outputs are written to:

```text
tests/test_classified_dataset.csv
```

---

# Technologies Used

* Python
* Pandas
* Google Gemini Flash
* Streamlit
* Plotly
* Git & GitHub

---

# Current Scope

Included

* Project initialization
* Exploratory Data Analysis
* Data preprocessing
* Dynamic taxonomy generation
* Gemini-based AI classification
* Batch processing
* Response validation
* Streamlit analytics dashboard
* Interactive filtering and search
* CSV export

Future Enhancements

* FastAPI endpoint for real-time article classification
* Automated daily data ingestion
* Database integration (SQLite/PostgreSQL)
* Cloud deployment
* CI/CD pipeline
* Authentication and user management

```
```
