"""Gemini-backed reputation driver classification pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .config import (
        BATCH_SIZE,
        CLASSIFIED_DATASET_PATH,
        CLEAN_DATASET_PATH,
        SENTIMENT_VALUES,
        TAXONOMY_PATH,
        TEST_CLASSIFIED_DATASET_PATH,
        TEST_MODE,
        TEST_ROWS,
    )
    from .gemini_client import GeminiClient
    from .preprocessing import configure_logging, validate_columns
    from .taxonomy import ensure_taxonomy, load_taxonomy
except ImportError:
    from config import (  # type: ignore
        BATCH_SIZE,
        CLASSIFIED_DATASET_PATH,
        CLEAN_DATASET_PATH,
        SENTIMENT_VALUES,
        TAXONOMY_PATH,
        TEST_CLASSIFIED_DATASET_PATH,
        TEST_MODE,
        TEST_ROWS,
    )
    from gemini_client import GeminiClient  # type: ignore
    from preprocessing import configure_logging, validate_columns  # type: ignore
    from taxonomy import ensure_taxonomy, load_taxonomy  # type: ignore


logger = logging.getLogger(__name__)


def build_classification_prompt(
    article_text: str,
    taxonomy: dict[str, list[str]],
) -> str:
    """Build a Gemini prompt from the persisted taxonomy and article text."""
    taxonomy_json = json.dumps(taxonomy, indent=2)
    return f"""You are classifying BFSI reputation intelligence mentions.

Use only this taxonomy:
{taxonomy_json}

Return ONLY valid JSON with this exact schema:
{{
  "driver": "...",
  "sub_driver": "...",
  "sentiment": "Positive|Neutral|Negative",
  "reason": "..."
}}

Rules:
- The driver must be one of the taxonomy keys.
- The sub_driver must belong to the selected driver.
- The sentiment must be Positive, Neutral, or Negative.
- Do not invent drivers or sub_drivers.
- Keep the reason concise and evidence-based.

Article text:
{article_text}
"""


def parse_gemini_json(response_text: str) -> dict[str, Any]:
    """Parse Gemini JSON, tolerating accidental markdown fences."""
    text = response_text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Gemini response must be a JSON object")
    return parsed


def parse_gemini_json_array(response_text: str) -> list[dict[str, Any]]:
    """Parse a Gemini JSON array, tolerating accidental markdown fences."""
    text = response_text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()

    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("Gemini response must be a JSON array")

    if not all(isinstance(item, dict) for item in parsed):
        raise ValueError("Every Gemini batch response item must be a JSON object")

    return parsed


def validate_classification_response(
    response: dict[str, Any],
    taxonomy: dict[str, list[str]],
) -> dict[str, str]:
    """Validate and normalize a Gemini classification response."""
    required_keys = {"driver", "sub_driver", "sentiment", "reason"}
    missing_keys = required_keys - set(response)
    if missing_keys:
        raise ValueError(f"Gemini response missing keys: {sorted(missing_keys)}")

    driver = str(response["driver"]).strip()
    sub_driver = str(response["sub_driver"]).strip()
    sentiment = str(response["sentiment"]).strip()
    reason = str(response["reason"]).strip()

    if driver not in taxonomy:
        raise ValueError(f"Invalid driver returned by Gemini: {driver}")

    if sub_driver not in taxonomy[driver]:
        raise ValueError(
            f"Invalid sub_driver '{sub_driver}' for driver '{driver}'"
        )

    if sentiment not in SENTIMENT_VALUES:
        raise ValueError(f"Invalid sentiment returned by Gemini: {sentiment}")

    if not reason:
        raise ValueError("Gemini response reason cannot be blank")

    return {
        "driver": driver,
        "sub_driver": sub_driver,
        "sentiment": sentiment,
        "reason": reason,
    }


def classify_text_with_retry(
    article_text: str,
    taxonomy: dict[str, list[str]],
    client: GeminiClient,
    max_attempts: int = 2,
) -> dict[str, str]:
    """Classify one article text, retrying once when validation fails."""
    prompt = build_classification_prompt(article_text=article_text, taxonomy=taxonomy)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response_text = client.generate_json(prompt)
            response = parse_gemini_json(response_text)
            return validate_classification_response(response, taxonomy)
        except Exception as exc:
            last_error = exc
            logger.warning("Classification attempt %s failed: %s", attempt, exc)

    raise ValueError(f"Gemini classification failed after retry: {last_error}")


def build_batch_classification_prompt(
    articles: list[dict[str, object]],
    taxonomy: dict[str, list[str]],
) -> str:
    """Build a Gemini prompt for a batch of article texts."""
    taxonomy_json = json.dumps(taxonomy, indent=2)
    articles_json = json.dumps(articles, indent=2, ensure_ascii=False)
    return f"""You are classifying BFSI reputation intelligence mentions.

Use only this taxonomy:
{taxonomy_json}

Return ONLY a valid JSON array with one classification object for every input article.
Each object must use this exact schema:
[
  {{
    "id": 0,
    "driver": "...",
    "sub_driver": "...",
    "sentiment": "Positive|Neutral|Negative",
    "reason": "..."
  }}
]

Each object must include the same id as its corresponding input article:
{{
  "id": 0,
  "driver": "...",
  "sub_driver": "...",
  "sentiment": "Positive|Neutral|Negative",
  "reason": "..."
}}

Rules:
- The driver must be one of the taxonomy keys.
- The sub_driver must belong to the selected driver.
- The sentiment must be Positive, Neutral, or Negative.
- Do not invent drivers or sub_drivers.
- Keep each reason concise and evidence-based.
- Return exactly {len(articles)} objects.
- Do not skip, merge, or pad articles.

Articles:
{articles_json}
"""


def classify_batch_with_retry(
    articles: list[dict[str, object]],
    taxonomy: dict[str, list[str]],
    client: GeminiClient,
    max_attempts: int = 2,
) -> dict[int, dict[str, str]]:
    """Classify one article batch, retrying once when the batch fails."""
    prompt = build_batch_classification_prompt(articles=articles, taxonomy=taxonomy)
    last_error: Exception | None = None
    article_ids = {int(article["id"]) for article in articles}

    for attempt in range(1, max_attempts + 1):
        try:
            response_text = client.generate_json(prompt)
            responses = parse_gemini_json_array(response_text)
            if len(responses) != len(articles):
                raise ValueError(
                    "Gemini batch response length mismatch: "
                    f"expected {len(articles)}, got {len(responses)}"
                )

            classifications: dict[int, dict[str, str]] = {}
            for response in responses:
                if "id" not in response:
                    raise ValueError("Gemini batch response item missing id")

                response_id = int(response["id"])
                if response_id not in article_ids:
                    raise ValueError(
                        f"Gemini returned unexpected article id: {response_id}"
                    )

                classifications[response_id] = validate_classification_response(
                    response,
                    taxonomy,
                )

            if set(classifications) != article_ids:
                raise ValueError(
                    "Gemini batch response ids do not match input ids: "
                    f"expected {sorted(article_ids)}, got {sorted(classifications)}"
                )

            return classifications
        except Exception as exc:
            last_error = exc
            logger.warning("Batch classification attempt %s failed: %s", attempt, exc)

    raise ValueError(f"Gemini batch classification failed after retry: {last_error}")


def classify_dataframe(
    df: pd.DataFrame,
    taxonomy: dict[str, list[str]],
    client: GeminiClient,
) -> pd.DataFrame:
    """Populate existing Driver/Sub Driver columns and add Reason."""
    validate_columns(df, ["Driver", "Sub driver", "Sentiment", "combined_text"])
    result = df.copy()
    result["Driver"] = result["Driver"].astype("object")
    result["Sub driver"] = result["Sub driver"].astype("object")
    result["Sentiment"] = result["Sentiment"].astype("object")
    result["Reason"] = ""

    total_records = len(result)
    for batch_start in range(0, total_records, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_records)
        batch = result.iloc[batch_start:batch_end]
        logger.info(
            "Processing batch %s-%s/%s",
            batch_start + 1,
            batch_end,
            total_records,
        )

        articles = [
            {
                "id": int(index),
                "article": str(row["combined_text"]).strip(),
            }
            for index, row in batch.iterrows()
        ]

        try:
            classifications = classify_batch_with_retry(articles, taxonomy, client)
        except Exception as exc:
            logger.error(
                "Skipping batch %s-%s/%s after classification failure: %s",
                batch_start + 1,
                batch_end,
                total_records,
                exc,
            )
            for index in batch.index:
                result.at[index, "Reason"] = f"Classification failed: {exc}"
            continue

        for position, index in enumerate(
            batch.index,
            start=batch_start + 1,
        ):
            classification = classifications[int(index)]
            result.at[index, "Driver"] = classification["driver"]
            result.at[index, "Sub driver"] = classification["sub_driver"]
            result.at[index, "Sentiment"] = classification["sentiment"]
            result.at[index, "Reason"] = classification["reason"]
            logger.info(
                "Classified article %s/%s: driver=%s | sub_driver=%s",
                position,
                total_records,
                classification["driver"],
                classification["sub_driver"],
            )

    return result


def run_classification_pipeline(
    clean_dataset_path: Path = CLEAN_DATASET_PATH,
    taxonomy_path: Path = TAXONOMY_PATH,
    output_path: Path = CLASSIFIED_DATASET_PATH,
) -> pd.DataFrame:
    """Run Gemini classification on the clean dataset."""
    configure_logging()
    ensure_taxonomy(taxonomy_path=taxonomy_path)
    taxonomy = load_taxonomy(taxonomy_path)

    clean_dataset_path = Path(clean_dataset_path)
    if not clean_dataset_path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {clean_dataset_path}. "
            "Run `python -m src.preprocessing` first."
        )

    logger.info("Loading clean dataset from %s", clean_dataset_path)
    df = pd.read_csv(clean_dataset_path)
    if TEST_MODE:
        logger.info(
            "Running in TEST MODE. Classifying first %s rows only.",
            TEST_ROWS,
        )
        df = df.head(TEST_ROWS).copy()
        if Path(output_path) == CLASSIFIED_DATASET_PATH:
            output_path = TEST_CLASSIFIED_DATASET_PATH

    client = GeminiClient()

    classified_df = classify_dataframe(df=df, taxonomy=taxonomy, client=client)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving classified dataset to %s", output_path)
    classified_df.to_csv(output_path, index=False)
    return classified_df


if __name__ == "__main__":
    run_classification_pipeline()
