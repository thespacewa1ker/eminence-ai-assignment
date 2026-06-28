"""Small Gemini API wrapper used by the classifier."""

from __future__ import annotations

import logging
import os

try:
    from .config import GEMINI_MODEL, PROJECT_ROOT
except ImportError:
    from config import GEMINI_MODEL, PROJECT_ROOT  # type: ignore


logger = logging.getLogger(__name__)


class GeminiClient:
    """Thin wrapper around the Google GenAI SDK."""

    def __init__(self, model: str = GEMINI_MODEL, api_key: str | None = None) -> None:
        self._load_environment()
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY."
            )

        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Google GenAI SDK is not installed. Run `pip install google-genai`."
            ) from exc

        self._client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _load_environment() -> None:
        """Load project-root .env values before reading Gemini credentials."""
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise ImportError(
                "python-dotenv is not installed. Run `pip install python-dotenv`."
            ) from exc

        load_dotenv(PROJECT_ROOT / ".env")

    def generate_json(self, prompt: str) -> str:
        """Call Gemini and return the raw text response."""
        logger.debug("Calling Gemini model %s", self.model)

        interaction = self._client.interactions.create(
            model=self.model,
            input=prompt,
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        )
        return interaction.output_text
