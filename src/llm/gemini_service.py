"""Google Gemini service for generating answers."""

import os

from dotenv import load_dotenv
from google import genai


class GeminiService:
    """Generate answers using the Google Gemini API."""

    def __init__(self) -> None:
        load_dotenv(override=True)

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from .env")

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-3.5-flash"
    def generate_answer(self, prompt: str) -> str:
        """Generate a response from Gemini."""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text or "No answer was generated."