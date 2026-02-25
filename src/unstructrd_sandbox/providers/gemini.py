from google import genai

from ..config import config
from .base import LLMProvider


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=config.google_api_key)
        self.model = model

    async def query(self, prompt: str, context: str = "") -> str:
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=full_prompt,
        )
        return response.text or ""
