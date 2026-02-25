import openai

from ..config import config
from .base import LLMProvider


class PerplexityProvider(LLMProvider):
    name = "perplexity"

    def __init__(self, model: str = "sonar-pro"):
        self.client = openai.AsyncOpenAI(
            api_key=config.perplexity_api_key,
            base_url="https://api.perplexity.ai",
        )
        self.model = model

    async def query(self, prompt: str, context: str = "") -> str:
        messages = [{"role": "user", "content": f"{context}\n\n{prompt}" if context else prompt}]
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content or ""
