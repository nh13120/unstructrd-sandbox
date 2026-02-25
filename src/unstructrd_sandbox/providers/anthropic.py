import anthropic

from ..config import config
from .base import LLMProvider


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        self.client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)
        self.model = model

    async def query(self, prompt: str, context: str = "") -> str:
        messages = [{"role": "user", "content": f"{context}\n\n{prompt}" if context else prompt}]
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=messages,
        )
        return response.content[0].text
