import openai

from ..config import config
from .base import LLMProvider


class ChatGPTProvider(LLMProvider):
    name = "chatgpt"

    def __init__(self, model: str = "gpt-4o"):
        self.client = openai.AsyncOpenAI(api_key=config.openai_api_key)
        self.model = model

    async def query(self, prompt: str, context: str = "") -> str:
        messages = [{"role": "user", "content": f"{context}\n\n{prompt}" if context else prompt}]
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content or ""
