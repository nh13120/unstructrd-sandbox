from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Base class for all LLM providers."""

    name: str

    @abstractmethod
    async def query(self, prompt: str, context: str = "") -> str:
        """Send a query to the LLM and return the response."""
        ...
