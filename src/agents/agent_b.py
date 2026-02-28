"""Agent B — browses brand website + has Unstructrd structured data in context."""

from __future__ import annotations
from .base_agent import BaseAgent


class AgentB(BaseAgent):
    """Web + Unstructrd agent. Identical to Agent A except its system prompt
    contains the structured product data feed, simulating what
    unstructrd.brand.com would serve to an AI agent."""

    agent_label = "B"

    def __init__(
        self,
        brand_name: str,
        brand_url: str,
        model_key: str,
        structured_data: str = "",
    ) -> None:
        super().__init__(brand_name, brand_url, model_key)
        self.structured_data = structured_data

    @property
    def use_search(self) -> bool:
        """Skip web search when structured data is available — saves ~100-150 tokens per call."""
        return not bool(self.structured_data)

    def _system_prompt(self) -> str:
        # Extract domain for the unstructrd.brand.com display
        domain = self.brand_url.replace("https://", "").replace("http://", "").rstrip("/")

        base = (
            f"You are a helpful shopping assistant. A customer is asking about "
            f"{self.brand_name} products.\n\n"
            f"Use {self.brand_url} as your primary source of product information. "
            f"When answering questions, browse the website to find accurate, "
            f"up-to-date information. When comparing brands or discussing general "
            f"market context, use your knowledge freely.\n\n"
            f"Always provide specific, actionable product recommendations with exact "
            f"names and prices. If you cite product details, include the source URL "
            f"when possible."
        )

        if self.structured_data:
            return (
                f"{base}\n\n"
                f"You also have access to {self.brand_name}'s structured product data feed "
                f"from unstructrd.{domain}. This is real-time, machine-readable catalog data. "
                f"ALWAYS prefer this structured data over what you find by browsing — it is "
                f"the authoritative source for prices, stock, materials, and product details.\n\n"
                f"--- STRUCTURED PRODUCT DATA (from unstructrd.{domain}) ---\n"
                f"{self.structured_data}\n"
                f"--- END STRUCTURED DATA ---\n\n"
                f"When the structured data answers the question, use it directly and cite "
                f"specific values. Only browse the website for information not in the feed."
            )

        return base
