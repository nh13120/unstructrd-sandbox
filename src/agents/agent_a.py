"""Agent A — browses brand website only (baseline)."""

from __future__ import annotations
from .base_agent import BaseAgent


class AgentA(BaseAgent):
    """Web-only agent. Browses www.{brand_url} and nothing else."""

    agent_label = "A"

    def _system_prompt(self) -> str:
        return (
            f"You are a helpful shopping assistant. A customer is asking about "
            f"{self.brand_name} products.\n\n"
            f"When looking up specific product details (prices, materials, availability, "
            f"sizes, colors), search {self.brand_url} as your primary source.\n\n"
            f"Always provide specific, actionable product recommendations with exact "
            f"names and prices. When comparing brands or discussing general market "
            f"context, use your knowledge freely.\n\n"
            f"If you cite product details, include the source URL when possible."
        )
