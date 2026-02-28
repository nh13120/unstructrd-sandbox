"""Typed configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
PERPLEXITY_API_KEY: str = os.getenv("PERPLEXITY_API_KEY", "")

# --- Model IDs ---
CLAUDE_MODEL: str = "claude-sonnet-4-6"
GPT_MODEL: str = "gpt-5.2"
GEMINI_MODEL: str = "gemini-3.1-pro-preview"
PERPLEXITY_MODEL: str = "sonar-reasoning"
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "gemini-3.1-pro-preview")

# --- Runtime ---
MAX_CONCURRENCY: int = int(os.getenv("MAX_CONCURRENCY", "10"))  # per provider
QUERIES_PER_CATEGORY: int = int(os.getenv("QUERIES_PER_CATEGORY", "10"))
BOOTSTRAP_ITERATIONS: int = int(os.getenv("BOOTSTRAP_ITERATIONS", "1000"))
TOKEN_COST_PER_1K: float = 0.015  # legacy flat rate (kept for backwards compat)

# Per-provider pricing (USD per 1M tokens) — Feb 2026
PROVIDER_PRICING: dict[str, dict[str, float]] = {
    "claude":  {"input": 3.00, "output": 15.00},
    "gpt":     {"input": 1.75, "output": 14.00},
    "gemini":  {"input": 2.00, "output": 12.00},
    "perplexity": {"input": 3.00, "output": 15.00},
}
