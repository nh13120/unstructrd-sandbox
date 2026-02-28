"""Shared agent logic: retry, token capture, latency, async execution."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod

import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

import config as cfg


# Per-provider semaphores — each provider runs independently so a slow
# Claude call never blocks Gemini/GPT/Perplexity slots.
_provider_semaphores: dict[str, asyncio.Semaphore] = {}
_DEFAULT_CONCURRENCY = int(cfg.MAX_CONCURRENCY)

# Claude web_search and Gemini google_search are heavy; keep concurrency low
# to avoid 529s / timeouts from overwhelming the provider.
_PROVIDER_CONCURRENCY: dict[str, int] = {
    "claude": 3,
    "gpt": _DEFAULT_CONCURRENCY,
    "gemini": 3,
    "perplexity": _DEFAULT_CONCURRENCY,
}

# Per-provider timeouts — Claude web_search and Gemini google_search do
# multi-step browsing that can take 2-4 minutes on complex queries.
_PROVIDER_TIMEOUTS: dict[str, int] = {
    "claude": 300,
    "gpt": 180,
    "gemini": 300,
    "perplexity": 180,
}
_DEFAULT_TIMEOUT = 180


def reset_provider_semaphores() -> None:
    """Clear cached semaphores so they're recreated on the current event loop.

    Must be called at the start of every ``asyncio.run()`` invocation
    (e.g. each Gradio audit run) to avoid using stale semaphores bound
    to a previous event loop.
    """
    _provider_semaphores.clear()


def _get_provider_semaphore(model_key: str) -> asyncio.Semaphore:
    if model_key not in _provider_semaphores:
        limit = _PROVIDER_CONCURRENCY.get(model_key, _DEFAULT_CONCURRENCY)
        _provider_semaphores[model_key] = asyncio.Semaphore(limit)
    return _provider_semaphores[model_key]


@dataclass
class AgentResponse:
    """Standardised return type for all agent calls."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    model: str
    agent: str  # "A" or "B"


class BaseAgent(ABC):
    """Abstract base for Agent A and Agent B.

    Concrete subclasses implement _build_messages() to control what each agent
    sees.  The run() method handles retry, latency, concurrency, and token
    capture uniformly.
    """

    agent_label: str  # "A" or "B"

    def __init__(
        self,
        brand_name: str,
        brand_url: str,
        model_key: str,
    ) -> None:
        self.brand_name = brand_name
        self.brand_url = brand_url
        self.model_key = model_key  # "claude", "gpt", "gemini", "perplexity"
        self.blind = False  # When True, agent acts as generic shopping assistant

    _BLIND_SYSTEM_PROMPT = (
        "You are a helpful shopping assistant. A customer is looking for "
        "product recommendations.\n\n"
        "Search the web for the best options across all brands. Always provide "
        "specific, actionable recommendations with exact product names, brand "
        "names, prices, and what makes each one stand out.\n\n"
        "If you cite product details, include the source URL when possible."
    )

    @abstractmethod
    def _system_prompt(self) -> str:
        """Return the system prompt for this agent variant."""
        ...

    @property
    def use_search(self) -> bool:
        """Whether this agent should use web search tools. Override in subclasses."""
        return True

    def _brand_domain(self) -> str:
        """Extract the bare domain from brand_url (e.g. 'mcm.com')."""
        from urllib.parse import urlparse
        return urlparse(self.brand_url).netloc.replace("www.", "")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    async def run(self, question: str) -> AgentResponse:
        """Execute a single query with retry + per-provider concurrency."""
        sem = _get_provider_semaphore(self.model_key)
        async with sem:
            return await self._call_with_retry(question)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=30),
        reraise=True,
    )
    async def _call_with_retry(self, question: str) -> AgentResponse:
        t0 = time.perf_counter()
        timeout = _PROVIDER_TIMEOUTS.get(self.model_key, _DEFAULT_TIMEOUT)
        try:
            response = await asyncio.wait_for(
                self._call_model(question),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.info(f"{self.model_key}: timed out after {timeout}s — will retry")
            raise
        except Exception as e:
            err_str = str(e)
            # Rate limit or server overload — back off before retry
            if "429" in err_str or "rate_limit" in err_str:
                logger.info(f"{self.model_key}: rate limited — waiting 30s")
                await asyncio.sleep(30)
            elif "529" in err_str or "overloaded" in err_str.lower():
                logger.info(f"{self.model_key}: server overloaded — waiting 15s")
                await asyncio.sleep(15)
            elif "500" in err_str or "502" in err_str or "503" in err_str:
                logger.info(f"{self.model_key}: server error — waiting 10s")
                await asyncio.sleep(10)
            raise
        response.latency_seconds = time.perf_counter() - t0
        return response

    async def _call_model(self, question: str) -> AgentResponse:
        """Dispatch to the correct provider implementation."""
        if self.model_key == "claude":
            return await self._call_claude(question)
        elif self.model_key == "gpt":
            return await self._call_gpt(question)
        elif self.model_key == "gemini":
            return await self._call_gemini(question)
        elif self.model_key == "perplexity":
            return await self._call_perplexity(question)
        else:
            raise ValueError(f"Unknown model_key: {self.model_key}")

    # ------------------------------------------------------------------
    # Claude — web_search server-side tool
    # ------------------------------------------------------------------
    async def _call_claude(self, question: str) -> AgentResponse:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=cfg.ANTHROPIC_API_KEY)
        system_prompt = self._BLIND_SYSTEM_PROMPT if self.blind else self._system_prompt()

        kwargs: dict = dict(
            model=cfg.CLAUDE_MODEL,
            max_tokens=4096,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        )
        if self.use_search:
            tool_def: dict = {"type": "web_search_20250305", "name": "web_search"}
            # In blind mode, don't restrict search to brand domain — let the agent
            # search freely so the brand must surface on its own merits.
            if not self.blind:
                domain = self._brand_domain()
                if domain:
                    tool_def["allowed_domains"] = [domain]
            kwargs["tools"] = [tool_def]

        response = await client.messages.create(**kwargs)

        text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )

        return AgentResponse(
            text=text,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            latency_seconds=0.0,  # filled by caller
            model=cfg.CLAUDE_MODEL,
            agent=self.agent_label,
        )

    # ------------------------------------------------------------------
    # GPT-4o — Responses API with web_search_preview
    # ------------------------------------------------------------------
    async def _call_gpt(self, question: str) -> AgentResponse:
        import openai

        client = openai.AsyncOpenAI(api_key=cfg.OPENAI_API_KEY)
        system_prompt = self._BLIND_SYSTEM_PROMPT if self.blind else self._system_prompt()

        kwargs: dict = dict(
            model=cfg.GPT_MODEL,
            temperature=0,
            instructions=system_prompt,
            input=question,
        )
        if self.use_search:
            kwargs["tools"] = [{"type": "web_search_preview"}]

        response = await client.responses.create(**kwargs)

        text = ""
        for item in response.output:
            if hasattr(item, "content"):
                for part in item.content:
                    if hasattr(part, "text"):
                        text += part.text

        usage = response.usage
        return AgentResponse(
            text=text,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            latency_seconds=0.0,
            model=cfg.GPT_MODEL,
            agent=self.agent_label,
        )

    # ------------------------------------------------------------------
    # Gemini — google_search grounding (client cached at class level)
    # ------------------------------------------------------------------
    _gemini_client = None

    @classmethod
    def _get_gemini_client(cls):
        if cls._gemini_client is None:
            from google import genai
            cls._gemini_client = genai.Client(api_key=cfg.GOOGLE_API_KEY)
        return cls._gemini_client

    async def _call_gemini(self, question: str) -> AgentResponse:
        from google.genai import types

        client = self._get_gemini_client()
        system_prompt = self._BLIND_SYSTEM_PROMPT if self.blind else self._system_prompt()

        config_kwargs: dict = dict(
            system_instruction=system_prompt,
            temperature=0,
        )
        if self.use_search:
            config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

        response = await client.aio.models.generate_content(
            model=cfg.GEMINI_MODEL,
            contents=question,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        text = response.text or ""
        meta = response.usage_metadata
        prompt_tok = meta.prompt_token_count if meta else 0
        completion_tok = meta.candidates_token_count if meta else 0

        return AgentResponse(
            text=text,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            total_tokens=prompt_tok + completion_tok,
            latency_seconds=0.0,
            model=cfg.GEMINI_MODEL,
            agent=self.agent_label,
        )

    # ------------------------------------------------------------------
    # Perplexity — sonar-pro (OpenAI-compatible, natively searches)
    # ------------------------------------------------------------------
    async def _call_perplexity(self, question: str) -> AgentResponse:
        import openai

        client = openai.AsyncOpenAI(
            api_key=cfg.PERPLEXITY_API_KEY,
            base_url="https://api.perplexity.ai",
        )
        system_prompt = self._BLIND_SYSTEM_PROMPT if self.blind else self._system_prompt()

        # Perplexity natively searches; include brand URL in system prompt
        response = await client.chat.completions.create(
            model=cfg.PERPLEXITY_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )

        text = response.choices[0].message.content or ""
        usage = response.usage
        return AgentResponse(
            text=text,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_seconds=0.0,
            model=cfg.PERPLEXITY_MODEL,
            agent=self.agent_label,
        )
