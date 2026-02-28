"""Pre-audit brand research — uses Gemini with Google Search to learn about the brand."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

import config as cfg

logger = logging.getLogger(__name__)

# Patterns that indicate the research returned placeholder/error data rather
# than real brand intelligence.  Matched case-insensitively as substrings.
_GARBAGE_PATTERNS: list[str] = [
    "unavailable",
    "unknown",
    "n/a",
    "not found",
    "offline",
    "error",
    "no information",
    "no data",
    "not accessible",
    "does not exist",
    "currently offline",
    "site offline",
]


def _looks_valid(text: str) -> bool:
    """Return True if *text* looks like real data rather than an error placeholder."""
    low = text.strip().lower()
    if not low:
        return False
    return not any(pat in low for pat in _GARBAGE_PATTERNS)


@dataclass
class FlagshipProduct:
    """A known product from the brand catalog."""
    name: str
    price: str       # String — research may return "$149" or "around $200"
    category: str
    verified: bool = False          # Confirmed on brand website via targeted crawl
    verified_url: str = ""          # Actual product page URL
    verified_price: str = ""        # Price from the website (may differ from research)
    verified_material: str = ""     # Material from the website


@dataclass
class BrandIntelligence:
    """Structured intelligence about a brand, gathered via LLM + web search."""
    positioning: str = ""
    target_audience: str = ""
    brand_values: list[str] = field(default_factory=list)
    product_categories: list[str] = field(default_factory=list)
    price_range: str = ""
    flagship_products: list[FlagshipProduct] = field(default_factory=list)
    key_technologies: list[str] = field(default_factory=list)
    collections: list[str] = field(default_factory=list)
    competitors: list[str] = field(default_factory=list)
    recent_news: list[str] = field(default_factory=list)
    raw_research: str = ""

    @property
    def valid_categories(self) -> list[str]:
        """Product categories with garbage/placeholder entries filtered out."""
        return [c for c in self.product_categories if _looks_valid(c)]

    @property
    def valid_flagship_products(self) -> list[FlagshipProduct]:
        """Flagship products with garbage/placeholder entries filtered out."""
        return [p for p in self.flagship_products if _looks_valid(p.name)]

    @property
    def is_populated(self) -> bool:
        """Return True if research yielded meaningful data."""
        return bool(self.valid_categories) and bool(self.valid_flagship_products)

    @property
    def quality_warnings(self) -> list[str]:
        """Return human-readable warnings about data quality issues."""
        warnings: list[str] = []
        bad_cats = len(self.product_categories) - len(self.valid_categories)
        if bad_cats:
            warnings.append(
                f"{bad_cats} of {len(self.product_categories)} product categories "
                "appear to be error placeholders"
            )
        bad_prods = len(self.flagship_products) - len(self.valid_flagship_products)
        if bad_prods:
            warnings.append(
                f"{bad_prods} of {len(self.flagship_products)} flagship products "
                "appear to be error placeholders"
            )
        return warnings

    def summary(self) -> str:
        """Human-readable summary for logging and judge context.

        Uses validated properties to exclude garbage/placeholder entries.
        """
        lines = []
        if self.positioning and _looks_valid(self.positioning):
            lines.append(f"Brand positioning: {self.positioning}")
        cats = self.valid_categories
        if cats:
            lines.append(f"Categories: {', '.join(cats[:8])}")
        if self.price_range and _looks_valid(self.price_range):
            lines.append(f"Price range: {self.price_range}")
        prods = self.valid_flagship_products
        if prods:
            prod_strs = [f"{p.name} ({p.price})" for p in prods[:6]]
            lines.append(f"Flagship products: {', '.join(prod_strs)}")
            if len(prods) > 6:
                lines.append(f"  +{len(prods) - 6} more products")
        if self.key_technologies:
            lines.append(f"Technologies: {', '.join(self.key_technologies[:5])}")
        if self.collections:
            lines.append(f"Collections: {', '.join(self.collections[:5])}")
        if self.brand_values:
            lines.append(f"Brand values: {', '.join(self.brand_values[:4])}")
        if self.competitors:
            lines.append(f"Competitors: {', '.join(self.competitors[:5])}")
        if self.target_audience and _looks_valid(self.target_audience):
            lines.append(f"Target audience: {self.target_audience}")
        return "\n".join(lines) if lines else "No brand intelligence available"


BRAND_RESEARCH_SYSTEM = (
    "You are a brand research analyst. Your job is to gather comprehensive, "
    "factual intelligence about e-commerce brands. Use web search to find current "
    "product information, prices, and brand details. Always verify information "
    "against the brand's actual website. Return structured JSON only."
)

BRAND_RESEARCH_PROMPT = """\
Research the brand "{brand_name}" ({brand_url}) thoroughly. Search their website and recent coverage.

Return a JSON object with these fields:
{{
  "positioning": "One sentence describing what the brand is and what market segment they serve",
  "target_audience": "Who buys from this brand",
  "brand_values": ["list of 3-5 brand values or principles"],
  "product_categories": ["list of all product categories they sell, e.g. 'running shoes', 'jackets', 'bags'"],
  "price_range": "lowest to highest typical price, e.g. '$60-$350'",
  "flagship_products": [
    {{"name": "exact product name", "price": "$X", "category": "category"}},
    {{"name": "exact product name", "price": "$X", "category": "category"}}
  ],
  "key_technologies": ["proprietary technologies or materials, e.g. 'Gore-Tex', 'Flyknit'"],
  "collections": ["named product lines or collections"],
  "competitors": ["3-5 direct competitor brand names"],
  "recent_news": ["2-3 recent launches, collaborations, or brand events"]
}}

IMPORTANT:
- Use real, current product names and prices from {brand_url}
- Include at least 5 flagship products with real prices
- Be specific about product categories (not just "clothing" — say "rain jackets", "tote bags", etc.)
- Return ONLY valid JSON, no markdown fencing, no preamble
"""


async def _verify_products_on_website(
    brand_url: str,
    products: list[FlagshipProduct],
    brand_name: str,
    max_products: int = 5,
) -> list[FlagshipProduct]:
    """Verify flagship products by crawling the actual brand website.

    Uses Claude with ``allowed_domains`` to restrict search to the brand
    domain.  Products confirmed on the website get their details updated;
    products not found are left with ``verified=False``.
    """
    import anthropic
    from urllib.parse import urlparse

    if not cfg.ANTHROPIC_API_KEY:
        logger.warning("No Anthropic API key — skipping website verification")
        return products

    brand_domain = urlparse(brand_url).netloc.replace("www.", "")
    client = anthropic.AsyncAnthropic(api_key=cfg.ANTHROPIC_API_KEY)

    verify_prompt = (
        "Find the product page for \"{product_name}\" on {brand_url}. "
        "Return ONLY a JSON object (no markdown fencing):\n"
        '{{"found": true/false, "name": "exact name on website", '
        '"price": "$X.XX", "url": "full product page URL", '
        '"material": "primary material", "in_stock": true/false}}\n'
        "If you cannot find this exact product on the website, return "
        '{{"found": false}}.'
    )

    to_verify = [p for p in products if _looks_valid(p.name)][:max_products]

    async def _verify_one(product: FlagshipProduct) -> None:
        prompt = verify_prompt.format(
            product_name=product.name,
            brand_url=brand_url,
        )
        try:
            response = await client.messages.create(
                model=cfg.CLAUDE_MODEL,
                max_tokens=512,
                temperature=0,
                system=(
                    f"You are verifying product data on {brand_url}. "
                    f"Only report information you find on {brand_url}. "
                    "Return JSON only."
                ),
                messages=[{"role": "user", "content": prompt}],
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "allowed_domains": [brand_domain],
                }],
            )
            raw = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*?\}', raw)
            if json_match:
                data = json.loads(json_match.group(0))
                if data.get("found", False):
                    product.verified = True
                    product.verified_url = data.get("url", "")
                    product.verified_price = data.get("price", "")
                    product.verified_material = data.get("material", "")
                    if product.verified_price:
                        product.price = product.verified_price
                    logger.info(f"Verified: {product.name} @ {product.verified_price}")
                else:
                    logger.info(f"Not found on website: {product.name}")
        except Exception as e:
            logger.warning(f"Verification failed for {product.name}: {e}")

    # Run all verifications concurrently
    await asyncio.gather(*[_verify_one(p) for p in to_verify])

    return products


async def research_brand(
    brand_name: str,
    brand_url: str,
) -> BrandIntelligence:
    """Research a brand using Gemini with Google Search grounding.

    Returns a BrandIntelligence dataclass. On failure, returns a
    mostly-empty BrandIntelligence with raw_research containing the
    error or unparsed response for debugging.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=cfg.GOOGLE_API_KEY)

    prompt = BRAND_RESEARCH_PROMPT.format(
        brand_name=brand_name,
        brand_url=brand_url,
    )

    try:
        response = await client.aio.models.generate_content(
            model=cfg.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=BRAND_RESEARCH_SYSTEM,
                temperature=0,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        raw_text = response.text or ""
    except Exception as e:
        logger.warning(f"Brand research API call failed: {e}")
        return BrandIntelligence(raw_research=f"API error: {e}")

    intel = _parse_brand_intelligence(raw_text)

    # Phase 2: Verify flagship products against the actual brand website
    if intel.valid_flagship_products:
        logger.info(
            f"Verifying {len(intel.valid_flagship_products)} products on {brand_url}..."
        )
        intel.flagship_products = await _verify_products_on_website(
            brand_url, intel.flagship_products, brand_name,
        )
        verified_count = sum(1 for p in intel.flagship_products if p.verified)
        logger.info(f"Verified {verified_count}/{len(intel.flagship_products)} products")

    return intel


def _parse_brand_intelligence(raw_text: str) -> BrandIntelligence:
    """Parse Gemini's JSON response into BrandIntelligence."""
    cleaned = raw_text.strip()

    # Strip markdown code fencing
    json_match = re.search(r'\{[\s\S]*\}', cleaned)
    if json_match:
        cleaned = json_match.group(0)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"Brand research returned invalid JSON: {raw_text[:300]}")
        return BrandIntelligence(raw_research=raw_text)

    # Parse flagship_products list of dicts into FlagshipProduct objects
    flagship_raw = data.get("flagship_products", [])
    flagships = []
    for item in flagship_raw:
        if isinstance(item, dict):
            flagships.append(FlagshipProduct(
                name=item.get("name", ""),
                price=str(item.get("price", "")),
                category=item.get("category", ""),
            ))

    return BrandIntelligence(
        positioning=data.get("positioning", ""),
        target_audience=data.get("target_audience", ""),
        brand_values=data.get("brand_values", []),
        product_categories=data.get("product_categories", []),
        price_range=data.get("price_range", ""),
        flagship_products=flagships,
        key_technologies=data.get("key_technologies", []),
        collections=data.get("collections", []),
        competitors=data.get("competitors", []),
        recent_news=data.get("recent_news", []),
        raw_research=raw_text,
    )
