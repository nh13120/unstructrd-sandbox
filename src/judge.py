"""LLM-as-a-Judge: scores both agents vs ground truth."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from google.genai import Client as GenaiClient
    from anthropic import AsyncAnthropic

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config as cfg
from src.query_generator import Query
from src.agents.base_agent import AgentResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared system prompt — scoring rules + JSON schema defined once.
# Category-specific user templates add only what's unique (evaluation criteria,
# category-specific rubrics).  This saves ~250-350 tokens per judge call.
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_BASE = """\
You are a rigorous e-commerce accuracy judge. Respond in valid JSON only — no preamble, no markdown.

JSON SCHEMA:
{"agent_a": {"factual_accuracy": 0.0, "constraint_adherence": 0.0, "oos_handling": null, "ranking_scores": null, "hallucinations": [], "missing_info": [], "constraints_extracted": [], "constraints_satisfied": [], "constraints_dropped": [], "hedge_detected": false, "source_cited": false}, "agent_b": {same}, "most_significant_failure": "...", "failure_type": "..."}

BASE SCORING:
- factual_accuracy: 0.0-1.0. Exact/specific = 1.0, partial = 0.5, wrong/vague = 0.0.
- constraint_adherence: 0.0-1.0. All constraints met = 1.0.
- hallucinations: Exact false claims quoted verbatim. Empty list if none.
  CRITICAL: The agent searched the brand's website. You did NOT. If the agent cites a product
  with a URL from the brand website, treat it as REAL — do not use your training data to
  decide what products exist. Only flag products as fabricated when NO source URL is cited
  AND the claim is highly implausible (e.g., a brand collaboration that makes no sense).
  Collaborations, limited editions, and new arrivals you haven't seen before are NOT hallucinations.
- missing_info: Labels: price, material, sizes, colors, features, availability.
- oos_handling: null unless stock query. 1.0 = correct status + alternatives.
- ranking_scores: null unless ranking query. Per-position relevance [1.0, 0.5, 0.0].
- source_cited: true if the response cites a URL from the brand website, false otherwise.

CONSTRAINT AUDITING (MANDATORY):
Before scoring, you MUST:
1. Extract EVERY constraint from the customer question into "constraints_extracted" (budget limits, feature requirements, negative constraints like "not leather", temporal requirements like "currently in stock today").
2. For each constraint, check if the agent's response EXPLICITLY satisfies it. List satisfied ones in "constraints_satisfied" and dropped/ignored ones in "constraints_dropped".
3. If ANY constraint was dropped (constraints_dropped is non-empty), set constraint_adherence = 0.0. There is no partial credit — an agentic checkout with a missing constraint is a failed transaction.

HEDGE HANDLING (MANDATORY):
Categorize the agent's language into ONE of three tiers:
1. KNOWLEDGE HEDGE — The agent does not know the answer: "I'm not sure", "I think maybe", "I cannot confirm", "I don't have access to", "I wasn't able to find".
   → Set factual_accuracy = 0.0, constraint_adherence = 0.0, hedge_detected = true.
   An agent that cannot provide the information cannot help the customer.
2. RESPONSIBLE DISCLAIMER — The agent provides specific, actionable data but adds a professional caveat: "prices may change", "check the website for current availability", "as of today", "subject to change", "for the most current info visit the website".
   → NO PENALTY. Score the CONTENT of the response normally. Set hedge_detected = false.
   An agent that says "$595, though prices may change" provided useful, transactable data.
3. REFUSAL — The agent refuses to engage: "I cannot help with purchases", "please contact customer service directly", "I'm unable to provide product information".
   → Set factual_accuracy = 0.0, constraint_adherence = 0.0, hedge_detected = true.

Key: Focus on whether the response contains ACTIONABLE product data. A response with real product names, prices, and details plus a disclaimer is a GOOD response."""

# --- Standard template (CSV mode with ground truth) ---

JUDGE_USER_TEMPLATE = """\
QUESTION: {prompt}
GROUND TRUTH: {expected}
GROUND TRUTH KEYS: {ground_truth_keys}

AGENT A RESPONSE:
{agent_a_response}

AGENT B RESPONSE:
{agent_b_response}

failure_type options: Ghost Inventory | Wrong Price | Wrong Sizing | Wrong Care | Non-existent Feature | Wrong Policy"""

# --- Discovery template (no ground truth, evaluates specificity) ---

DISCOVERY_JUDGE_USER_TEMPLATE = """\
CUSTOMER QUESTION: {prompt}
BRAND: {brand_name} ({brand_url})

AGENT A RESPONSE:
{agent_a_response}

AGENT B RESPONSE:
{agent_b_response}

CONSTRAINT EXTRACTION (MANDATORY FIRST STEP):
Identify every constraint in the customer question: budget limits, feature requirements, negative exclusions ("not X", "excluding Y", "without Z"), and temporal requirements ("currently in stock", "available today", "as of today"). List them in constraints_extracted.
For each constraint, verify the agent's response explicitly addressed it. If ANY constraint was dropped or ignored, set constraint_adherence = 0.0 and list the dropped constraint in constraints_dropped.

HEDGE HANDLING: Apply the 3-tier system from the base instructions.
- Knowledge hedges ("I'm not sure", "I think maybe") → score 0.0, hedge_detected = true.
- Responsible disclaimers ("prices may change", "check the site for availability") → NO penalty, score the content normally.
- Refusals ("I cannot help") → score 0.0, hedge_detected = true.

SOURCE ATTRIBUTION:
Set source_cited = true if the response includes any URL from {brand_url} (e.g. a product page link). Set source_cited = false if no brand website URLs are cited.

PRODUCT DATA QUALITY RUBRIC for factual_accuracy:
1.0 = Specific products with exact prices and details, citing the brand website as source.
0.8 = Specific products with exact prices but no source citation from the brand website.
0.6 = Specific products but some details vague, incomplete, or potentially inaccurate.
0.4 = Some products named but lacking price/detail specificity.
0.2 = Mostly vague, few specific products, prices missing.
0.0 = No specific products, only generic descriptions, or knowledge hedge / refusal.

ACCURACY CHECK: Penalize responses that give confident but WRONG details. A vague-but-honest response should score higher than a specific-but-wrong one.

constraint_adherence: Budget respected, occasion matched, negative exclusions honored, temporal requirements met. If ANY constraint from constraints_extracted is missing from the response, score = 0.0.

hallucinations: Only flag CLEARLY false claims with NO source URL. The agent browsed the brand website — if it cites a product page URL, the product is real. Do NOT use your own training data to decide what products exist. NEVER flag availability/stock status claims as hallucinations. Forward-dated seasonal collections and brand collaborations are NOT hallucinations.

ADVERSARIAL TRAP RULES: If the question embeds false info, a GOOD response pushes back. 1.0=clearly pushed back with reasoning, 0.7=pushed back weakly, 0.5=generic avoidance, 0.0=confirmed false premise.

failure_type options: Fabricated Product | Wrong Price | Missing Price | Missing Details | Constraint Violation | Vague Response | Wrong Brand | Hedge Detected | No Source Citation
"Missing Details" = details the customer EXPLICITLY asked for that were not provided."""

# --- Blind discovery template (brand surfacing test) ---

BLIND_DISCOVERY_JUDGE_TEMPLATE = """\
CUSTOMER QUESTION: {prompt}
TARGET BRAND: {brand_name} ({brand_url})

NOTE: The customer asked a GENERIC question without naming any brand.
The purpose of this test is to determine whether the agent surfaces {brand_name}
products from a category/attribute description alone. Score PRIMARILY on whether
{brand_name} was recommended — not on general recommendation quality.

AGENT A RESPONSE:
{agent_a_response}

AGENT B RESPONSE:
{agent_b_response}

BRAND SURFACING ANALYSIS (MANDATORY FIRST STEP):
For each agent, determine and include these fields in your JSON:
1. "brand_surfaced": Did the agent recommend at least one SPECIFIC product from {brand_name}?
   - true = Named a specific {brand_name} product (e.g. "{brand_name} Stark Backpack")
   - false = Did not mention {brand_name} at all, OR mentioned {brand_name} only in passing
     without recommending a specific product ("{brand_name} is a luxury brand" does NOT count)
2. "brand_product_count": How many distinct {brand_name} products were specifically recommended
   with enough detail to act on (name + price or name + description)? Integer, 0 if none.
3. "brand_recommendation_quality": Quality of the {brand_name} product recommendations:
   - 1.0 = Specific {brand_name} products with prices and details
   - 0.7 = Specific {brand_name} products but missing prices or key details
   - 0.3 = {brand_name} mentioned with vague product references ("they have backpacks")
   - 0.0 = {brand_name} not surfaced at all

CONSTRAINT EXTRACTION (MANDATORY):
Identify every constraint in the customer question: budget limits, feature requirements,
negative exclusions ("not X", "excluding Y", "without Z"), temporal requirements
("currently in stock", "available today"). List them in constraints_extracted.
For each constraint, verify the agent addressed it. If ANY constraint was dropped,
set constraint_adherence = 0.0.

HEDGE HANDLING: Apply the 3-tier system from the base instructions.

SOURCE ATTRIBUTION:
Set source_cited = true if the response includes any URL from {brand_url}.

FACTUAL ACCURACY RUBRIC (brand-gated):
The factual_accuracy score is GATED on brand surfacing:
- If brand_surfaced = false: factual_accuracy MUST be 0.0–0.2 regardless of how good
  the other recommendations are. The test objective failed.
  0.2 = other recommendations were real products with accurate details (good agent, wrong brand)
  0.0 = vague or hedging response
- If brand_surfaced = true: score the {brand_name} product recommendations:
  1.0 = Specific {brand_name} products with exact prices, citing brand website
  0.8 = Specific {brand_name} products with prices but no source citation
  0.6 = Specific {brand_name} products but some details vague or incomplete
  0.4 = {brand_name} products named but lacking prices or detail specificity

ranking_scores: Rate up to 3 recommended products by relevance TO THE TARGET BRAND:
- 1.0 = A specific {brand_name} product matching the customer's described need
- 0.5 = A {brand_name} product mentioned but not well-matched to the need
- 0.0 = A non-{brand_name} product, or no product in this position

hallucinations: Only flag CLEARLY false claims with NO source URL. The agent browsed
the brand website — if it cites a product page URL, the product is real.

most_significant_failure: If brand not surfaced, this MUST be "Target brand {brand_name} not recommended".
failure_type options: Brand Not Surfaced | Wrong Brand | Vague Response | Missing Price | Hedge Detected | Fabricated Product
{cross_reference_section}"""

# --- Attribute completeness template ---

ATTRIBUTE_COMPLETENESS_JUDGE_TEMPLATE = """\
CUSTOMER QUESTION: {prompt}
BRAND: {brand_name} ({brand_url})
{ground_truth_section}

AGENT A RESPONSE:
{agent_a_response}

AGENT B RESPONSE:
{agent_b_response}

HEDGE HANDLING: Apply the 3-tier system from the base instructions.
- Knowledge hedges ("I'm not sure", "I think maybe") → score 0.0, hedge_detected = true.
- Responsible disclaimers ("prices may change", "check the site") → NO penalty, score the content.
- Refusals ("I cannot help") → score 0.0, hedge_detected = true.
The goal is transactable product data — disclaimers alongside real data are fine.

SOURCE ATTRIBUTION: Set source_cited = true if the response includes a URL from {brand_url}.

Add "attribute_coverage" to each agent's JSON with 6 attributes (price, material, sizes, colors, availability, care):
{{"present": bool, "specific": bool, "accurate": bool, "value_found": "quoted value (80 chars max)"}}
"specific" = concrete value given. "$149.99"=specific, "around $150"=not. "100% nylon"=specific, "high-quality"=not.
"accurate" = the value is correct for this product. Verification rules:
- When BRAND RESEARCH GROUND TRUTH is provided above, use it as primary reference:
  - Price: mark accurate=true ONLY if within 15% of the verified price, or if the agent explicitly hedges ("around $X", "approximately $X").
  - Category: if ground truth says "handbag", sizes like "S/M/L" or shoe sizes are INACCURATE (bags use dimensions).
  - Use brand price range to sanity-check prices for products without an exact ground truth price.
- When NO ground truth is available, mark accurate=true ONLY if the value is plausible for the product type AND brand tier. Be skeptical — a plausible-sounding but unverifiable claim should get accurate=false.
- Availability: mark accurate=true if the agent states whether the product is available or not (e.g. "in stock", "available on the website", "out of stock"). The goal is whether the agent addressed availability — NOT whether the specific inventory data is verifiable in real time.
- Examples of INACCURATE regardless:
  - Sizes "S/M/L" or shoe sizes for a handbag (bags use dimensions, not clothing sizes)
  - Price "$29.99" for a luxury item that clearly costs more
  - Material "100% leather" when the brand is known for vegan/synthetic materials

factual_accuracy = (accurate_and_specific_count / 6) where accurate_and_specific = specific AND accurate. missing_info = attribute names where present=false.
constraint_adherence: 1.0 if all 6 requested, 0.5 if 3-5, 0.0 if fewer.

hallucinations: ONLY flag claims that are clearly fabricated AND have NO source URL. The agent browsed the brand website — if it cites a product page URL, trust the claim. Do NOT flag:
- Products the agent found via web search with a cited URL — they exist on the brand's site
- Availability or stock status claims — NORMAL product info, NEVER a hallucination
- Approximate price ranges that are in the right ballpark for the brand tier
- Generic color/size lists that are plausible for the product type
- Brand collaborations or limited editions you haven't seen before

most_significant_failure: Describe the SINGLE most impactful gap. NEVER describe an availability/stock status claim as a "hallucination" or "inventory hallucination" — the agent is expected to state availability.

failure_type options: Missing Price | Missing Material | Missing Sizes | Missing Colors | Missing Availability | Missing Care | Vague Response
"Missing Availability" means the agent did NOT address availability at all. If the agent said anything about whether the product is in stock or available, do NOT use "Missing Availability"."""

# --- Temporal freshness template ---

TEMPORAL_FRESHNESS_JUDGE_TEMPLATE = """\
CUSTOMER QUESTION: {prompt}
BRAND: {brand_name} ({brand_url})

AGENT A RESPONSE:
{agent_a_response}

AGENT B RESPONSE:
{agent_b_response}

TEMPORAL ACCURACY RUBRIC for factual_accuracy:
1.0 = Provides specific, plausible product info (names, prices, availability). Date caveats are a bonus but NOT required for a high score.
0.7 = Specific product info but some details appear outdated or unverifiable. OR hedges excessively instead of providing useful info.
0.5 = Vague response — no specific products/prices, or only generic hedging ("prices may vary").
0.3 = Provides specific but clearly wrong or fabricated details (invented sale events, fake restock dates).
0.0 = Fabricates dates, invented promotions, or makes definitive claims about real-time events that cannot be verified.

hallucinations: Flag fabricated dates, sale events, or restock claims with specific invented dates ONLY when no source URL is cited. The agent browsed the brand website — if it cites a URL, trust the claim. EXCEPTIONS — these are NOT hallucinations:
- Forward-dated seasonal collections (e.g. "Spring/Summer 2026") — standard in fashion retail.
- "New Collection", "New Arrival", "New In" labels — these are standard merchandising tags that brands apply to products on their websites. Do NOT flag these unless the claim is clearly impossible (e.g., a product discontinued years ago).
- Stock status claims ("in stock", "out of stock") — agents browse live pages and may see different availability at different times. Do NOT flag stock status as a hallucination.

failure_type options: Stale Data | Fabricated Date | No Freshness Caveat | Invented Promotion | Fake Restock

{cross_reference_section}"""

# --- Brand knowledge template ---

BRAND_KNOWLEDGE_JUDGE_TEMPLATE = """\
CUSTOMER QUESTION: {prompt}
BRAND: {brand_name} ({brand_url})
{brand_context_section}

AGENT A RESPONSE:
{agent_a_response}

AGENT B RESPONSE:
{agent_b_response}

BRAND DEPTH RUBRIC for factual_accuracy:
1.0 = Specific products, correct technologies, accurate positioning, verifiable details — all plausible for this brand.
0.7 = Some specific knowledge but gaps in technologies/collections/positioning. Minor inaccuracies.
0.5 = Surface-level — correct brand category but no deep specifics, OR specific but with some clearly wrong claims.
0.3 = Mostly generic, could apply to any brand in the category, OR contains multiple inaccurate claims.
0.0 = No brand-specific knowledge, completely generic, or mostly incorrect/fabricated details.

IMPORTANT: Confident but WRONG details should score LOWER than vague but honest responses. A model that invents plausible-sounding but incorrect product lines, technologies, or brand values is worse than one that hedges.

hallucinations: Flag wrong technologies or incorrect brand values ONLY when no source URL is cited. The agent browsed the brand website — if it cites a URL, trust the claim. Do NOT flag:
- Approximate price ranges that are in the right ballpark for the brand (e.g. "~$39-$99" is fine if actual range is $45-$95)
- Forward-dated seasonal collections (e.g. "Spring/Summer 2026") — standard in fashion retail
- Corporate/ownership claims that don't affect the shopping experience (e.g. wrong parent company doesn't matter; wrong return policy does)

failure_type options: Wrong Product Details | Missing Technology Knowledge | Wrong Brand Values | Missing Collection Info | Generic Response | Wrong Competitive Positioning

{cross_reference_section}"""


# --- Cross-model comparison template (runs once per query, sees all models) ---

CROSS_MODEL_COMPARISON_SYSTEM = """\
You are an AI commerce analyst. You will see the same customer question answered by multiple AI models.
Respond in valid JSON only — no preamble, no markdown."""

CROSS_MODEL_COMPARISON_TEMPLATE = """\
CUSTOMER QUESTION: {prompt}
BRAND: {brand_name} ({brand_url})

{model_responses_section}

Produce a JSON object with:
1. "model_summaries": For each model, a 2-3 sentence summary of what it recommended (products, prices, key claims). Be specific — name the products and prices each model cited.
2. "key_differences": A list of 2-5 bullet strings describing the most important differences across models. Focus on:
   - Different products recommended
   - Conflicting facts (prices, materials, availability, sizing)
   - Gaps in detail (one model is specific, another is vague)
   - Confidence differences (one hedges, another states as fact)
   Do NOT focus only on pricing. Cover all differences.
3. "factual_claims": Extract specific factual claims from each model's response. For each claim type (price, material, sizes, colors, stock_status), list what each model explicitly stated. Only include claims that were explicitly stated — do not infer.

JSON SCHEMA:
{{"model_summaries": {{"claude": "...", "gpt": "...", ...}}, "key_differences": ["...", "..."], "factual_claims": {{"price": {{"claude": "$149", "gpt": "$159"}}, "material": {{"claude": "nylon", "gpt": "recycled nylon"}}, "sizes": {{"claude": "S,M,L,XL", "gpt": "XS-XXL"}}, "colors": {{"claude": "Black, Navy", "gpt": "Black, Blue, Red"}}, "stock_status": {{"claude": "in stock", "gpt": "limited"}}}}}}"""


def _strip_agent_b_section(prompt: str) -> str:
    """Remove the 'AGENT B RESPONSE:' block from a judge user prompt.

    Removes from 'AGENT B RESPONSE:' up to (but not including) the next
    section, identified by a blank line followed by non-blank content.
    """
    return re.sub(
        r"\nAGENT B RESPONSE:\n.*?(?=\n\n\S|\Z)",
        "", prompt, count=1, flags=re.DOTALL,
    )


def _build_judge_prompt(
    query_result: QueryResult,
    discovery_mode: bool = False,
    brand_name: str = "",
    brand_url: str = "",
    baseline_mode: bool = False,
) -> tuple[str, str]:
    """Build the judge user prompt and system prompt, selecting the right template.

    Returns (system_prompt, user_prompt).
    """
    q = query_result.query
    a_text = query_result.agent_a_response.text if query_result.agent_a_response else "N/A — Agent A was not run."
    if baseline_mode:
        b_text = ""
    else:
        b_text = query_result.agent_b_response.text if query_result.agent_b_response else "N/A — Agent B was not run."
    cross_ref = query_result.cross_reference_context

    # Build cross-reference section for discovery judge prompts
    cross_ref_section = ""
    if cross_ref:
        cross_ref_section = (
            f"\nCROSS-REFERENCE DATA (other models' answers to this same question):\n{cross_ref}\n"
            "CRITICAL RULES FOR CROSS-REFERENCE DATA:\n"
            "- This data is CONTEXT ONLY. Other models' claims are NOT ground truth.\n"
            "- Do NOT treat another model's claim (e.g., 'out of stock', 'new arrival') as a fact "
            "that the evaluated model must also state. Each model browses independently and may see "
            "different page states.\n"
            "- Do NOT penalize a model for omitting information that only appears in another model's "
            "response. Judge each response ONLY against the customer's question and what the model "
            "itself claimed.\n"
            "- Only flag claims as hallucinations if they are implausible for the brand on their own "
            "merits — not because another model said something different.\n"
            "- The majority can be wrong. A single model giving a different answer is not evidence of "
            "a hallucination."
        )

    # In baseline mode, modify system prompt so the judge only evaluates Agent A
    system_base = JUDGE_SYSTEM_BASE
    if baseline_mode:
        system_base += (
            "\n\nBASELINE MODE: Only Agent A was run. Evaluate Agent A only. "
            "Set all agent_b fields to null (factual_accuracy: null, constraint_adherence: null, "
            "hallucinations: null, missing_info: null). Focus your most_significant_failure "
            "and failure_type solely on Agent A's response."
        )

    from src.query_generator import ATTRIBUTE_COMPLETENESS, TEMPORAL_FRESHNESS, BRAND_KNOWLEDGE, BLIND_DISCOVERY

    # Brand knowledge gets its own specialized template
    if q.category == BRAND_KNOWLEDGE:
        brand_context = q.ground_truth_keys.get("brand_context", "")
        brand_context_section = f"BRAND INTELLIGENCE (pre-researched):\n{brand_context}" if brand_context else ""
        user_prompt = BRAND_KNOWLEDGE_JUDGE_TEMPLATE.format(
            prompt=q.prompt,
            brand_name=brand_name,
            brand_url=brand_url,
            brand_context_section=brand_context_section,
            agent_a_response=a_text,
            agent_b_response=b_text,
            cross_reference_section=cross_ref_section,
        )
        if baseline_mode:
            user_prompt = _strip_agent_b_section(user_prompt)
        return (system_base, user_prompt)

    # Temporal freshness gets its own specialized template
    if q.category == TEMPORAL_FRESHNESS:
        user_prompt = TEMPORAL_FRESHNESS_JUDGE_TEMPLATE.format(
            prompt=q.prompt,
            brand_name=brand_name,
            brand_url=brand_url,
            agent_a_response=a_text,
            agent_b_response=b_text,
            cross_reference_section=cross_ref_section,
        )
        if baseline_mode:
            user_prompt = _strip_agent_b_section(user_prompt)
        return (system_base, user_prompt)

    if q.category == ATTRIBUTE_COMPLETENESS:
        # Build ground truth section — CSV mode (full GT) or discovery mode (research data)
        gt_section = ""
        if q.expected and q.expected != "Complete product profile with all 6 attributes: price, material, sizes, colors, availability, care":
            gt_section = f"GROUND TRUTH: {q.expected}"
        elif q.ground_truth_keys.get("research_product_price"):
            # Discovery mode: use brand research data as ground truth
            gt_parts = []
            rp_name = q.ground_truth_keys.get("research_product_name", "")
            rp_price = q.ground_truth_keys.get("research_product_price", "")
            rp_category = q.ground_truth_keys.get("research_product_category", "")
            brand_range = q.ground_truth_keys.get("brand_price_range", "")
            is_verified = q.ground_truth_keys.get("ground_truth_verified", False)
            verified_url = q.ground_truth_keys.get("verified_url", "")
            verified_material = q.ground_truth_keys.get("verified_material", "")
            if rp_name:
                gt_parts.append(f"Product: {rp_name}")
            if rp_price:
                source_label = f"from {brand_url}" if is_verified else "from web search, not directly verified"
                gt_parts.append(f"Price ({source_label}): {rp_price}")
            if rp_category:
                gt_parts.append(f"Category: {rp_category}")
            if verified_material:
                gt_parts.append(f"Material (from {brand_url}): {verified_material}")
            if verified_url:
                gt_parts.append(f"Product page: {verified_url}")
            if brand_range:
                gt_parts.append(f"Brand price range: {brand_range}")
            if is_verified:
                gt_label = f"VERIFIED GROUND TRUTH (from {brand_url})"
            else:
                gt_label = "RESEARCH DATA (unverified — from web search, not the brand website)"
            gt_section = f"{gt_label}:\n" + "\n".join(gt_parts)
        user_prompt = ATTRIBUTE_COMPLETENESS_JUDGE_TEMPLATE.format(
            prompt=q.prompt,
            brand_name=brand_name,
            brand_url=brand_url,
            ground_truth_section=gt_section,
            agent_a_response=a_text,
            agent_b_response=b_text,
        )
        if baseline_mode:
            user_prompt = _strip_agent_b_section(user_prompt)
        return (system_base, user_prompt)

    if q.category == BLIND_DISCOVERY:
        user_prompt = BLIND_DISCOVERY_JUDGE_TEMPLATE.format(
            prompt=q.prompt,
            brand_name=brand_name,
            brand_url=brand_url,
            agent_a_response=a_text,
            agent_b_response=b_text,
            cross_reference_section=cross_ref_section,
        )
        if baseline_mode:
            user_prompt = _strip_agent_b_section(user_prompt)
        return (system_base, user_prompt)

    if discovery_mode:
        user_prompt = DISCOVERY_JUDGE_USER_TEMPLATE.format(
            prompt=q.prompt,
            brand_name=brand_name,
            brand_url=brand_url,
            agent_a_response=a_text,
            agent_b_response=b_text,
        )
        if cross_ref_section:
            user_prompt += cross_ref_section
        if baseline_mode:
            user_prompt = _strip_agent_b_section(user_prompt)
        return (system_base, user_prompt)

    user_prompt = JUDGE_USER_TEMPLATE.format(
        prompt=q.prompt,
        expected=q.expected,
        ground_truth_keys=json.dumps(q.ground_truth_keys),
        agent_a_response=a_text,
        agent_b_response=b_text,
    )
    if baseline_mode:
        user_prompt = _strip_agent_b_section(user_prompt)
    return (system_base, user_prompt)


@dataclass
class JudgeResult:
    """Parsed judge evaluation for a single query."""
    query_id: str
    category: str
    prompt: str
    agent_a_scores: dict[str, Any] = field(default_factory=dict)
    agent_b_scores: dict[str, Any] = field(default_factory=dict)
    most_significant_failure: str = ""
    failure_type: str = ""
    raw_json: dict[str, Any] = field(default_factory=dict)
    model_key: str = ""


@dataclass
class QueryResult:
    """Bundles a query with both agent responses for judging."""
    query: Query
    agent_a_response: AgentResponse | None = None
    agent_b_response: AgentResponse | None = None
    model_key: str = ""
    cross_reference_context: str = ""  # Other models' responses for cross-validation


def make_error_judge_result(
    query: Query,
    error_text: str,
    model_key: str = "",
) -> JudgeResult:
    """Create a JudgeResult for an error response without calling the judge LLM."""
    return JudgeResult(
        query_id=query.query_id,
        category=query.category,
        prompt=query.prompt,
        agent_a_scores={
            "factual_accuracy": 0.0,
            "constraint_adherence": 0.0,
            "oos_handling": None,
            "ranking_scores": None,
            "hallucinations": [],
            "missing_info": [],
        },
        agent_b_scores={},
        most_significant_failure=error_text,
        failure_type="API Error",
        raw_json={},
        model_key=model_key,
    )


def _is_gemini_judge() -> bool:
    return "gemini" in cfg.JUDGE_MODEL.lower()


def _is_anthropic_judge() -> bool:
    return "claude" in cfg.JUDGE_MODEL.lower()


def _parse_judge_json(raw_text: str, query_id: str) -> dict:
    """Extract JSON from judge response, handling markdown fencing."""
    cleaned = raw_text.strip()
    json_match = re.search(r'\{[\s\S]*\}', cleaned)
    if json_match:
        cleaned = json_match.group(0)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"Judge returned invalid JSON for {query_id}: {raw_text[:200]}")
        return {
            "agent_a": {"factual_accuracy": 0.0, "constraint_adherence": 0.0, "oos_handling": None, "hallucinations": []},
            "agent_b": {"factual_accuracy": 0.0, "constraint_adherence": 0.0, "oos_handling": None, "hallucinations": []},
            "most_significant_failure": "Judge returned invalid JSON",
            "failure_type": "Unknown",
        }


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=5, max=120),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _judge_single_gemini(
    client: GenaiClient,
    query_result: QueryResult,
    discovery_mode: bool = False,
    brand_name: str = "",
    brand_url: str = "",
    baseline_mode: bool = False,
) -> JudgeResult:
    """Judge using Gemini."""
    from google.genai import types

    q = query_result.query
    system_prompt, user_prompt = _build_judge_prompt(
        query_result, discovery_mode, brand_name, brand_url, baseline_mode=baseline_mode,
    )

    response = await client.aio.models.generate_content(
        model=cfg.JUDGE_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
        ),
    )

    raw_text = response.text or ""
    parsed = _parse_judge_json(raw_text, q.query_id)

    return JudgeResult(
        query_id=q.query_id,
        category=q.category,
        prompt=q.prompt,
        agent_a_scores=parsed.get("agent_a", {}),
        agent_b_scores=parsed.get("agent_b", {}),
        most_significant_failure=parsed.get("most_significant_failure", ""),
        failure_type=parsed.get("failure_type", ""),
        raw_json=parsed,
        model_key=query_result.model_key,
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=5, max=120),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _judge_single_anthropic(
    client: AsyncAnthropic,
    query_result: QueryResult,
    discovery_mode: bool = False,
    brand_name: str = "",
    brand_url: str = "",
    baseline_mode: bool = False,
) -> JudgeResult:
    """Judge using Claude."""
    q = query_result.query
    system_prompt, user_prompt = _build_judge_prompt(
        query_result, discovery_mode, brand_name, brand_url, baseline_mode=baseline_mode,
    )

    response = await client.messages.create(
        model=cfg.JUDGE_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw_text = "".join(
        block.text for block in response.content if hasattr(block, "text")
    )
    parsed = _parse_judge_json(raw_text, q.query_id)

    return JudgeResult(
        query_id=q.query_id,
        category=q.category,
        prompt=q.prompt,
        agent_a_scores=parsed.get("agent_a", {}),
        agent_b_scores=parsed.get("agent_b", {}),
        most_significant_failure=parsed.get("most_significant_failure", ""),
        failure_type=parsed.get("failure_type", ""),
        raw_json=parsed,
        model_key=query_result.model_key,
    )


async def run_judge_batch(
    query_results: list[QueryResult],
    progress_callback: Callable[[int, int], None] | None = None,
    discovery_mode: bool = False,
    brand_name: str = "",
    brand_url: str = "",
    baseline_mode: bool = False,
) -> list[JudgeResult]:
    """Batch all judge calls after all test queries complete."""
    sem = asyncio.Semaphore(cfg.MAX_CONCURRENCY)
    completed = 0

    if _is_gemini_judge():
        from google import genai
        client = genai.Client(api_key=cfg.GOOGLE_API_KEY)
        judge_fn = _judge_single_gemini
    else:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=cfg.ANTHROPIC_API_KEY)
        judge_fn = _judge_single_anthropic

    async def _judge_with_sem(qr: QueryResult) -> JudgeResult:
        nonlocal completed
        async with sem:
            try:
                result = await asyncio.wait_for(
                    judge_fn(
                        client, qr,
                        discovery_mode=discovery_mode,
                        brand_name=brand_name,
                        brand_url=brand_url,
                        baseline_mode=baseline_mode,
                    ),
                    timeout=120,
                )
            except asyncio.TimeoutError:
                result = JudgeResult(
                    query_id=qr.query.query_id,
                    category=qr.query.category,
                    prompt=qr.query.prompt,
                    most_significant_failure="Judge timed out",
                    failure_type="Timeout",
                    model_key=qr.model_key or "",
                )
            completed += 1
            if progress_callback:
                progress_callback(completed, len(query_results))
            return result

    tasks = [_judge_with_sem(qr) for qr in query_results]
    results = await asyncio.gather(*tasks)
    return list(results)


# ---------------------------------------------------------------------------
# LLM-based product name extraction for catalog coverage
# ---------------------------------------------------------------------------

_PRODUCT_EXTRACTION_SYSTEM = """\
You are a product name extractor. Given AI agent responses about an e-commerce brand, \
extract ONLY real product names from that specific brand. Respond in valid JSON only."""

_PRODUCT_EXTRACTION_TEMPLATE = """\
BRAND: {brand_name}

Below are AI agent responses about {brand_name}. Extract every unique product name \
mentioned that belongs to {brand_name}.

RULES:
- Only include products from {brand_name}. Exclude products from competitor brands (e.g. ALDO, Zara, etc.).
- Return actual product names like "Petra Curved Shoulder Bag", "Aelin Kitten-Heel Mules".
- Do NOT include: sentence fragments, section headings, care instructions, descriptions, \
brand slogans, category names, or analytical commentary.
- Deduplicate: "Bryna Backpack" and "Bryna Backpack – Cream" are the same product.
- If a product appears with a color variant, use the base name without the color.

AGENT RESPONSES:
{responses_text}

Return JSON: {{"products": ["Product Name 1", "Product Name 2", ...]}}"""


async def extract_catalog_products(
    agent_responses: dict[str, 'AgentResponse'],
    brand_name: str,
) -> dict[str, list[str]]:
    """Use LLM to extract clean product names per model from agent responses.

    Returns: {model_key: [product_name, ...]}
    """
    from src.agents.base_agent import AgentResponse

    # Group responses by model key
    by_model: dict[str, list[str]] = {}
    for key, resp in agent_responses.items():
        if resp.text.startswith("ERROR"):
            continue
        # key format: "{query_id}_{model_key}"
        parts = key.rsplit("_", 1)
        if len(parts) != 2:
            continue
        mk = parts[1]
        by_model.setdefault(mk, []).append(resp.text)

    results: dict[str, list[str]] = {}

    if _is_gemini_judge():
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=cfg.GOOGLE_API_KEY)

        for mk, texts in by_model.items():
            combined = "\n\n---\n\n".join(texts[:30])  # cap to avoid token overflow
            prompt = _PRODUCT_EXTRACTION_TEMPLATE.format(
                brand_name=brand_name, responses_text=combined,
            )
            try:
                response = await client.aio.models.generate_content(
                    model=cfg.JUDGE_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=_PRODUCT_EXTRACTION_SYSTEM,
                    ),
                )
                parsed = _parse_judge_json(response.text or "", f"catalog_{mk}")
                results[mk] = parsed.get("products", [])
            except Exception as e:
                logger.warning("Catalog extraction failed for %s: %s", mk, e)
                results[mk] = []
    else:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=cfg.ANTHROPIC_API_KEY)
        sem = asyncio.Semaphore(cfg.MAX_CONCURRENCY)

        async def _extract_one(mk: str, texts: list[str]) -> tuple[str, list[str]]:
            combined = "\n\n---\n\n".join(texts[:30])
            prompt = _PRODUCT_EXTRACTION_TEMPLATE.format(
                brand_name=brand_name, responses_text=combined,
            )
            async with sem:
                try:
                    response = await client.messages.create(
                        model=cfg.JUDGE_MODEL,
                        max_tokens=1024,
                        system=_PRODUCT_EXTRACTION_SYSTEM,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    raw = "".join(
                        block.text for block in response.content if hasattr(block, "text")
                    )
                    parsed = _parse_judge_json(raw, f"catalog_{mk}")
                    return mk, parsed.get("products", [])
                except Exception as e:
                    logger.warning("Catalog extraction failed for %s: %s", mk, e)
                    return mk, []

        tasks = [_extract_one(mk, texts) for mk, texts in by_model.items()]
        for mk, products in await asyncio.gather(*tasks):
            results[mk] = products

    return results


# ---------------------------------------------------------------------------
# Cross-model comparison judge (one call per query, sees all model responses)
# ---------------------------------------------------------------------------

MODEL_DISPLAY = {
    "claude": "Claude", "gpt": "GPT", "gemini": "Gemini",
    "perplexity": "Perplexity",
}


def _build_cross_model_prompt(
    prompt: str,
    model_responses: dict[str, str],
    brand_name: str,
    brand_url: str,
) -> tuple[str, str]:
    """Build system + user prompt for cross-model comparison."""
    resp_parts = []
    for mk, text in model_responses.items():
        label = MODEL_DISPLAY.get(mk, mk)
        # Trim to ~600 chars to keep judge prompt manageable
        snippet = text[:600] + ("…" if len(text) > 600 else "")
        resp_parts.append(f"{label.upper()} RESPONSE:\n{snippet}")

    user_prompt = CROSS_MODEL_COMPARISON_TEMPLATE.format(
        prompt=prompt,
        brand_name=brand_name,
        brand_url=brand_url,
        model_responses_section="\n\n".join(resp_parts),
    )
    return (CROSS_MODEL_COMPARISON_SYSTEM, user_prompt)


# ---------------------------------------------------------------------------
# Cross-model discrepancy calculator
# ---------------------------------------------------------------------------

_CLAIM_TYPES = ["price", "material", "sizes", "colors", "stock_status"]


def _extract_single_price(text: str) -> float | None:
    """Extract a single dollar price from text."""
    m = re.search(r'\$[\d,]+(?:\.\d{1,2})?', text)
    if m:
        try:
            return float(m.group().replace('$', '').replace(',', ''))
        except ValueError:
            pass
    return None


def _text_overlap(a: str, b: str) -> float:
    """Compute word-level Jaccard similarity."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0


def _compute_discrepancy(
    factual_claims: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, Any]], str]:
    """Compute a conflict matrix from extracted factual claims.

    Returns (conflict_matrix, fragmentation_level).
    """
    conflict_matrix: dict[str, dict[str, Any]] = {}
    fragmented_count = 0

    for claim_type in _CLAIM_TYPES:
        claims = factual_claims.get(claim_type, {})
        if len(claims) < 2:
            continue

        # Compute pairwise agreement
        models = list(claims.keys())
        agreeing = 0
        total_pairs = 0
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                total_pairs += 1
                v_i = str(claims[models[i]]).strip().lower()
                v_j = str(claims[models[j]]).strip().lower()
                if claim_type == "price":
                    # Price agreement: within 5% tolerance
                    p_i = _extract_single_price(v_i)
                    p_j = _extract_single_price(v_j)
                    if p_i and p_j and abs(p_i - p_j) / max(p_i, p_j) <= 0.05:
                        agreeing += 1
                else:
                    # Fuzzy text match: exact or high overlap
                    if v_i == v_j or _text_overlap(v_i, v_j) > 0.7:
                        agreeing += 1

        agreement_rate = agreeing / total_pairs if total_pairs > 0 else 1.0
        is_fragmented = agreement_rate < 0.5
        if is_fragmented:
            fragmented_count += 1

        conflict_matrix[claim_type] = {
            "claims": dict(claims),
            "agreement_rate": round(agreement_rate, 2),
            "is_fragmented": is_fragmented,
        }

    # Determine fragmentation level
    if fragmented_count == 0:
        level = "none"
    elif fragmented_count == 1:
        level = "minor"
    elif fragmented_count <= 3:
        level = "major"
    else:
        level = "critical"

    return conflict_matrix, level


@dataclass
class CrossModelComparison:
    """Judge-produced comparison of responses across models for one query."""
    query_id: str
    prompt: str
    model_summaries: dict[str, str] = field(default_factory=dict)
    key_differences: list[str] = field(default_factory=list)
    # Structured discrepancy data
    conflict_matrix: dict[str, dict[str, Any]] = field(default_factory=dict)
    fragmentation_level: str = ""  # "none", "minor", "major", "critical"


async def _compare_single_gemini(
    client: "GenaiClient",
    query_id: str,
    prompt: str,
    model_responses: dict[str, str],
    brand_name: str,
    brand_url: str,
) -> CrossModelComparison:
    from google.genai import types

    system_prompt, user_prompt = _build_cross_model_prompt(
        prompt, model_responses, brand_name, brand_url,
    )
    response = await client.aio.models.generate_content(
        model=cfg.JUDGE_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    parsed = _parse_judge_json(response.text or "", query_id)
    factual_claims = parsed.get("factual_claims", {})
    conflict_matrix, frag_level = _compute_discrepancy(factual_claims)
    return CrossModelComparison(
        query_id=query_id,
        prompt=prompt,
        model_summaries=parsed.get("model_summaries", {}),
        key_differences=parsed.get("key_differences", []),
        conflict_matrix=conflict_matrix,
        fragmentation_level=frag_level,
    )


async def _compare_single_anthropic(
    client: "AsyncAnthropic",
    query_id: str,
    prompt: str,
    model_responses: dict[str, str],
    brand_name: str,
    brand_url: str,
) -> CrossModelComparison:
    system_prompt, user_prompt = _build_cross_model_prompt(
        prompt, model_responses, brand_name, brand_url,
    )
    response = await client.messages.create(
        model=cfg.JUDGE_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw_text = "".join(
        block.text for block in response.content if hasattr(block, "text")
    )
    parsed = _parse_judge_json(raw_text, query_id)
    factual_claims = parsed.get("factual_claims", {})
    conflict_matrix, frag_level = _compute_discrepancy(factual_claims)
    return CrossModelComparison(
        query_id=query_id,
        prompt=prompt,
        model_summaries=parsed.get("model_summaries", {}),
        key_differences=parsed.get("key_differences", []),
        conflict_matrix=conflict_matrix,
        fragmentation_level=frag_level,
    )


async def run_cross_model_comparisons(
    consistency_responses: dict[str, dict[str, AgentResponse]],
    brand_name: str = "",
    brand_url: str = "",
    queries: list[Query] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    max_comparisons: int = 5,
) -> list[CrossModelComparison]:
    """Run cross-model comparison judge calls.

    Takes the per-query, per-model response dict and asks the judge to
    summarize and compare all model responses for each query.
    Only processes queries where at least 2 models responded.
    Returns up to *max_comparisons* results, prioritizing queries with the
    most models responding.
    """
    query_map = {q.query_id: q for q in queries} if queries else {}
    sem = asyncio.Semaphore(cfg.MAX_CONCURRENCY)
    completed = 0

    # Select queries with most models, up to max_comparisons
    candidates = []
    for qid, model_resps in consistency_responses.items():
        valid = {m: r.text for m, r in model_resps.items() if not r.text.startswith("ERROR")}
        if len(valid) >= 2:
            candidates.append((qid, valid))
    candidates.sort(key=lambda x: len(x[1]), reverse=True)
    candidates = candidates[:max_comparisons]

    if not candidates:
        return []

    if _is_gemini_judge():
        from google import genai
        client = genai.Client(api_key=cfg.GOOGLE_API_KEY)
        compare_fn = _compare_single_gemini
    else:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=cfg.ANTHROPIC_API_KEY)
        compare_fn = _compare_single_anthropic

    async def _compare_with_sem(qid: str, model_resps: dict[str, str]) -> CrossModelComparison:
        nonlocal completed
        async with sem:
            q = query_map.get(qid)
            prompt = q.prompt if q else ""
            try:
                result = await asyncio.wait_for(
                    compare_fn(client, qid, prompt, model_resps, brand_name, brand_url),
                    timeout=120,
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Cross-model comparison failed for {qid}: {e}")
                result = CrossModelComparison(
                    query_id=qid,
                    prompt=prompt,
                    key_differences=["Comparison could not be generated"],
                )
            completed += 1
            if progress_callback:
                progress_callback(completed, len(candidates))
            return result

    tasks = [_compare_with_sem(qid, resps) for qid, resps in candidates]
    results = await asyncio.gather(*tasks)
    return list(results)
