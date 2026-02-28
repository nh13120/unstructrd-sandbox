"""Scoring module: nDCG, bootstrap CI, precision/recall, CoV, platform consistency."""

from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass

from src.judge import JudgeResult, CrossModelComparison
from src.query_generator import (
    TABLE_STAKES,
    INVENTORY_OOS,
    MULTI_PRODUCT_COMPARISON,
    OCCASION_REASONING,
    BLIND_DISCOVERY,
    PRODUCT_COMPARISON,
    CROSS_PLATFORM_CONSISTENCY,
    TOKEN_EFFICIENCY,
    ADVERSARIAL,
    ATTRIBUTE_COMPLETENESS,
    TEMPORAL_FRESHNESS,
    BRAND_KNOWLEDGE,
    REQUIRED_ATTRIBUTES,
    LAYER_MAPPING,
    LAYER_1_DISCOVERABILITY,
    LAYER_2_BRAND_DNA,
)
from src.agents.base_agent import AgentResponse

import config as cfg


def _to_float(val) -> float:
    """Safely coerce a judge score to float."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _query_is_actionable(
    jr: JudgeResult,
    agent_response_text: str,
    cross_model_consistency: float | None = None,
) -> tuple[bool, list[str]]:
    """Return (is_actionable, failure_reasons) for a single query result.

    Fails if:
    1. Judge flagged hedge_detected (3-tier system: knowledge hedges & refusals)
    2. Judge found a dropped constraint (constraint_adherence < 0.5)
    3. Cross-model consistency for this query is below threshold (< 0.5)
    """
    reasons: list[str] = []
    # Trust the judge's 3-tier hedge determination instead of independent regex.
    # The judge distinguishes knowledge hedges (penalized) from responsible
    # disclaimers like "prices may change" (not penalized).
    if jr.agent_a_scores.get("hedge_detected", False):
        reasons.append("hedge_detected")
    ca = jr.agent_a_scores.get("constraint_adherence")
    if ca is not None and _to_float(ca) < 0.5:
        reasons.append("dropped_constraint")
    if cross_model_consistency is not None and cross_model_consistency < 0.5:
        reasons.append("cross_model_conflict")
    return (len(reasons) == 0, reasons)


# ------------------------------------------------------------------
# Core metric functions
# ------------------------------------------------------------------

def ndcg_at_k(relevance_scores: list[float], k: int = 3) -> float:
    """Normalized Discounted Cumulative Gain at position k."""
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores[:k]))
    idcg = sum(
        rel / np.log2(i + 2)
        for i, rel in enumerate(sorted(relevance_scores, reverse=True)[:k])
    )
    return float(dcg / idcg) if idcg > 0 else 0.0


def bootstrap_ci(
    scores: list[float],
    n_iter: int | None = None,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    if n_iter is None:
        n_iter = cfg.BOOTSTRAP_ITERATIONS
    if not scores:
        return (0.0, 0.0)
    arr = np.array(scores)
    means = [
        float(np.mean(np.random.choice(arr, size=len(arr), replace=True)))
        for _ in range(n_iter)
    ]
    alpha = (1 - confidence) / 2
    return (
        float(np.percentile(means, alpha * 100)),
        float(np.percentile(means, (1 - alpha) * 100)),
    )


def coeff_of_variation(scores: list[float]) -> float:
    """Coefficient of Variation — measures response consistency."""
    if not scores:
        return 0.0
    mean = float(np.mean(scores))
    return float(np.std(scores) / mean) if mean > 0 else 0.0


def _benchmark_efficiency(total_tokens: int, benchmark_median: int) -> float:
    """Efficiency ratio relative to the per-run benchmark median.

    Returns 1.0 at or below the median, degrades linearly to 0.5 at 4x median.
    Never below 0.5 — quality already penalizes bad answers separately.
    """
    if benchmark_median <= 0 or total_tokens <= 0:
        return 1.0
    ratio = total_tokens / benchmark_median
    if ratio <= 1.0:
        return 1.0
    return max(0.5, 1.0 - (ratio - 1.0) * (0.5 / 3.0))


def _is_api_error(jr: 'JudgeResult') -> bool:
    """Check if a judge result represents an API error/timeout rather than a real evaluation."""
    return (jr.failure_type or "") in ("API Error", "Timeout")


def platform_consistency_rate(
    model_answers: dict[str, str],
    ground_truth_keys: dict,
) -> float:
    """Fraction of model pairs that agree on factual claims.

    Uses ground truth keys when available (CSV mode), falls back to
    extracting prices from responses for comparison (discovery mode).
    """
    models = list(model_answers.keys())
    if len(models) < 2:
        return 1.0
    agreeing = 0
    total = 0
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            total += 1
            a_text = model_answers[models[i]]
            b_text = model_answers[models[j]]
            a_lower = a_text.strip().lower()
            b_lower = b_text.strip().lower()
            if a_lower == b_lower:
                agreeing += 1
            elif _fuzzy_fact_match(a_text, b_text, ground_truth_keys):
                agreeing += 1
    return agreeing / total if total > 0 else 1.0


def _extract_prices(text: str) -> set[float]:
    """Extract dollar amounts from text as normalized floats for comparison."""
    raw = re.findall(r'\$[\d,]+(?:\.\d{1,2})?', text)
    prices: set[float] = set()
    for p in raw:
        try:
            prices.add(float(p.replace('$', '').replace(',', '')))
        except ValueError:
            continue
    return prices


def _is_likely_product_name(name: str, brand_name: str = "") -> bool:
    """Filter out generic terms and non-product-name patterns.

    Product names in e-commerce typically look like:
      "Petra Curved Shoulder Bag", "Aelin Kitten-Heel Mules"
    NOT like:
      "Available Colors", "Add to bag", "Browse all product categories"
    """
    stripped = name.strip()
    lower = stripped.lower()

    # Too short or too long
    if len(stripped) < 8 or len(stripped) > 80:
        return False

    # Must have at least 3 words (e.g. "Gabine Saddle Bag")
    words = stripped.split()
    if len(words) < 3:
        return False

    # Contains digits — likely a price, size, or dimension
    if re.search(r'\d', stripped):
        return False

    # Ends with colon — it's a heading/label, not a product name
    if stripped.endswith(':') or stripped.endswith('):'):
        return False

    # Contains parenthetical with non-color content — likely a description
    paren = re.search(r'\(([^)]+)\)', stripped)
    if paren:
        inner = paren.group(1).lower()
        # Allow color parentheticals like "(Black)" or "(Beige Textured)"
        # Reject descriptive ones like "(Applies to both)" or "(this M size)"
        if len(inner.split()) > 3 or any(w in inner for w in ("apply", "both", "size", "this", "note")):
            return False

    # Starts with common non-product prefixes (includes UI/CTA verbs and gerunds)
    _BAD_STARTS = (
        "a ", "an ", "the ", "all ", "if ", "is ", "it ", "no ", "or ", "my ",
        "do ", "are ", "was ", "has ", "for ", "not ", "but ", "how ", "why ",
        "this ", "that ", "each ", "here ", "best ", "bold ", "dark ", "light ",
        "exact ", "key ", "main ", "special ", "additional ", "design ",
        "available ", "overall ", "quick ",
        # UI/CTA action verbs
        "add ", "browse ", "check ", "shop ", "view ", "sign ", "log ",
        "search ", "visit ", "go ", "see ", "try ", "click ", "tap ",
        "select ", "choose ", "find ", "discover ", "explore ", "filter ",
        # Gerund forms (LLM narration)
        "checking ", "browsing ", "shopping ", "searching ", "looking ",
        "visiting ", "showing ", "featuring ", "including ", "offering ",
        "according ", "based ", "comparing ", "considering ",
        # E-commerce category starts
        "accessories ", "clothing ", "footwear ", "jewelry ", "eyewear ",
    )
    if any(lower.startswith(p) for p in _BAD_STARTS):
        return False

    # Full-phrase blocklist for multi-word non-product patterns
    _BAD_PHRASES = (
        "add to bag", "add to cart", "shop now", "browse all", "view details",
        "sign up", "log in", "united states", "free shipping", "out of stock",
        "back in stock", "notify me", "sold out", "see more", "read more",
        "learn more", "view all", "shop all",
    )
    if any(p in lower for p in _BAD_PHRASES):
        return False

    # Contains website/page metadata terms — not a product name
    _SITE_TERMS = ("website", " site", "homepage", " page", " url", "online store")
    if any(t in lower for t in _SITE_TERMS):
        return False

    # Common non-product phrases (attribute labels, section headers, generic terms)
    _BLOCKLIST_WORDS = {
        "colors", "sizes", "dimensions", "details", "features", "instructions",
        "shipping", "policy", "reviews", "guide", "chart", "stock", "strap",
        "edition", "arrival", "seller", "sale", "heels", "hues", "silhouettes",
        "price", "prices", "applies", "materials",
        # Category/section terms
        "accessories", "categories", "category", "collection", "collections",
        "products", "product", "items", "section", "department",
        # Descriptive/analytical terms
        "affordability", "trendy", "styles", "trends", "quality", "overview",
        "comparison", "coverage", "summary", "analysis", "options",
        "selection", "variety", "lineup",
        # Error/placeholder terms
        "unavailable", "unknown", "information", "error", "offline",
        # UI/action terms
        "bag", "cart", "browse", "store", "official",
    }
    # If more than half the words are blocklisted, reject
    clean_words = [w.lower().rstrip(':,;').strip('()') for w in words]
    blocked = sum(1 for w in clean_words if w in _BLOCKLIST_WORDS)
    if blocked >= len(words) / 2:
        return False

    # If the name is essentially just the brand name (with minor additions), reject
    if brand_name:
        brand_lower = brand_name.lower().strip()
        remaining = lower.replace(brand_lower, "").strip(" &-")
        remaining_words = [w for w in remaining.split() if w.strip("&-")]
        if len(remaining_words) <= 1:
            return False

    # Must contain at least one word that looks like a proper noun (capitalized, not
    # a common English word). This helps distinguish "Petra Curved Shoulder Bag"
    # from "Available Colors And Sizes"
    _COMMON_CAPS = {
        "Available", "Additional", "Best", "Bold", "Block", "Dark", "Design",
        "Exact", "Free", "Key", "Light", "Limited", "Main", "New", "Overall",
        "Quick", "Return", "Special", "Standard",
    }
    has_proper = any(
        w[0].isupper() and w not in _COMMON_CAPS
        for w in words if len(w) > 1
    )
    if not has_proper:
        return False

    return True


def _extract_product_names(text: str, brand_name: str = "") -> set[str]:
    """Extract likely product names from various LLM formatting styles."""
    # Match **Product Name** (bold markdown) — common in LLM responses
    bold = set(re.findall(r'\*\*([A-Z][^*]{3,60})\*\*', text))
    # Also match product names after bullets, numbers, or pipe separators
    listed = set(re.findall(
        r'(?:^|\||\d+\.\s+|-\s+)\s*([A-Z][A-Za-z\s&\-\']{5,60}?)(?:\s*[\|\-\(]|\s*$)',
        text, re.MULTILINE,
    ))
    # Filter out generic terms and non-product patterns
    return {n.strip() for n in (bold | listed) if _is_likely_product_name(n, brand_name)}


def _fuzzy_fact_match(a: str, b: str, ground_truth_keys: dict) -> bool:
    """Check if both responses contain the same key facts.

    CSV mode: checks if ground truth values appear in both responses.
    Discovery mode: extracts and compares prices and product names.
    """
    # Try ground truth keys first (works in CSV mode)
    has_real_keys = False
    for key, value in ground_truth_keys.items():
        if key in ("evaluation_mode", "brand_url", "trap_type", "required_attributes"):
            continue
        has_real_keys = True
        val_str = str(value).lower()
        if val_str in a.lower() and val_str in b.lower():
            return True

    if has_real_keys:
        return False

    # Discovery mode fallback: compare extracted prices and product names
    prices_a = _extract_prices(a)
    prices_b = _extract_prices(b)
    names_a = _extract_product_names(a)
    names_b = _extract_product_names(b)

    # Require at least 1 shared price AND 1 shared product name
    shared_prices = prices_a & prices_b
    shared_names = names_a & names_b

    if shared_prices and shared_names:
        return True

    return False


# ------------------------------------------------------------------
# Aggregate scoring
# ------------------------------------------------------------------

@dataclass
class CategoryScores:
    """Aggregated scores for a single category."""
    category: str
    agent_a_mean: float = 0.0
    agent_b_mean: float = 0.0
    agent_a_ci: tuple[float, float] = (0.0, 0.0)
    agent_b_ci: tuple[float, float] = (0.0, 0.0)
    delta: float = 0.0
    agent_a_cov: float = 0.0
    agent_b_cov: float = 0.0
    agent_a_scores: list[float] | None = None
    agent_b_scores: list[float] | None = None


@dataclass
class TokenStats:
    """Token and latency aggregates."""
    avg_prompt_tokens_a: float = 0.0
    avg_prompt_tokens_b: float = 0.0
    avg_completion_tokens_a: float = 0.0
    avg_completion_tokens_b: float = 0.0
    avg_total_tokens_a: float = 0.0
    avg_total_tokens_b: float = 0.0
    avg_latency_a: float = 0.0
    avg_latency_b: float = 0.0
    token_reduction_pct: float = 0.0
    latency_reduction_pct: float = 0.0
    estimated_monthly_savings: float = 0.0
    # Benchmark: median total tokens across all models (auto-calibrated per run)
    benchmark_median_total: float = 0.0
    benchmark_median_completion: float = 0.0


@dataclass
class ModelCategoryScore:
    """Score for a single (model, category) pair."""
    model_key: str
    category: str
    mean_score: float = 0.0
    num_queries: int = 0


@dataclass
class AuditScores:
    """Complete scoring output for the report."""
    category_scores: dict[str, CategoryScores]
    token_stats: TokenStats
    agentic_readiness_score: float
    readiness_level: str  # "critical", "at-risk", "competitive"
    top_failures: list[dict]
    hallucination_details: list[dict]
    platform_consistency_a: float
    platform_consistency_b: float
    consistency_details: dict
    per_model_scores: dict[str, dict[str, float]] | None = None  # model_key -> category -> mean Agent A score
    per_model_scores_b: dict[str, dict[str, float]] | None = None  # model_key -> category -> mean Agent B score
    per_model_failures: dict[str, dict[str, int]] | None = None  # model_key -> failure_type -> count
    per_model_missing: dict[str, dict[str, int]] | None = None  # model_key -> missing_info_type -> count
    per_model_token_stats: dict[str, dict[str, float]] | None = None  # model_key -> metric -> value
    per_model_query_count: dict[str, int] | None = None  # model_key -> total judge results
    tokens_per_correct_fact_a: float = 0.0
    tokens_per_correct_fact_b: float = 0.0
    # Per-model attribute completeness: model_key -> {attr -> {"present_rate", "specific_rate"}}
    per_model_attribute_coverage: dict[str, dict[str, dict[str, float]]] | None = None
    # Before/after examples: best A/B comparisons showing structured data impact
    comparison_examples: list[dict] | None = None
    # Product-level vulnerability analysis (CSV mode only)
    product_vulnerabilities: list[dict] | None = None
    # Auto-generated recommendations
    recommendations: list[dict] | None = None
    # Per-model failure examples with actual response quotes
    per_model_failure_examples: dict[str, list[dict]] | None = None
    # Cross-platform disagreement examples with side-by-side snippets
    consistency_examples: list[dict] | None = None
    # Per-model catalog coverage: model_key -> {unique_products_mentioned, product_names}
    per_model_catalog_coverage: dict[str, dict] | None = None
    # Per-layer aggregate scores (discovery mode only): "discoverability" -> 0.72
    layer_scores: dict[str, float] | None = None
    # Readiness score breakdown: list of {category, label, weight, score, weighted}
    readiness_breakdown: list[dict] | None = None
    # Per-category sample sizes: {category: {model_key: count}}
    sample_sizes: dict[str, dict[str, int]] | None = None
    # Agentic actionability: per-model pass/fail rate for checkout confidence
    agentic_actionability: dict[str, float] | None = None
    agentic_actionability_overall: float = 0.0
    agentic_actionability_failures: list[dict] | None = None
    # Cross-model discrepancy analysis
    cross_model_conflict_matrix: list[dict] | None = None
    fragmentation_summary: dict[str, Any] | None = None
    # Dynamic per-category "bad" descriptions derived from actual audit failures
    category_failure_summaries: dict[str, str] | None = None
    # Dynamic per-layer failure summaries (e.g., "Top issues: Hedge Detected (3x), Wrong Price (2x)")
    layer_failure_summaries: dict[str, str] | None = None
    # Dynamic consistency summary from actual cross-model disagreements
    consistency_summary: str = ""
    # Per-model source citation rate: model_key -> fraction of responses citing brand URL
    per_model_source_citation_rate: dict[str, float] | None = None
    # Reliability scoring (discovery mode only) — set externally after compute_scores()
    reliability_scores: Any = None


def _blind_discovery_score(scores: dict) -> float:
    """Score a blind discovery response, gated on whether the target brand surfaced.

    If the brand was not surfaced, the score is capped at 0.15 regardless of
    how good the other recommendations were. The whole point of blind discovery
    is brand surfacing.
    """
    brand_surfaced = scores.get("brand_surfaced", False)
    fa = _to_float(scores.get("factual_accuracy", 0.0))

    if not brand_surfaced:
        return min(fa, 0.15)

    # Brand WAS surfaced — score based on quality of brand recommendations
    brand_quality = _to_float(scores.get("brand_recommendation_quality", 0.0))
    ranks = scores.get("ranking_scores")
    if ranks and isinstance(ranks, list):
        ndcg = ndcg_at_k([_to_float(s) for s in ranks], 3)
        return 0.4 * ndcg + 0.3 * brand_quality + 0.3 * fa
    return 0.5 * brand_quality + 0.5 * fa


def compute_scores(
    judge_results: list[JudgeResult],
    agent_a_responses: dict[str, AgentResponse],
    agent_b_responses: dict[str, AgentResponse],
    consistency_a_responses: dict[str, dict[str, AgentResponse]] | None = None,
    consistency_b_responses: dict[str, dict[str, AgentResponse]] | None = None,
    queries: list | None = None,
    discovery_mode: bool = False,
    cross_model_comparisons: list[CrossModelComparison] | None = None,
    brand_name: str = "",
    catalog_products: dict[str, list[str]] | None = None,
) -> AuditScores:
    """Compute all aggregate scores from judge results and raw responses."""

    # Group judge results by category
    by_category: dict[str, list[JudgeResult]] = {}
    for jr in judge_results:
        by_category.setdefault(jr.category, []).append(jr)

    category_scores: dict[str, CategoryScores] = {}

    # --- Score each category ---
    for cat, results_raw in by_category.items():
        # Filter out API errors/timeouts — they measure infrastructure, not brand readiness
        results = [r for r in results_raw if not _is_api_error(r)]
        if not results:
            results = results_raw  # If ALL failed, keep them so the category isn't empty

        a_scores = [_to_float(r.agent_a_scores.get("factual_accuracy", 0.0)) for r in results]
        b_scores = [_to_float(r.agent_b_scores.get("factual_accuracy", 0.0)) for r in results]

        if cat == INVENTORY_OOS:
            a_scores = [_to_float(r.agent_a_scores.get("oos_handling")) for r in results]
            b_scores = [_to_float(r.agent_b_scores.get("oos_handling")) for r in results]
        elif cat == OCCASION_REASONING:
            a_scores = [_to_float(r.agent_a_scores.get("constraint_adherence")) for r in results]
            b_scores = [_to_float(r.agent_b_scores.get("constraint_adherence")) for r in results]
        elif cat == BLIND_DISCOVERY:
            # Brand surfacing gate — the primary signal is whether the brand appeared
            a_scores = []
            b_scores = []
            for r in results:
                a_scores.append(_blind_discovery_score(r.agent_a_scores))
                b_scores.append(_blind_discovery_score(r.agent_b_scores))
        elif cat in (MULTI_PRODUCT_COMPARISON, PRODUCT_COMPARISON):
            # Use nDCG@3 from judge ranking_scores; fall back to factual_accuracy
            a_scores = []
            b_scores = []
            for r in results:
                a_ranks = r.agent_a_scores.get("ranking_scores")
                b_ranks = r.agent_b_scores.get("ranking_scores")
                if a_ranks and isinstance(a_ranks, list):
                    a_scores.append(ndcg_at_k([_to_float(s) for s in a_ranks], 3))
                else:
                    a_scores.append(_to_float(r.agent_a_scores.get("factual_accuracy", 0.0)))
                if b_ranks and isinstance(b_ranks, list):
                    b_scores.append(ndcg_at_k([_to_float(s) for s in b_ranks], 3))
                else:
                    b_scores.append(_to_float(r.agent_b_scores.get("factual_accuracy", 0.0)))
        elif cat == ADVERSARIAL:
            if discovery_mode:
                # Discovery mode: judge scores refusal quality on 0-1 scale via factual_accuracy
                a_scores = [_to_float(r.agent_a_scores.get("factual_accuracy", 0.0)) for r in results]
                b_scores = [_to_float(r.agent_b_scores.get("factual_accuracy", 0.0)) for r in results]
            else:
                # CSV mode: binary — no hallucinations = 1.0, any hallucination = 0.0
                a_scores = [1.0 if not r.agent_a_scores.get("hallucinations") else 0.0 for r in results]
                b_scores = [1.0 if not r.agent_b_scores.get("hallucinations") else 0.0 for r in results]
        elif cat == ATTRIBUTE_COMPLETENESS:
            # Score = fraction of attributes with specific=true (judge sets factual_accuracy to this)
            a_scores = [_to_float(r.agent_a_scores.get("factual_accuracy", 0.0)) for r in results]
            b_scores = [_to_float(r.agent_b_scores.get("factual_accuracy", 0.0)) for r in results]
        elif cat == TOKEN_EFFICIENCY:
            # Token efficiency = purely how many tokens were consumed.
            # The benchmark is the median total tokens across all models for
            # token-efficiency queries in this audit run (auto-calibrated).
            # Score is 1.0 at or below median, degrades for higher usage.
            a_scores = []
            b_scores = []

            # Compute per-query benchmark median from all models' responses
            query_ids = {r.query_id for r in results}
            all_token_counts: list[int] = []
            for qid in query_ids:
                for key, resp in agent_a_responses.items():
                    if key.startswith(qid) and resp.total_tokens > 0:
                        all_token_counts.append(resp.total_tokens)
            benchmark_median = int(np.median(all_token_counts)) if all_token_counts else 1

            for r in results:
                key_a = f"{r.query_id}_{r.model_key}"
                resp_a = agent_a_responses.get(key_a)
                resp_b = agent_b_responses.get(key_a)
                tokens_a = resp_a.total_tokens if resp_a else 0
                tokens_b = resp_b.total_tokens if resp_b else 0

                a_scores.append(_benchmark_efficiency(tokens_a, benchmark_median))
                b_scores.append(_benchmark_efficiency(tokens_b, benchmark_median))

        a_mean = float(np.mean(a_scores)) if a_scores else 0.0
        b_mean = float(np.mean(b_scores)) if b_scores else 0.0

        category_scores[cat] = CategoryScores(
            category=cat,
            agent_a_mean=a_mean,
            agent_b_mean=b_mean,
            agent_a_ci=bootstrap_ci(a_scores) if a_scores else (0.0, 0.0),
            agent_b_ci=bootstrap_ci(b_scores) if b_scores else (0.0, 0.0),
            delta=b_mean - a_mean,
            agent_a_cov=coeff_of_variation(a_scores),
            agent_b_cov=coeff_of_variation(b_scores),
            agent_a_scores=a_scores,
            agent_b_scores=b_scores,
        )

    # --- Token stats ---
    a_resps = list(agent_a_responses.values())
    b_resps = list(agent_b_responses.values())

    token_stats = TokenStats(
        avg_prompt_tokens_a=float(np.mean([r.prompt_tokens for r in a_resps])) if a_resps else 0,
        avg_prompt_tokens_b=float(np.mean([r.prompt_tokens for r in b_resps])) if b_resps else 0,
        avg_completion_tokens_a=float(np.mean([r.completion_tokens for r in a_resps])) if a_resps else 0,
        avg_completion_tokens_b=float(np.mean([r.completion_tokens for r in b_resps])) if b_resps else 0,
        avg_total_tokens_a=float(np.mean([r.total_tokens for r in a_resps])) if a_resps else 0,
        avg_total_tokens_b=float(np.mean([r.total_tokens for r in b_resps])) if b_resps else 0,
        avg_latency_a=float(np.mean([r.latency_seconds for r in a_resps])) if a_resps else 0,
        avg_latency_b=float(np.mean([r.latency_seconds for r in b_resps])) if b_resps else 0,
    )

    if token_stats.avg_total_tokens_a > 0 and b_resps:
        token_stats.token_reduction_pct = (
            (token_stats.avg_total_tokens_a - token_stats.avg_total_tokens_b)
            / token_stats.avg_total_tokens_a * 100
        )
        token_stats.latency_reduction_pct = (
            (token_stats.avg_latency_a - token_stats.avg_latency_b)
            / token_stats.avg_latency_a * 100
        ) if token_stats.avg_latency_a > 0 else 0.0

        token_delta = token_stats.avg_total_tokens_a - token_stats.avg_total_tokens_b
        token_stats.estimated_monthly_savings = (
            token_delta * (1_000_000 / 1000) * cfg.TOKEN_COST_PER_1K
        )

    # --- Per-model token stats ---
    per_model_token_stats: dict[str, dict[str, float]] = {}
    by_model_resps: dict[str, list[AgentResponse]] = {}
    for key, resp in agent_a_responses.items():
        mk = key.rsplit("_", 1)[1]
        by_model_resps.setdefault(mk, []).append(resp)

    # Compute cross-model benchmark medians (total tokens = input + output)
    all_valid_resps = [r for r in a_resps if r.total_tokens > 0]
    benchmark_median_total = float(np.median([r.total_tokens for r in all_valid_resps])) if all_valid_resps else 0.0
    benchmark_median_completion = float(np.median([r.completion_tokens for r in all_valid_resps])) if all_valid_resps else 0.0
    token_stats.benchmark_median_total = benchmark_median_total
    token_stats.benchmark_median_completion = benchmark_median_completion

    for mk, resps in by_model_resps.items():
        valid = [r for r in resps if r.total_tokens > 0]
        if valid:
            avg_prompt = float(np.mean([r.prompt_tokens for r in valid]))
            avg_total = float(np.mean([r.total_tokens for r in valid]))
            avg_completion = float(np.mean([r.completion_tokens for r in valid]))
            avg_latency = float(np.mean([r.latency_seconds for r in valid]))
            # Benchmark uses total tokens (input + output — reflects real consumption including web search)
            vs_benchmark = avg_total / benchmark_median_total if benchmark_median_total > 0 else 0.0
            # Per-model factual accuracy
            model_jrs = [jr for jr in judge_results if jr.model_key == mk and not _is_api_error(jr)]
            model_factual = float(np.mean([
                _to_float(jr.agent_a_scores.get("factual_accuracy", 0)) for jr in model_jrs
            ])) if model_jrs else 0.0
            # Accuracy per second — primary efficiency metric
            accuracy_per_second = round(model_factual / avg_latency, 4) if avg_latency > 0 else 0.0
            # Real per-provider cost using actual API pricing
            pricing = cfg.PROVIDER_PRICING.get(mk, {"input": 0.0, "output": 0.0})
            est_cost = (avg_prompt * pricing["input"] + avg_completion * pricing["output"]) / 1_000_000
            per_model_token_stats[mk] = {
                "avg_prompt_tokens": avg_prompt,
                "avg_completion_tokens": avg_completion,
                "avg_total_tokens": avg_total,
                "avg_latency": avg_latency,
                "num_queries": len(valid),
                "vs_benchmark": round(vs_benchmark, 2),
                "completion_per_correct_fact": round(avg_completion / model_factual, 0) if model_factual > 0 else 0.0,
                "est_cost_per_query": round(est_cost, 4),
                "accuracy_per_second": accuracy_per_second,
                "model_accuracy": round(model_factual, 3),
            }

    # --- Top failures ---
    # Build a lookup for target products per query_id
    _query_products: dict[str, list[str]] = {}
    if queries:
        for q in queries:
            if hasattr(q, "target_products") and q.target_products:
                _query_products[q.query_id] = q.target_products

    _FAILURE_ROOT_CAUSES = {
        # Discovery / general
        "Fabricated Product": "LLM generated plausible-sounding product details without verifying against the brand's actual catalog.",
        "Wrong Price": "LLM bypassed the official site and scraped secondary market pricing, or hallucinated a price from training data.",
        "Missing Price": "The brand's pricing data isn't indexed in the LLM's training data or web search results.",
        "Missing Details": "Product attributes aren't structured in a way that LLMs can reliably extract.",
        "Vague Response": "LLM lacked specific product data and fell back to generic category descriptions.",
        "Wrong Brand": "LLM confused this brand with a similarly-named competitor or unrelated company.",
        "Generic Response": "LLM had no brand-specific knowledge and produced boilerplate advice.",
        "Constraint Violation": "LLM ignored the customer's budget or occasion constraints when recommending products.",
        # CSV ground-truth
        "Ghost Inventory": "LLM claimed a product is in stock without access to real-time inventory data.",
        "Wrong Sizing": "LLM applied the wrong size system for this product type (e.g., clothing sizes for bags).",
        "Wrong Care": "LLM fabricated care instructions that don't match the product's actual material or label.",
        "Non-existent Feature": "LLM invented a product feature or technology that doesn't exist for this item.",
        "Wrong Policy": "LLM stated incorrect return, shipping, or warranty policy details for this brand.",
        # Temporal freshness
        "Stale Data": "LLM's training data is outdated and doesn't reflect current catalog or pricing.",
        "No Freshness Caveat": "LLM presented potentially outdated information as current without hedging.",
        "Fabricated Date": "LLM invented specific dates for sales, launches, or restocks with no factual basis.",
        "Invented Promotion": "LLM fabricated a sale, discount, or promotional event that doesn't exist.",
        "Fake Restock": "LLM claimed an item would be restocked on a specific date without inventory access.",
        # Attribute completeness
        "Missing Material": "The product's material or fabric composition isn't available in LLM-accessible sources.",
        "Missing Sizes": "Size availability data isn't structured for LLM extraction.",
        "Missing Colors": "Color/variant data isn't surfaced in a way LLMs can reliably retrieve.",
        "Missing Availability": "The agent did not address whether the product is currently available for purchase.",
        "Missing Care": "Care instruction data isn't included in the brand's public product pages.",
        # Brand knowledge
        "Wrong Product Details": "LLM cited incorrect specifications, materials, or features for a real product.",
        "Missing Technology Knowledge": "LLM had no awareness of the brand's proprietary technologies or materials.",
        "Wrong Brand Values": "LLM mischaracterized the brand's mission, values, or market positioning.",
        "Missing Collection Info": "LLM couldn't identify the brand's named product lines or seasonal collections.",
        "Wrong Competitive Positioning": "LLM incorrectly placed the brand relative to its actual competitors.",
        # Infrastructure
        "API Error": "The model's API returned an error or was unreachable.",
        "Timeout": "The model took too long to respond and was terminated.",
    }

    def _failure_root_cause(failure_type: str, judge_narrative: str = "") -> str:
        """Return a root cause description, preferring the judge's specific narrative."""
        if judge_narrative:
            return judge_narrative
        return _FAILURE_ROOT_CAUSES.get(failure_type, "")

    _FAILURE_FILTER_PATTERNS = (
        "was not run", "baseline", "n/a", "did not provide",
        "agent b", "not run", "no response",
    )
    all_failures = []
    for jr in judge_results:
        if not jr.most_significant_failure or not jr.failure_type:
            continue
        # Filter out API errors/timeouts (infrastructure, not brand issues)
        if _is_api_error(jr):
            continue
        # Filter out fake failures about Agent B not being available
        fail_lower = jr.most_significant_failure.lower()
        ftype_lower = jr.failure_type.lower()
        if any(p in fail_lower for p in _FAILURE_FILTER_PATTERNS):
            continue
        if ftype_lower in ("n/a", "na", "none"):
            continue
        # Skip high-accuracy results — a 70%+ score is a partial success, not a critical failure
        fa_score = _to_float(jr.agent_a_scores.get("factual_accuracy", 0.0))
        if fa_score >= 0.7:
            continue
        all_failures.append({
            "query_id": jr.query_id,
            "prompt": jr.prompt,
            "failure": jr.most_significant_failure,
            "failure_type": jr.failure_type,
            "category": jr.category,
            "model_key": jr.model_key or "",
            "target_products": _query_products.get(jr.query_id, []),
            "root_cause": _failure_root_cause(jr.failure_type, jr.most_significant_failure),
        })
    # --- Per-category failure summaries for dynamic "bad" descriptions ---
    _cat_failures_raw: dict[str, list[str]] = {}
    _cat_failure_types: dict[str, list[str]] = {}
    for f in all_failures:
        cat = f.get("category", "")
        narrative = f.get("failure", "")
        ftype = f.get("failure_type", "")
        if cat and narrative:
            _cat_failures_raw.setdefault(cat, []).append(narrative)
        if cat and ftype:
            _cat_failure_types.setdefault(cat, []).append(ftype)
    category_failure_summaries: dict[str, str] = {}
    for cat, narratives in _cat_failures_raw.items():
        best = max(narratives, key=len)
        if len(best) > 200:
            best = best[:197] + "..."
        category_failure_summaries[cat] = best

    # --- Per-layer failure summaries for dynamic layer descriptions ---
    _l1_cats = {TABLE_STAKES, OCCASION_REASONING, BLIND_DISCOVERY, PRODUCT_COMPARISON,
                CROSS_PLATFORM_CONSISTENCY, TOKEN_EFFICIENCY}
    _l2_cats = {BRAND_KNOWLEDGE, ADVERSARIAL, ATTRIBUTE_COMPLETENESS, TEMPORAL_FRESHNESS}

    def _build_layer_summary(layer_cats: set[str]) -> str:
        """Build a summary from the top failure types across a layer's categories."""
        ftypes: list[str] = []
        for cat in layer_cats:
            ftypes.extend(_cat_failure_types.get(cat, []))
        if not ftypes:
            return ""
        # Count and rank failure types
        from collections import Counter
        counts = Counter(ftypes).most_common(3)
        parts = [f"{ft} ({n}x)" for ft, n in counts]
        return "Top issues: " + ", ".join(parts)

    layer_failure_summaries: dict[str, str] = {
        LAYER_1_DISCOVERABILITY: _build_layer_summary(_l1_cats),
        LAYER_2_BRAND_DNA: _build_layer_summary(_l2_cats),
    }

    # --- Consistency summary from actual disagreement examples ---
    consistency_summary: str = ""
    if cross_model_comparisons:
        diffs: list[str] = []
        for comp in cross_model_comparisons:
            if comp.key_differences and comp.key_differences != ["Comparison could not be generated"]:
                for d in comp.key_differences[:2]:
                    if len(d) <= 150:
                        diffs.append(d)
        if diffs:
            consistency_summary = diffs[0]

    # Collect which models failed on each prompt (for display)
    prompt_to_models: dict[str, set[str]] = {}
    for f in all_failures:
        pk = f["prompt"][:80]
        if f.get("model_key"):
            prompt_to_models.setdefault(pk, set()).add(f["model_key"])

    # Deduplicate by prompt AND product — ensure diverse failures
    seen_prompts: set[str] = set()
    product_counts: dict[str, int] = {}
    deduped_failures: list[dict] = []
    for f in sorted(all_failures, key=lambda f: len(f.get("failure", "")), reverse=True):
        prompt_key = f["prompt"][:80]
        products = f.get("target_products", [])
        product_key = products[0] if products else ""
        # Skip if same prompt already shown
        if prompt_key in seen_prompts:
            continue
        # Skip if same product already has 2+ failures (ensure diversity)
        if product_key and product_counts.get(product_key, 0) >= 2:
            continue
        seen_prompts.add(prompt_key)
        if product_key:
            product_counts[product_key] = product_counts.get(product_key, 0) + 1
        # Attach all models that failed on this prompt
        f["failed_models"] = sorted(prompt_to_models.get(prompt_key, set()))
        deduped_failures.append(f)
    top_failures = deduped_failures[:5]

    # --- Hallucination deep dive ---
    hallucination_details = []
    for jr in judge_results:
        for h in jr.agent_a_scores.get("hallucinations", []):
            hallucination_details.append({
                "query_id": jr.query_id,
                "prompt": jr.prompt,
                "suspicious_claim": h,
                "failure_type": jr.failure_type,
                "category": jr.category,
                "model_key": jr.model_key or "",
                "root_cause": _failure_root_cause(jr.failure_type, jr.most_significant_failure),
            })
    hallucination_details = hallucination_details[:10]

    # --- Platform consistency ---
    platform_consistency_a = 0.0
    platform_consistency_b = 0.0
    consistency_detail = {}

    # Build query_id -> ground_truth_keys lookup
    gt_keys_map: dict[str, dict] = {}
    if queries:
        for q in queries:
            gt_keys_map[q.query_id] = q.ground_truth_keys

    if consistency_a_responses:
        # consistency_a_responses: {query_id: {model_key: AgentResponse}}
        rates_a = []
        pair_agree_a: dict[tuple[str, str], list[int]] = {}  # (m1, m2) -> [0/1 per query]
        for qid, model_resps in consistency_a_responses.items():
            answers = {m: r.text for m, r in model_resps.items() if not r.text.startswith("ERROR")}
            gt_keys = gt_keys_map.get(qid, {})
            if len(answers) >= 2:
                rates_a.append(platform_consistency_rate(answers, gt_keys))
                # Track per-pair agreement for the matrix
                models_list = sorted(answers.keys())
                for ii in range(len(models_list)):
                    for jj in range(ii + 1, len(models_list)):
                        pair = (models_list[ii], models_list[jj])
                        a_text = answers[models_list[ii]]
                        b_text = answers[models_list[jj]]
                        agreed = (
                            a_text.strip().lower() == b_text.strip().lower()
                            or _fuzzy_fact_match(a_text, b_text, gt_keys)
                        )
                        pair_agree_a.setdefault(pair, []).append(1 if agreed else 0)
        platform_consistency_a = float(np.mean(rates_a)) if rates_a else 0.0
        consistency_detail["agent_a_per_query"] = rates_a
        # Convert per-pair lists to rates
        consistency_detail["agent_a_per_pair"] = {
            f"{m1}_{m2}": float(np.mean(vals)) if vals else 0.0
            for (m1, m2), vals in pair_agree_a.items()
        }

    if consistency_b_responses:
        rates_b = []
        pair_agree_b: dict[tuple[str, str], list[int]] = {}
        for qid, model_resps in consistency_b_responses.items():
            answers = {m: r.text for m, r in model_resps.items() if not r.text.startswith("ERROR")}
            gt_keys = gt_keys_map.get(qid, {})
            if len(answers) >= 2:
                rates_b.append(platform_consistency_rate(answers, gt_keys))
                models_list = sorted(answers.keys())
                for ii in range(len(models_list)):
                    for jj in range(ii + 1, len(models_list)):
                        pair = (models_list[ii], models_list[jj])
                        a_text = answers[models_list[ii]]
                        b_text = answers[models_list[jj]]
                        agreed = (
                            a_text.strip().lower() == b_text.strip().lower()
                            or _fuzzy_fact_match(a_text, b_text, gt_keys)
                        )
                        pair_agree_b.setdefault(pair, []).append(1 if agreed else 0)
        platform_consistency_b = float(np.mean(rates_b)) if rates_b else 0.0
        consistency_detail["agent_b_per_query"] = rates_b
        consistency_detail["agent_b_per_pair"] = {
            f"{m1}_{m2}": float(np.mean(vals)) if vals else 0.0
            for (m1, m2), vals in pair_agree_b.items()
        }

    # --- Cross-platform disagreement examples ---
    # Only use judge-produced cross-model comparisons (no heuristic fallback).
    # These are higher quality: the judge produces 2-3 sentence summaries per
    # model and holistic key differences (not just pricing).
    consistency_examples: list[dict] = []

    if cross_model_comparisons:
        query_map = {q.query_id: q for q in queries} if queries else {}
        for comp in cross_model_comparisons:
            # Skip failed comparisons
            if not comp.key_differences or comp.key_differences == ["Comparison could not be generated"]:
                continue
            q = query_map.get(comp.query_id)
            consistency_examples.append({
                "query_id": comp.query_id,
                "prompt": q.prompt if q else comp.prompt,
                "model_summaries": comp.model_summaries,
                "key_differences": comp.key_differences,
                })

        consistency_examples = consistency_examples[:5]

    # --- Cross-Model Discrepancy Analysis ---
    cross_model_conflict_data: list[dict] = []
    frag_counts = {"none": 0, "minor": 0, "major": 0, "critical": 0}

    if cross_model_comparisons:
        for comp in cross_model_comparisons:
            if comp.conflict_matrix:
                cross_model_conflict_data.append({
                    "query_id": comp.query_id,
                    "prompt": comp.prompt,
                    "conflict_matrix": comp.conflict_matrix,
                    "fragmentation_level": comp.fragmentation_level,
                    "model_summaries": comp.model_summaries,
                })
            if comp.fragmentation_level:
                frag_counts[comp.fragmentation_level] = frag_counts.get(comp.fragmentation_level, 0) + 1

    total_comparisons = len(cross_model_comparisons) if cross_model_comparisons else 0
    fragmentation_summary = {
        "total_comparisons": total_comparisons,
        "critical_fragmentation": frag_counts["critical"],
        "major_fragmentation": frag_counts["major"],
        "minor_fragmentation": frag_counts["minor"],
        "no_fragmentation": frag_counts["none"],
        "critical_fragmentation_rate": (
            frag_counts["critical"] / total_comparisons
            if total_comparisons > 0 else 0.0
        ),
    }

    # Penalize high fragmentation on cross-platform consistency score
    if fragmentation_summary["critical_fragmentation_rate"] > 0.5:
        platform_consistency_a = min(platform_consistency_a, 0.3)

    # --- Blended scoring helper ---
    # Most categories blend factual_accuracy (60%) + constraint_adherence (40%)
    # so that constraint violations (missing requested details) actually penalize.
    _FA_W, _CA_W = 0.6, 0.4

    def _category_score(scores: dict, category: str) -> float:
        """Compute the primary score for a category, blending metrics as appropriate."""
        if category == INVENTORY_OOS:
            return _to_float(scores.get("oos_handling"))
        if category == OCCASION_REASONING:
            return _to_float(scores.get("constraint_adherence"))
        if category == BLIND_DISCOVERY:
            return _blind_discovery_score(scores)
        if category in (MULTI_PRODUCT_COMPARISON, PRODUCT_COMPARISON):
            ranks = scores.get("ranking_scores")
            fa = _to_float(scores.get("factual_accuracy", 0.0))
            if ranks and isinstance(ranks, list):
                ndcg = ndcg_at_k([_to_float(s) for s in ranks], 3)
                return 0.5 * ndcg + 0.5 * fa
            return fa
        if category == ADVERSARIAL:
            if discovery_mode:
                return _to_float(scores.get("factual_accuracy", 0.0))
            return 1.0 if not scores.get("hallucinations") else 0.0
        # All other categories: blend factual_accuracy + constraint_adherence
        fa = _to_float(scores.get("factual_accuracy", 0.0))
        ca = _to_float(scores.get("constraint_adherence", 0.0))
        return _FA_W * fa + _CA_W * ca

    # --- Per-model scoring ---
    # Group judge results by (model_key, category) and extract the primary score
    per_model_scores: dict[str, dict[str, float]] = {}
    by_model_cat: dict[str, dict[str, list[float]]] = {}
    for jr in judge_results:
        if not jr.model_key or _is_api_error(jr):
            continue
        by_model_cat.setdefault(jr.model_key, {}).setdefault(jr.category, [])
        score = _category_score(jr.agent_a_scores, jr.category)
        by_model_cat[jr.model_key][jr.category].append(score)

    for model_key, cats in by_model_cat.items():
        per_model_scores[model_key] = {}
        for cat, scores_list in cats.items():
            per_model_scores[model_key][cat] = float(np.mean(scores_list)) if scores_list else 0.0

    # --- Per-model Agent B scoring ---
    per_model_scores_b: dict[str, dict[str, float]] = {}
    by_model_cat_b: dict[str, dict[str, list[float]]] = {}
    for jr in judge_results:
        if not jr.model_key or not jr.agent_b_scores or _is_api_error(jr):
            continue
        by_model_cat_b.setdefault(jr.model_key, {}).setdefault(jr.category, [])
        score_b = _category_score(jr.agent_b_scores, jr.category)
        by_model_cat_b[jr.model_key][jr.category].append(score_b)

    for model_key, cats in by_model_cat_b.items():
        per_model_scores_b[model_key] = {}
        for cat, scores_list in cats.items():
            per_model_scores_b[model_key][cat] = float(np.mean(scores_list)) if scores_list else 0.0

    # --- Per-model source citation rate ---
    # Fraction of judge results where source_cited == true (brand URL cited)
    per_model_source_citation_rate: dict[str, float] = {}
    _source_cite_counts: dict[str, list[bool]] = {}
    for jr in judge_results:
        if not jr.model_key or _is_api_error(jr):
            continue
        cited = bool(jr.agent_a_scores.get("source_cited", False))
        _source_cite_counts.setdefault(jr.model_key, []).append(cited)
    for mk, citations in _source_cite_counts.items():
        per_model_source_citation_rate[mk] = sum(citations) / len(citations) if citations else 0.0

    # --- Per-model failure analysis ---
    per_model_failures: dict[str, dict[str, int]] = {}
    per_model_missing: dict[str, dict[str, int]] = {}
    per_model_query_count: dict[str, int] = {}
    for jr in judge_results:
        if not jr.model_key:
            continue
        mk = jr.model_key
        per_model_query_count[mk] = per_model_query_count.get(mk, 0) + 1
        # Count failure types — only count genuine failures (low blended score, not API errors)
        if jr.failure_type and jr.failure_type.lower() not in ("none", "n/a", "na"):
            if not _is_api_error(jr):
                blended = _category_score(jr.agent_a_scores, jr.category)
                if blended < 0.7:
                    per_model_failures.setdefault(mk, {})
                    per_model_failures[mk][jr.failure_type] = per_model_failures[mk].get(jr.failure_type, 0) + 1
        # Count missing info types
        missing = jr.agent_a_scores.get("missing_info", [])
        if isinstance(missing, list):
            per_model_missing.setdefault(mk, {})
            for info_type in missing:
                if isinstance(info_type, str) and info_type.strip():
                    per_model_missing[mk][info_type] = per_model_missing[mk].get(info_type, 0) + 1

    # --- Per-model failure examples with actual response quotes ---
    _FAKE_FAILURE_PATTERNS = ("was not run", "baseline", "n/a", "did not provide a response", "agent b")

    per_model_failure_examples: dict[str, list[dict]] = {}
    for jr in judge_results:
        if not jr.model_key:
            continue
        mk = jr.model_key
        if jr.most_significant_failure and jr.failure_type and jr.failure_type.lower() != "none":
            # Skip fake failures about Agent B not being run or baseline artifacts
            fail_lower = jr.most_significant_failure.lower()
            ftype_lower = jr.failure_type.lower()
            if any(p in fail_lower for p in _FAKE_FAILURE_PATTERNS):
                continue
            if ftype_lower in ("n/a", "na"):
                continue
            resp_key = f"{jr.query_id}_{mk}"
            resp = agent_a_responses.get(resp_key)
            resp_text = resp.text if resp else ""
            blended = _category_score(jr.agent_a_scores, jr.category)
            per_model_failure_examples.setdefault(mk, []).append({
                "query_id": jr.query_id,
                "prompt": jr.prompt,
                "response_snippet": resp_text,
                "failure_type": jr.failure_type,
                "failure_quote": jr.most_significant_failure,
                "category": jr.category,
                "factual_accuracy": blended,
            })
    # Keep top 5 per model, worst accuracy first
    for mk in per_model_failure_examples:
        per_model_failure_examples[mk].sort(key=lambda x: x["factual_accuracy"])
        per_model_failure_examples[mk] = per_model_failure_examples[mk][:5]

    # --- Per-model catalog coverage ---
    per_model_catalog_coverage: dict[str, dict] = {}
    model_keys_seen = set(jr.model_key for jr in judge_results if jr.model_key)
    _model_product_sets: dict[str, set[str]] = {}
    if catalog_products:
        # Use LLM-extracted product names (accurate, brand-filtered)
        for mk in model_keys_seen:
            names = catalog_products.get(mk, [])
            _model_product_sets[mk] = set(names)
            per_model_catalog_coverage[mk] = {
                "unique_products_mentioned": len(names),
                "product_names": sorted(names)[:20],
            }
    else:
        # Fallback to regex extraction
        for mk in model_keys_seen:
            product_names_set: set[str] = set()
            for key, resp in agent_a_responses.items():
                if key.endswith(f"_{mk}") and not resp.text.startswith("ERROR"):
                    names = _extract_product_names(resp.text, brand_name)
                    product_names_set.update(names)
            _model_product_sets[mk] = product_names_set
            per_model_catalog_coverage[mk] = {
                "unique_products_mentioned": len(product_names_set),
                "product_names": sorted(product_names_set)[:20],
            }

    # Build product visibility matrix: which models mention each product
    _all_products: dict[str, set[str]] = {}
    for mk, pset in _model_product_sets.items():
        for name in pset:
            _all_products.setdefault(name, set()).add(mk)
    num_models = len(_model_product_sets)
    # Products known by all models (high confidence real products)
    consensus_products = sorted(
        [p for p, ms in _all_products.items() if len(ms) == num_models]
    ) if num_models > 1 else []
    # Products known by only one model (potential hallucinations or blind spots)
    single_model_products = sorted(
        [(p, next(iter(ms))) for p, ms in _all_products.items() if len(ms) == 1],
        key=lambda x: x[1],
    ) if num_models > 1 else []
    # Build per-product row for the visibility matrix (sorted by coverage desc, then name)
    product_visibility = []
    for name, models in sorted(_all_products.items(), key=lambda x: (-len(x[1]), x[0])):
        product_visibility.append({
            "name": name,
            "models": sorted(models),
            "coverage": len(models),
        })
    for mk_cov in per_model_catalog_coverage.values():
        mk_cov["consensus_count"] = len(consensus_products)
        mk_cov["single_model_count"] = len(single_model_products)
    # Attach visibility data to the coverage dict for the reporter
    per_model_catalog_coverage["_visibility"] = {
        "total_unique": len(_all_products),
        "consensus_products": consensus_products[:15],
        "single_model_products": single_model_products[:15],
        "product_visibility": product_visibility[:30],
        "num_models": num_models,
    }

    # --- Tokens per correct fact ---
    # Measures efficiency: how many tokens does the agent need per unit of accuracy?
    all_a_factual = [_to_float(jr.agent_a_scores.get("factual_accuracy", 0)) for jr in judge_results]
    all_b_factual = [_to_float(jr.agent_b_scores.get("factual_accuracy", 0)) for jr in judge_results if jr.agent_b_scores]
    avg_factual_a = float(np.mean(all_a_factual)) if all_a_factual else 0.0
    avg_factual_b = float(np.mean(all_b_factual)) if all_b_factual else 0.0
    tokens_per_correct_fact_a = (
        token_stats.avg_total_tokens_a / avg_factual_a if avg_factual_a > 0 else 0.0
    )
    tokens_per_correct_fact_b = (
        token_stats.avg_total_tokens_b / avg_factual_b if avg_factual_b > 0 else 0.0
    )

    # --- Per-model attribute completeness ---
    # For ATTRIBUTE_COMPLETENESS queries, extract per-attribute present/specific rates per model.
    # Uses code-level verification against brand research ground truth instead of trusting the
    # judge's accuracy judgment (which is unreliable for things like availability, prices).
    per_model_attribute_coverage: dict[str, dict[str, dict[str, float]]] = {}
    attr_results = [jr for jr in judge_results if jr.category == ATTRIBUTE_COMPLETENESS]

    # --- Helpers for code-level attribute verification ---
    _BAG_CATEGORIES = {"bag", "bags", "handbag", "handbags", "tote", "clutch", "backpack",
                       "shoulder bag", "crossbody", "satchel", "hobo", "wallet", "purse"}
    _CLOTHING_SIZE_PATTERN = re.compile(r'\b(X?S|S|M|L|X?L|XXL|XXXL|one size)\b', re.IGNORECASE)
    _SHOE_SIZE_PATTERN = re.compile(r'\b(US\s*\d|EU\s*\d|\d{1,2}\.?5?\s*(US|EU|UK))\b', re.IGNORECASE)

    def _parse_price(text: str) -> float | None:
        """Extract a single dollar price from a value_found string."""
        m = re.search(r'\$[\d,]+(?:\.\d{1,2})?', text)
        if m:
            try:
                return float(m.group().replace('$', '').replace(',', ''))
            except ValueError:
                pass
        return None

    def _parse_price_range(text: str) -> tuple[float, float] | None:
        """Parse a brand price range like '$39-$199' or '$39 - $199'."""
        m = re.search(r'\$?([\d,]+(?:\.\d{1,2})?)\s*[-–—to]+\s*\$?([\d,]+(?:\.\d{1,2})?)', text)
        if m:
            try:
                lo = float(m.group(1).replace(',', ''))
                hi = float(m.group(2).replace(',', ''))
                return (lo, hi)
            except ValueError:
                pass
        return None

    def _is_bag_category(cat: str) -> bool:
        return any(b in cat.lower() for b in _BAG_CATEGORIES)

    def _verify_attribute(attr: str, value_found: str, gt: dict) -> bool | None:
        """Code-level verification of an attribute value against ground truth.

        Returns True (verified accurate), False (verified inaccurate), or None (can't verify).
        """
        if not value_found:
            return None

        if attr == "availability":
            # Availability: can't reliably verify stock status in code —
            # websites change in real-time. Defer to consensus or judge.
            return None

        if attr == "price":
            claimed = _parse_price(value_found)
            if claimed is None:
                return None
            # Check against exact research price (±15%)
            research_price_str = gt.get("research_product_price", "")
            if research_price_str:
                research_price = _parse_price(str(research_price_str))
                if research_price and research_price > 0:
                    pct_diff = abs(claimed - research_price) / research_price
                    return pct_diff <= 0.15
            # Fall back to brand price range sanity check
            price_range_str = gt.get("brand_price_range", "")
            if price_range_str:
                rng = _parse_price_range(str(price_range_str))
                if rng:
                    lo, hi = rng
                    # Allow 20% margin outside the range for products at the edges
                    return claimed >= lo * 0.8 and claimed <= hi * 1.2
            return None

        if attr == "sizes":
            # If product is a bag, clothing sizes (S/M/L) or shoe sizes are wrong
            product_cat = gt.get("research_product_category", "")
            if product_cat and _is_bag_category(product_cat):
                has_clothing = bool(_CLOTHING_SIZE_PATTERN.search(value_found))
                has_shoe = bool(_SHOE_SIZE_PATTERN.search(value_found))
                if has_clothing or has_shoe:
                    return False  # Wrong size format for a bag
            return None

        # material, colors, care: can't reliably verify in code
        return None

    if attr_results:
        # Group by model, keeping query_id for ground truth lookup
        by_model_attr: dict[str, list[tuple[dict, dict]]] = {}  # mk -> [(coverage, gt_keys)]
        # Also collect per-query per-attribute values across ALL models for consensus
        per_query_attr_values: dict[str, dict[str, dict[str, str]]] = {}  # qid -> attr -> mk -> value
        for jr in attr_results:
            if not jr.model_key:
                continue
            coverage = jr.agent_a_scores.get("attribute_coverage", {})
            if not coverage:
                continue
            q = query_map.get(jr.query_id)
            gt = q.ground_truth_keys if q else {}
            by_model_attr.setdefault(jr.model_key, []).append((coverage, gt, jr.query_id))
            # Collect values for cross-model consensus
            for a in REQUIRED_ATTRIBUTES:
                cell = coverage.get(a, {})
                val = cell.get("value_found", "")
                if val and cell.get("specific", False):
                    per_query_attr_values.setdefault(jr.query_id, {}).setdefault(a, {})[jr.model_key] = val.strip().lower()

        def _consensus_check(qid: str, attr: str, mk: str) -> bool | None:
            """Check if this model's value agrees with the majority of other models.

            Returns True (consensus match), False (outlier), or None (not enough data).
            """
            attr_vals = per_query_attr_values.get(qid, {}).get(attr, {})
            if len(attr_vals) < 3:  # Need at least 3 models for meaningful consensus
                return None
            my_val = attr_vals.get(mk, "")
            if not my_val:
                return None
            # Count how many other models roughly agree with this value
            from difflib import SequenceMatcher
            agree_count = 0
            other_vals = {m: v for m, v in attr_vals.items() if m != mk}
            for other_val in other_vals.values():
                # Fuzzy match: >0.6 similarity or one contains the other
                if (my_val in other_val or other_val in my_val or
                        SequenceMatcher(None, my_val, other_val).ratio() > 0.6):
                    agree_count += 1
            # Consensus = majority of other models agree
            return agree_count >= len(other_vals) / 2

        for mk, entries in by_model_attr.items():
            per_model_attribute_coverage[mk] = {}
            for attr in REQUIRED_ATTRIBUTES:
                present_count = 0
                specific_count = 0
                accurate_count = 0
                code_verified_count = 0
                consensus_verified_count = 0
                judge_only_count = 0
                values_found: list[str] = []
                inaccurate_values: list[str] = []

                for coverage, gt, qid in entries:
                    cell = coverage.get(attr, {})
                    is_present = cell.get("present", False)
                    is_specific = cell.get("specific", False)
                    val = cell.get("value_found", "")

                    if is_present:
                        present_count += 1
                    if is_specific:
                        specific_count += 1
                    if val:
                        values_found.append(val)

                    # Code-level verification overrides judge's accuracy judgment
                    code_verdict = _verify_attribute(attr, val, gt) if is_specific else None
                    if code_verdict is True:
                        accurate_count += 1
                        code_verified_count += 1
                    elif code_verdict is False:
                        if val:
                            inaccurate_values.append(val)
                    else:
                        # Can't verify in code — try cross-model consensus
                        consensus = _consensus_check(qid, attr, mk) if is_specific else None
                        if consensus is True:
                            accurate_count += 1
                            consensus_verified_count += 1
                        elif consensus is False:
                            # Outlier: this model disagrees with majority
                            if val:
                                inaccurate_values.append(val)
                        else:
                            # No consensus possible — fall back to judge
                            if is_specific and cell.get("accurate", False):
                                accurate_count += 1
                                judge_only_count += 1

                n = len(entries)
                # Determine primary verification method for this attribute
                verified_by_code = code_verified_count > 0
                verified_by_consensus = consensus_verified_count > 0
                verified_by_judge_only = (not verified_by_code and not verified_by_consensus
                                          and judge_only_count > 0)
                if verified_by_code:
                    verification = "code"
                elif verified_by_consensus:
                    verification = "consensus"
                elif verified_by_judge_only:
                    verification = "judge"
                else:
                    verification = "none"

                per_model_attribute_coverage[mk][attr] = {
                    "present_rate": present_count / n if n > 0 else 0.0,
                    "specific_rate": specific_count / n if n > 0 else 0.0,
                    "accurate_rate": accurate_count / n if n > 0 else 0.0,
                    "present_count": present_count,
                    "specific_count": specific_count,
                    "accurate_count": accurate_count,
                    "total": n,
                    "values_found": values_found[:3],
                    "inaccurate_values": inaccurate_values[:3],
                    "verification": verification,
                    "code_verified": code_verified_count,
                    "consensus_verified": consensus_verified_count,
                    "judge_only": judge_only_count,
                }

    # --- Agentic Actionability ---
    # Per-query pass/fail: could an autonomous agent confidently complete a checkout?
    per_model_actionability: dict[str, list[bool]] = {}
    actionability_fail_examples: list[dict] = []

    # Build per-query consistency lookup from consistency_a_responses
    per_query_consistency: dict[str, float] = {}
    if consistency_a_responses:
        for qid, model_resps in consistency_a_responses.items():
            answers = {m: r.text for m, r in model_resps.items() if not r.text.startswith("ERROR")}
            gt_keys = gt_keys_map.get(qid, {})
            if len(answers) >= 2:
                per_query_consistency[qid] = platform_consistency_rate(answers, gt_keys)

    for jr in judge_results:
        if not jr.model_key or _is_api_error(jr):
            continue
        mk = jr.model_key
        resp_key = f"{jr.query_id}_{mk}"
        resp = agent_a_responses.get(resp_key)
        resp_text = resp.text if resp else ""

        cross_consistency = per_query_consistency.get(jr.query_id)
        is_actionable, fail_reasons = _query_is_actionable(jr, resp_text, cross_consistency)
        per_model_actionability.setdefault(mk, []).append(is_actionable)

        if not is_actionable and len(actionability_fail_examples) < 10:
            actionability_fail_examples.append({
                "query_id": jr.query_id,
                "model_key": mk,
                "prompt": jr.prompt,
                "category": jr.category,
                "failure_reasons": fail_reasons,
            })

    agentic_actionability = {
        mk: sum(vals) / len(vals) if vals else 0.0
        for mk, vals in per_model_actionability.items()
    }
    all_actionable = [v for vals in per_model_actionability.values() for v in vals]
    agentic_actionability_overall = (
        sum(all_actionable) / len(all_actionable) if all_actionable else 0.0
    )

    # --- Agentic Readiness Score ---
    factual = category_scores.get(TABLE_STAKES, CategoryScores(TABLE_STAKES)).agent_a_mean
    constraint = category_scores.get(OCCASION_REASONING, CategoryScores(OCCASION_REASONING)).agent_a_mean
    blind_discovery_score = category_scores.get(BLIND_DISCOVERY, CategoryScores(BLIND_DISCOVERY)).agent_a_mean
    product_comparison_score = category_scores.get(PRODUCT_COMPARISON, CategoryScores(PRODUCT_COMPARISON)).agent_a_mean

    attr_completeness = category_scores.get(ATTRIBUTE_COMPLETENESS, CategoryScores(ATTRIBUTE_COMPLETENESS)).agent_a_mean

    temporal = category_scores.get(TEMPORAL_FRESHNESS, CategoryScores(TEMPORAL_FRESHNESS)).agent_a_mean

    adversarial = category_scores.get(ADVERSARIAL, CategoryScores(ADVERSARIAL)).agent_a_mean
    token_eff = category_scores.get(TOKEN_EFFICIENCY, CategoryScores(TOKEN_EFFICIENCY)).agent_a_mean
    brand_knowledge = category_scores.get(BRAND_KNOWLEDGE, CategoryScores(BRAND_KNOWLEDGE)).agent_a_mean

    # Build readiness breakdown for transparency
    readiness_breakdown: list[dict] = []

    if discovery_mode:
        # Discovery mode: 10 categories — two-layer structure
        # Layer 1 (Discoverability): Can users find products? (60%)
        # Layer 2 (Brand DNA): Does the LLM deeply understand this brand? (40%)
        _weights_raw = [
            # Layer 1: Discoverability
            (TABLE_STAKES, "Table Stakes", 0.22, factual, "L1"),
            (BLIND_DISCOVERY, "Blind Discovery", 0.12, blind_discovery_score, "L1"),
            (PRODUCT_COMPARISON, "Product Comparison", 0.10, product_comparison_score, "L1"),
            (OCCASION_REASONING, "Occasion Reasoning", 0.08, constraint, "L1"),
            (CROSS_PLATFORM_CONSISTENCY, "Cross-Platform Consistency", 0.08, platform_consistency_a, "L1"),
            (TOKEN_EFFICIENCY, "Token Efficiency", 0.02, token_eff, "L1"),
            # Layer 2: Brand DNA
            (ATTRIBUTE_COMPLETENESS, "Attribute Completeness", 0.18, attr_completeness, "L2"),
            (TEMPORAL_FRESHNESS, "Temporal Freshness", 0.12, temporal, "L2"),
            (ADVERSARIAL, "Adversarial", 0.05, adversarial, "L2"),
            (BRAND_KNOWLEDGE, "Brand Knowledge", 0.03, brand_knowledge, "L2"),
        ]
        # Drop categories that had no queries and renormalize weights
        _weights = [t for t in _weights_raw if t[0] in category_scores]
        if not _weights:
            _weights = _weights_raw  # fallback: keep all if somehow none matched
        raw_total = sum(w for _, _, w, _, _ in _weights)
        if raw_total > 0 and abs(raw_total - 1.0) > 0.001:
            _weights = [(c, l, w / raw_total, s, ly) for c, l, w, s, ly in _weights]
        readiness = sum(w * s for _, _, w, s, _ in _weights)
        for cat, label, weight, score, layer in _weights:
            readiness_breakdown.append({
                "category": cat, "label": label, "weight": weight,
                "score": score, "weighted": weight * score, "layer": layer,
            })
    else:
        # CSV mode: all 11 categories included
        oos = category_scores.get(INVENTORY_OOS, CategoryScores(INVENTORY_OOS)).agent_a_mean
        multi_product = category_scores.get(MULTI_PRODUCT_COMPARISON, CategoryScores(MULTI_PRODUCT_COMPARISON)).agent_a_mean
        _weights_raw = [
            (TABLE_STAKES, "Table Stakes", 0.18, factual),
            (INVENTORY_OOS, "Inventory / OOS", 0.12, oos),
            (MULTI_PRODUCT_COMPARISON, "Multi-Product Comparison", 0.06, multi_product),
            (OCCASION_REASONING, "Occasion Reasoning", 0.14, constraint),
            (BLIND_DISCOVERY, "Blind Discovery", 0.06, blind_discovery_score),
            (CROSS_PLATFORM_CONSISTENCY, "Cross-Platform Consistency", 0.10, platform_consistency_a),
            (TOKEN_EFFICIENCY, "Token Efficiency", 0.04, token_eff),
            (ADVERSARIAL, "Adversarial", 0.10, adversarial),
            (ATTRIBUTE_COMPLETENESS, "Attribute Completeness", 0.08, attr_completeness),
            (TEMPORAL_FRESHNESS, "Temporal Freshness", 0.06, temporal),
            (BRAND_KNOWLEDGE, "Brand Knowledge", 0.06, brand_knowledge),
        ]
        # Drop categories that had no queries and renormalize weights
        _weights = [t for t in _weights_raw if t[0] in category_scores]
        if not _weights:
            _weights = _weights_raw
        raw_total = sum(w for _, _, w, _ in _weights)
        if raw_total > 0 and abs(raw_total - 1.0) > 0.001:
            _weights = [(c, l, w / raw_total, s) for c, l, w, s in _weights]
        readiness = sum(w * s for _, _, w, s in _weights)
        for cat, label, weight, score in _weights:
            readiness_breakdown.append({
                "category": cat, "label": label, "weight": weight,
                "score": score, "weighted": weight * score,
            })
    readiness_pct = readiness * 100

    if readiness_pct < 40:
        readiness_level = "critical"
    elif readiness_pct < 70:
        readiness_level = "at-risk"
    else:
        readiness_level = "competitive"

    # --- Per-Layer Scores (discovery mode) ---
    # Use weighted averages matching the readiness formula, not simple means
    layer_scores: dict[str, float] | None = None
    if discovery_mode:
        layer_weighted_sums: dict[str, float] = {}
        layer_weight_totals: dict[str, float] = {}
        for item in readiness_breakdown:
            layer = item.get("layer", "")
            if layer:
                layer_weighted_sums[layer] = layer_weighted_sums.get(layer, 0.0) + item["weighted"]
                layer_weight_totals[layer] = layer_weight_totals.get(layer, 0.0) + item["weight"]
        layer_scores = {}
        if layer_weight_totals.get("L1", 0) > 0:
            layer_scores[LAYER_1_DISCOVERABILITY] = layer_weighted_sums["L1"] / layer_weight_totals["L1"]
        if layer_weight_totals.get("L2", 0) > 0:
            layer_scores[LAYER_2_BRAND_DNA] = layer_weighted_sums["L2"] / layer_weight_totals["L2"]

    # --- Before/After Comparison Examples (A/B mode only) ---
    comparison_examples: list[dict] = []
    if agent_b_responses:
        for jr in judge_results:
            if not jr.model_key:
                continue
            key = f"{jr.query_id}_{jr.model_key}"
            a_resp = agent_a_responses.get(key)
            b_resp = agent_b_responses.get(key)
            if not a_resp or not b_resp:
                continue
            a_score = _to_float(jr.agent_a_scores.get("factual_accuracy", 0))
            b_score = _to_float(jr.agent_b_scores.get("factual_accuracy", 0))
            delta_ab = b_score - a_score
            if delta_ab > 0.2:  # Agent B significantly outperformed Agent A
                comparison_examples.append({
                    "query_id": jr.query_id,
                    "model_key": jr.model_key,
                    "prompt": jr.prompt,
                    "category": jr.category,
                    "agent_a_snippet": a_resp.text,
                    "agent_b_snippet": b_resp.text,
                    "agent_a_score": a_score,
                    "agent_b_score": b_score,
                    "delta": delta_ab,
                    "failure_type": jr.failure_type,
                })
        # Keep top 5, sorted by biggest delta
        comparison_examples.sort(key=lambda x: x["delta"], reverse=True)
        comparison_examples = comparison_examples[:5]

    # --- Product-Level Vulnerability (CSV mode only) ---
    product_vulnerabilities: list[dict] = []
    if queries and not discovery_mode:
        query_map = {q.query_id: q for q in queries}
        # Aggregate scores by product
        product_scores: dict[str, list[float]] = {}
        product_failures: dict[str, list[str]] = {}
        product_categories: dict[str, dict[str, list[float]]] = {}
        product_names: dict[str, str] = {}

        for jr in judge_results:
            q = query_map.get(jr.query_id)
            if not q or not q.target_products:
                continue
            a_score = _to_float(jr.agent_a_scores.get("factual_accuracy", 0))
            for pid in q.target_products:
                if pid == "none":
                    continue
                product_scores.setdefault(pid, []).append(a_score)
                product_categories.setdefault(pid, {}).setdefault(q.category, []).append(a_score)
                if jr.failure_type and jr.failure_type.lower() != "none":
                    product_failures.setdefault(pid, []).append(jr.failure_type)
                # Try to get product name from query
                if pid not in product_names:
                    for pq in queries:
                        if pid in pq.target_products and pq.ground_truth_keys:
                            product_names[pid] = pq.prompt.split("the ")[-1].split(" at ")[0].split(" from ")[0][:50]
                            break

        for pid, scores_list in product_scores.items():
            avg = float(np.mean(scores_list)) if scores_list else 0.0
            # Find worst category
            worst_cat = ""
            worst_cat_score = 1.0
            for cat, cat_scores_list in product_categories.get(pid, {}).items():
                cat_avg = float(np.mean(cat_scores_list)) if cat_scores_list else 0.0
                if cat_avg < worst_cat_score:
                    worst_cat_score = cat_avg
                    worst_cat = cat
            failures = product_failures.get(pid, [])
            product_vulnerabilities.append({
                "product_id": pid,
                "product_name": product_names.get(pid, pid),
                "avg_score": avg,
                "num_queries": len(scores_list),
                "worst_category": worst_cat,
                "worst_category_score": worst_cat_score,
                "failure_count": len(failures),
                "top_failure_types": list(set(failures))[:3],
            })
        # Sort by worst average score (most vulnerable first)
        product_vulnerabilities.sort(key=lambda x: x["avg_score"])
        product_vulnerabilities = product_vulnerabilities[:10]

    # --- Auto-generated Recommendations ---
    recommendations: list[dict] = []
    # Based on weakest categories
    weak_cats = sorted(
        category_scores.items(),
        key=lambda x: x[1].agent_a_mean,
    )
    cat_recommendations = {
        TABLE_STAKES: "Improve core product data accuracy",
        INVENTORY_OOS: "Fix real-time inventory visibility",
        OCCASION_REASONING: "Add occasion/use-case metadata to products",
        BLIND_DISCOVERY: "Improve product discoverability for AI agents",
        PRODUCT_COMPARISON: "Improve product comparison data quality",
        ADVERSARIAL: "Reduce hallucination exposure",
        ATTRIBUTE_COMPLETENESS: "Fill gaps in product attribute coverage",
        CROSS_PLATFORM_CONSISTENCY: "Standardize data across AI platforms",
        TOKEN_EFFICIENCY: "Optimize data delivery for AI agents",
        BRAND_KNOWLEDGE: "Deepen brand-specific AI knowledge",
    }
    for cat, cat_score in weak_cats[:3]:
        if cat_score.agent_a_mean < 0.7 and cat in cat_recommendations:
            # Use actual failure narrative from this audit instead of static text
            dynamic_desc = category_failure_summaries.get(cat, "")
            if not dynamic_desc:
                # Fallback: summarize from failure types
                ftypes = _cat_failure_types.get(cat, [])
                if ftypes:
                    from collections import Counter
                    top = Counter(ftypes).most_common(2)
                    dynamic_desc = "Main issues: " + ", ".join(
                        f"{ft} ({n}x)" for ft, n in top
                    )
            recommendations.append({
                "priority": len(recommendations) + 1,
                "title": cat_recommendations[cat],
                "description": dynamic_desc,
                "category": cat,
                "current_score": round(cat_score.agent_a_mean * 100, 1),
            })

    # --- Per-category sample sizes ---
    sample_sizes: dict[str, dict[str, int]] = {}
    for jr in judge_results:
        if not jr.model_key or _is_api_error(jr):
            continue
        sample_sizes.setdefault(jr.category, {})
        sample_sizes[jr.category][jr.model_key] = sample_sizes[jr.category].get(jr.model_key, 0) + 1

    return AuditScores(
        category_scores=category_scores,
        token_stats=token_stats,
        agentic_readiness_score=readiness_pct,
        readiness_level=readiness_level,
        top_failures=top_failures,
        hallucination_details=hallucination_details,
        platform_consistency_a=platform_consistency_a,
        platform_consistency_b=platform_consistency_b,
        consistency_details=consistency_detail,
        per_model_scores=per_model_scores or None,
        per_model_scores_b=per_model_scores_b or None,
        per_model_failures=per_model_failures or None,
        per_model_missing=per_model_missing or None,
        per_model_token_stats=per_model_token_stats or None,
        per_model_query_count=per_model_query_count or None,
        tokens_per_correct_fact_a=tokens_per_correct_fact_a,
        tokens_per_correct_fact_b=tokens_per_correct_fact_b,
        per_model_attribute_coverage=per_model_attribute_coverage or None,
        comparison_examples=comparison_examples or None,
        product_vulnerabilities=product_vulnerabilities or None,
        recommendations=recommendations or None,
        per_model_failure_examples=per_model_failure_examples or None,
        consistency_examples=consistency_examples or None,
        per_model_catalog_coverage=per_model_catalog_coverage or None,
        layer_scores=layer_scores,
        readiness_breakdown=readiness_breakdown or None,
        sample_sizes=sample_sizes or None,
        agentic_actionability=agentic_actionability or None,
        agentic_actionability_overall=agentic_actionability_overall,
        agentic_actionability_failures=actionability_fail_examples or None,
        cross_model_conflict_matrix=cross_model_conflict_data or None,
        fragmentation_summary=fragmentation_summary if total_comparisons > 0 else None,
        category_failure_summaries=category_failure_summaries or None,
        layer_failure_summaries=layer_failure_summaries or None,
        consistency_summary=consistency_summary,
        per_model_source_citation_rate=per_model_source_citation_rate or None,
    )
