"""Reliability scoring for discovery mode — measures confidence without ground truth.

Four dimensions:
1. Self-consistency: Same question rephrased → same answer? (probe-based)
2. Cross-model convergence: Multiple models → same answer? (pairwise comparison)
3. Source grounding: Does the model cite verifiable sources? (regex extraction)
4. Specificity: Concrete details vs vague hedging? (from judge results)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from src.scorer import (
    _extract_prices,
    _extract_product_names,
    _fuzzy_fact_match,
)
from src.judge import JudgeResult
from src.agents.base_agent import AgentResponse
from src.query_generator import Query


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Result of comparing a probe response to its parent response."""

    parent_query_id: str
    probe_query_id: str
    model_key: str
    category: str
    facts_match: bool
    price_overlap: float
    product_overlap: float
    consistency_score: float


@dataclass
class ReliabilityDimension:
    """Score for a single reliability dimension."""

    score: float = 0.0
    n_samples: int = 0


@dataclass
class ModelReliability:
    """Per-model reliability breakdown."""

    model_key: str = ""
    self_consistency: ReliabilityDimension = field(default_factory=ReliabilityDimension)
    cross_model_convergence: ReliabilityDimension = field(default_factory=ReliabilityDimension)
    source_grounding: ReliabilityDimension = field(default_factory=ReliabilityDimension)
    specificity: ReliabilityDimension = field(default_factory=ReliabilityDimension)
    composite_score: float = 0.0


@dataclass
class CategoryReliability:
    """Per-category reliability breakdown (averaged across models)."""

    category: str = ""
    self_consistency: float = 0.0
    cross_model_convergence: float = 0.0
    source_grounding: float = 0.0
    specificity: float = 0.0
    composite_score: float = 0.0
    n_models: int = 0


@dataclass
class ReliabilityScores:
    """Top-level container for all reliability data."""

    overall_score: float = 0.0
    per_model: dict[str, ModelReliability] = field(default_factory=dict)
    per_category: dict[str, CategoryReliability] = field(default_factory=dict)
    probe_results: list[ProbeResult] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    probes_enabled: bool = True


# ---------------------------------------------------------------------------
# Default weights
# ---------------------------------------------------------------------------

RELIABILITY_WEIGHTS: dict[str, float] = {
    "self_consistency": 0.40,
    "cross_model_convergence": 0.30,
    "specificity": 0.15,
    "source_grounding": 0.15,
}


# ---------------------------------------------------------------------------
# Dimension 1: Source Grounding
# ---------------------------------------------------------------------------

_URL_PATTERN = re.compile(r'https?://[^\s\)\]>"\']+', re.IGNORECASE)

_SOURCE_INDICATORS = re.compile(
    r'(?i)(?:on\s+(?:the\s+)?(?:official\s+)?(?:brand\s+)?website|'
    r'from\s+(?:the\s+)?(?:official\s+)?(?:online\s+)?store|'
    r'product\s+page|listed\s+(?:on|at))',
)


def _source_grounding_score(response_text: str, brand_url: str) -> float:
    """Score how well-grounded a response is by checking for source URLs.

    Returns:
        1.0 — URL matching brand domain found
        0.7 — Any URL found (not brand-specific)
        0.3 — Indirect source reference ("on the website")
        0.0 — No grounding at all
    """
    urls = _URL_PATTERN.findall(response_text)
    if urls:
        brand_domain = urlparse(brand_url).netloc.replace("www.", "") if brand_url else ""
        if brand_domain:
            has_brand_url = any(brand_domain in url for url in urls)
            return 1.0 if has_brand_url else 0.7
        return 0.7

    if _SOURCE_INDICATORS.search(response_text):
        return 0.3
    return 0.0


# ---------------------------------------------------------------------------
# Dimension 2: Specificity (from judge results)
# ---------------------------------------------------------------------------

def _specificity_score(jr: JudgeResult, response_text: str) -> float:
    """Score response specificity from judge results.

    Combines:
    - Judge's factual_accuracy (penalizes vagueness): weight 0.4
    - Attribute coverage specificity rate (if available): weight 0.3
    - Inverse hedge detection (hedged = 0.0): weight 0.3
    """
    scores_dict = jr.agent_a_scores

    # Component 1: Factual accuracy as specificity proxy
    factual = float(scores_dict.get("factual_accuracy", 0) or 0)

    # Component 2: Attribute specificity from coverage data
    attr_cov = scores_dict.get("attribute_coverage", {})
    if attr_cov and isinstance(attr_cov, dict):
        specific_count = sum(
            1 for v in attr_cov.values()
            if isinstance(v, dict) and v.get("specific", False)
        )
        total_attrs = max(len(attr_cov), 1)
        attr_specificity = specific_count / total_attrs
    else:
        attr_specificity = factual  # fallback

    # Component 3: Hedge inverse
    hedge_detected = scores_dict.get("hedge_detected", False)
    hedge_inverse = 0.0 if hedge_detected else 1.0

    return 0.4 * factual + 0.3 * attr_specificity + 0.3 * hedge_inverse


# ---------------------------------------------------------------------------
# Dimension 3: Cross-Model Convergence
# ---------------------------------------------------------------------------

def _convergence_for_model(
    consistency_responses: dict[str, dict[str, AgentResponse]],
    queries: list[Query],
    model_key: str,
) -> ReliabilityDimension:
    """Compute cross-model convergence for a single model.

    For each query this model answered, check what fraction of OTHER models
    agree (using _fuzzy_fact_match). High agreement = high reliability.
    """
    query_map = {q.query_id: q for q in queries}
    agreements: list[float] = []

    for qid, model_resps in consistency_responses.items():
        if model_key not in model_resps:
            continue
        this_resp = model_resps[model_key]
        if this_resp.text.startswith("ERROR"):
            continue

        q = query_map.get(qid)
        gt_keys = q.ground_truth_keys if q else {}

        other_agree = 0
        other_total = 0
        for other_mk, other_resp in model_resps.items():
            if other_mk == model_key or other_resp.text.startswith("ERROR"):
                continue
            other_total += 1
            if _fuzzy_fact_match(this_resp.text, other_resp.text, gt_keys):
                other_agree += 1

        if other_total > 0:
            agreements.append(other_agree / other_total)

    avg = sum(agreements) / len(agreements) if agreements else 0.0
    return ReliabilityDimension(score=round(avg, 3), n_samples=len(agreements))


# ---------------------------------------------------------------------------
# Dimension 4: Self-Consistency (probe comparison)
# ---------------------------------------------------------------------------

def _compare_probe_to_parent(
    parent_text: str,
    probe_text: str,
    parent_query: Query,
) -> ProbeResult:
    """Compare a probe response to its parent using fuzzy fact matching.

    Uses the same _fuzzy_fact_match logic from scorer.py — no LLM judge needed.
    Also computes Jaccard overlap on extracted prices and product names for a
    continuous similarity signal.
    """
    gt_keys = parent_query.ground_truth_keys
    facts_match = _fuzzy_fact_match(parent_text, probe_text, gt_keys)

    prices_parent = _extract_prices(parent_text)
    prices_probe = _extract_prices(probe_text)
    if prices_parent or prices_probe:
        price_union = prices_parent | prices_probe
        price_overlap = len(prices_parent & prices_probe) / len(price_union) if price_union else 0.0
    else:
        price_overlap = 0.5  # neutral when neither mentions prices

    names_parent = _extract_product_names(parent_text)
    names_probe = _extract_product_names(probe_text)
    if names_parent or names_probe:
        name_union = names_parent | names_probe
        product_overlap = len(names_parent & names_probe) / len(name_union) if name_union else 0.0
    else:
        product_overlap = 0.5  # neutral

    consistency_score = 1.0 if facts_match else (0.5 * price_overlap + 0.5 * product_overlap)

    return ProbeResult(
        parent_query_id=parent_query.query_id,
        probe_query_id=f"{parent_query.query_id}_P1",
        model_key="",  # filled by caller
        category=parent_query.category,
        facts_match=facts_match,
        price_overlap=round(price_overlap, 3),
        product_overlap=round(product_overlap, 3),
        consistency_score=round(consistency_score, 3),
    )


# ---------------------------------------------------------------------------
# Main: compute_reliability()
# ---------------------------------------------------------------------------

def compute_reliability(
    judge_results: list[JudgeResult],
    agent_a_responses: dict[str, AgentResponse],
    consistency_responses: dict[str, dict[str, AgentResponse]] | None,
    queries: list[Query],
    probe_queries: list[Query] | None,
    probe_responses: dict[str, AgentResponse] | None,
    brand_url: str = "",
    probes_enabled: bool = True,
) -> ReliabilityScores:
    """Compute all reliability dimensions for discovery mode.

    Returns a ReliabilityScores container with per-model, per-category, and
    overall reliability scores.
    """
    # --- Determine active weights ---
    weights = dict(RELIABILITY_WEIGHTS)
    if not probes_enabled or not probe_queries:
        # Redistribute self_consistency weight proportionally to other dimensions
        sc_weight = weights.pop("self_consistency")
        remaining_total = sum(weights.values())
        if remaining_total > 0:
            weights = {k: v + (v / remaining_total) * sc_weight for k, v in weights.items()}
        probes_enabled = False

    query_map = {q.query_id: q for q in queries}
    model_keys: set[str] = set()
    for jr in judge_results:
        if jr.model_key:
            model_keys.add(jr.model_key)

    # --- Per-model computation ---
    per_model: dict[str, ModelReliability] = {}
    all_probe_results: list[ProbeResult] = []

    for mk in sorted(model_keys):
        mk_jrs = [jr for jr in judge_results if jr.model_key == mk]

        # 1. Source grounding
        sg_scores: list[float] = []
        for jr in mk_jrs:
            key = f"{jr.query_id}_{mk}"
            resp = agent_a_responses.get(key)
            if resp and not resp.text.startswith("ERROR"):
                sg_scores.append(_source_grounding_score(resp.text, brand_url))
        sg_dim = ReliabilityDimension(
            score=round(sum(sg_scores) / len(sg_scores), 3) if sg_scores else 0.0,
            n_samples=len(sg_scores),
        )

        # 2. Specificity
        spec_scores: list[float] = []
        for jr in mk_jrs:
            spec_scores.append(_specificity_score(jr, ""))
        spec_dim = ReliabilityDimension(
            score=round(sum(spec_scores) / len(spec_scores), 3) if spec_scores else 0.0,
            n_samples=len(spec_scores),
        )

        # 3. Cross-model convergence
        if consistency_responses and len(model_keys) >= 2:
            conv_dim = _convergence_for_model(consistency_responses, queries, mk)
        else:
            conv_dim = ReliabilityDimension(score=0.5, n_samples=0)

        # 4. Self-consistency (probes)
        sc_dim = ReliabilityDimension(score=0.0, n_samples=0)
        if probes_enabled and probe_queries and probe_responses:
            probe_scores: list[float] = []
            for pq in probe_queries:
                probe_key = f"{pq.query_id}_{mk}"
                parent_key = f"{pq.probe_parent_id}_{mk}"
                probe_resp = probe_responses.get(probe_key)
                parent_resp = agent_a_responses.get(parent_key)
                parent_q = query_map.get(pq.probe_parent_id)
                if probe_resp and parent_resp and parent_q:
                    if not probe_resp.text.startswith("ERROR") and not parent_resp.text.startswith("ERROR"):
                        pr = _compare_probe_to_parent(parent_resp.text, probe_resp.text, parent_q)
                        pr.model_key = mk
                        all_probe_results.append(pr)
                        probe_scores.append(pr.consistency_score)
            sc_dim = ReliabilityDimension(
                score=round(sum(probe_scores) / len(probe_scores), 3) if probe_scores else 0.0,
                n_samples=len(probe_scores),
            )

        # Composite
        composite = 0.0
        if "self_consistency" in weights:
            composite += weights["self_consistency"] * sc_dim.score
        composite += weights.get("cross_model_convergence", 0) * conv_dim.score
        composite += weights.get("specificity", 0) * spec_dim.score
        composite += weights.get("source_grounding", 0) * sg_dim.score

        per_model[mk] = ModelReliability(
            model_key=mk,
            self_consistency=sc_dim,
            cross_model_convergence=conv_dim,
            source_grounding=sg_dim,
            specificity=spec_dim,
            composite_score=round(composite, 3),
        )

    # --- Per-category computation ---
    per_category: dict[str, CategoryReliability] = {}
    by_cat: dict[str, list[JudgeResult]] = {}
    for jr in judge_results:
        if jr.model_key:
            by_cat.setdefault(jr.category, []).append(jr)

    for cat, cat_jrs in by_cat.items():
        cat_models = set(jr.model_key for jr in cat_jrs if jr.model_key)
        if not cat_models:
            continue

        # Per-category self-consistency from probe results
        cat_probe_scores = [
            pr.consistency_score for pr in all_probe_results
            if pr.category == cat
        ]
        sc_avg = sum(cat_probe_scores) / len(cat_probe_scores) if cat_probe_scores else 0.0

        # Per-category source grounding
        cat_sg: list[float] = []
        for jr in cat_jrs:
            key = f"{jr.query_id}_{jr.model_key}"
            resp = agent_a_responses.get(key)
            if resp and not resp.text.startswith("ERROR"):
                cat_sg.append(_source_grounding_score(resp.text, brand_url))
        sg_avg = sum(cat_sg) / len(cat_sg) if cat_sg else 0.0

        # Per-category specificity
        cat_spec = [_specificity_score(jr, "") for jr in cat_jrs]
        spec_avg = sum(cat_spec) / len(cat_spec) if cat_spec else 0.0

        # Per-category convergence (average model convergence for this category)
        cat_conv: list[float] = []
        for mk in cat_models:
            mr = per_model.get(mk)
            if mr and mr.cross_model_convergence.n_samples > 0:
                cat_conv.append(mr.cross_model_convergence.score)
        conv_avg = sum(cat_conv) / len(cat_conv) if cat_conv else 0.0

        composite = 0.0
        if "self_consistency" in weights:
            composite += weights["self_consistency"] * sc_avg
        composite += weights.get("cross_model_convergence", 0) * conv_avg
        composite += weights.get("specificity", 0) * spec_avg
        composite += weights.get("source_grounding", 0) * sg_avg

        per_category[cat] = CategoryReliability(
            category=cat,
            self_consistency=round(sc_avg, 3),
            cross_model_convergence=round(conv_avg, 3),
            source_grounding=round(sg_avg, 3),
            specificity=round(spec_avg, 3),
            composite_score=round(composite, 3),
            n_models=len(cat_models),
        )

    # --- Overall ---
    all_composites = [mr.composite_score for mr in per_model.values()]
    overall = sum(all_composites) / len(all_composites) if all_composites else 0.0

    return ReliabilityScores(
        overall_score=round(overall, 3),
        per_model=per_model,
        per_category=per_category,
        probe_results=all_probe_results,
        weights=weights,
        probes_enabled=probes_enabled,
    )
