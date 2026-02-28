"""Generates the full test matrix from ground truth CSV products."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.brand_researcher import BrandIntelligence

from src.data_loader import Product

# Category constants
TABLE_STAKES = "table_stakes"
INVENTORY_OOS = "inventory_oos"
MULTI_PRODUCT_COMPARISON = "multi_product_comparison"
OCCASION_REASONING = "occasion_reasoning"
BLIND_DISCOVERY = "blind_discovery"
PRODUCT_COMPARISON = "product_comparison"
CROSS_PLATFORM_CONSISTENCY = "cross_platform_consistency"
TOKEN_EFFICIENCY = "token_efficiency"
ADVERSARIAL = "adversarial"
ATTRIBUTE_COMPLETENESS = "attribute_completeness"
TEMPORAL_FRESHNESS = "temporal_freshness"
BRAND_KNOWLEDGE = "brand_knowledge"

ALL_CATEGORIES = [
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
]

DISCOVERY_CATEGORIES = [
    TABLE_STAKES,
    OCCASION_REASONING,
    BLIND_DISCOVERY,
    PRODUCT_COMPARISON,
    CROSS_PLATFORM_CONSISTENCY,
    TOKEN_EFFICIENCY,
    ADVERSARIAL,
    ATTRIBUTE_COMPLETENESS,
    TEMPORAL_FRESHNESS,
    BRAND_KNOWLEDGE,
]

# Two-layer classification for discovery mode
LAYER_1_DISCOVERABILITY = "discoverability"
LAYER_2_BRAND_DNA = "brand_dna"

LAYER_MAPPING = {
    TABLE_STAKES: LAYER_1_DISCOVERABILITY,
    OCCASION_REASONING: LAYER_1_DISCOVERABILITY,
    BLIND_DISCOVERY: LAYER_1_DISCOVERABILITY,
    PRODUCT_COMPARISON: LAYER_1_DISCOVERABILITY,
    CROSS_PLATFORM_CONSISTENCY: LAYER_1_DISCOVERABILITY,
    TOKEN_EFFICIENCY: LAYER_1_DISCOVERABILITY,
    BRAND_KNOWLEDGE: LAYER_2_BRAND_DNA,
    ADVERSARIAL: LAYER_2_BRAND_DNA,
    ATTRIBUTE_COMPLETENESS: LAYER_2_BRAND_DNA,
    TEMPORAL_FRESHNESS: LAYER_2_BRAND_DNA,
}

LAYER_LABELS = {
    LAYER_1_DISCOVERABILITY: "Layer 1: Discoverability",
    LAYER_2_BRAND_DNA: "Layer 2: Brand DNA",
}

# The 6 core product attributes every AI agent should surface
REQUIRED_ATTRIBUTES = ["price", "material", "sizes", "colors", "availability", "care"]

PERSONAS = [
    ("Business Traveler", "I travel 3 weeks a month for client meetings"),
    ("Remote Worker", "I work from coffee shops and occasionally visit HQ"),
    ("Weekend Explorer", "I hike on weekends but go straight to brunch after"),
    ("Student", "I cycle to campus in all weather"),
    ("Urban Commuter", "I take the subway and walk 10 mins to the office"),
]

# Negative constraints — at least one must appear in every multi-constraint query
NEGATIVE_CONSTRAINTS = [
    "does not feature the all-over logo print",
    "has no visible branding",
    "is not made of leather",
    "does not require dry cleaning",
    "is not in black",
    "has no zipper closures",
    "is not from a collaboration or limited-edition line",
    "is not discontinued or past-season",
]

# Temporal/live-state triggers — force real-time data retrieval
LIVE_STATE_TRIGGERS = [
    "currently in stock today",
    "available to ship by this Friday",
    "that I can buy right now on the website",
    "currently available in my size",
    "in the colors available as of today",
    "with current pricing (not sale or promotional)",
]


@dataclass
class Query:
    """A single test query to be sent to both agents."""
    query_id: str
    category: str
    prompt: str
    expected: str
    ground_truth_keys: dict[str, Any]
    target_products: list[str]
    rank_order: list[str] = field(default_factory=list)
    persona: str = ""
    probe_parent_id: str = ""  # Non-empty = this is a consistency probe


def _shuffle_copy(items: list, rng: random.Random) -> list:
    """Return a shuffled copy of items using the given RNG."""
    copy = list(items)
    rng.shuffle(copy)
    return copy


def generate_query_matrix(
    products: list[Product],
    brand_name: str,
    queries_per_category: int = 10,
    seed: int | None = None,
) -> list[Query]:
    """Generate the full test matrix across all 8 categories.

    Each run shuffles templates, products, and constraints so different runs
    produce different query combinations. Pass a fixed seed for reproducibility.
    """
    rng = random.Random(seed)
    queries: list[Query] = []
    counter = 0

    def _next_id() -> str:
        nonlocal counter
        counter += 1
        return f"TS_{counter:03d}"

    def _pick(items: list, index: int):
        """Pick from a list by index. Lists are pre-shuffled by the seed."""
        return items[index % len(items)]

    # Shuffle shared lists so different seeds produce different combinations
    products = _shuffle_copy(products, rng)
    neg_pool = _shuffle_copy(NEGATIVE_CONSTRAINTS, rng)
    live_pool = _shuffle_copy(LIVE_STATE_TRIGGERS, rng)

    # --- 1. TABLE STAKES (Multi-Constraint) ---
    # Each query combines [Budget/Price] + [Feature] + [Negative Constraint] + [Live-State Trigger]
    ts_templates = [
        (
            "I'm looking at the {product_name} from {brand_name}. Is it currently priced "
            "under ${budget_ceil}? I need it to be {live_trigger} and it "
            "{neg_constraint}. Give me the exact price.",
            lambda p: f"{p.price} {p.currency}",
            lambda p: {"price": p.price, "currency": p.currency},
        ),
        (
            "What is the {product_name} from {brand_name} made from? I need something "
            "that {neg_constraint}. Confirm the material and whether it's {live_trigger}.",
            lambda p: p.material,
            lambda p: {"material": p.material},
        ),
        (
            "I want to buy the {product_name} from {brand_name} for under ${budget_ceil}, "
            "but only if it {neg_constraint}. What are the care instructions, and is it "
            "{live_trigger}?",
            lambda p: p.care_instructions,
            lambda p: {"care_instructions": p.care_instructions, "price": p.price},
        ),
        (
            "I'm considering a return on a {product_name} from {brand_name} that cost "
            "under ${budget_ceil}. What is the exact return policy? I need a product "
            "that {neg_constraint}.",
            lambda p: p.return_policy,
            lambda p: {"return_policy": p.return_policy, "price": p.price},
        ),
        (
            "I need the {product_name} from {brand_name} delivered this week — how long "
            "does standard shipping take? The product must be {live_trigger} and it "
            "{neg_constraint}.",
            lambda p: f"{p.shipping_standard_days} days",
            lambda p: {"shipping_standard_days": p.shipping_standard_days},
        ),
        (
            "How fast can I get the {product_name} from {brand_name} via express shipping? "
            "I need it under ${budget_ceil} and it {neg_constraint}.",
            lambda p: f"{p.shipping_express_days} days",
            lambda p: {"shipping_express_days": p.shipping_express_days, "price": p.price},
        ),
        (
            "What hardware color does the {product_name} have? I need one "
            "under ${budget_ceil} that {neg_constraint} and is {live_trigger}.",
            lambda p: p.hardware_color if p.hardware_color else "Not specified",
            lambda p: {"hardware_color": p.hardware_color, "price": p.price},
        ),
        (
            "Does the {product_name} from {brand_name} run large or small? It "
            "{neg_constraint}. What's the fit like for someone 5'10\" and 170 lbs? "
            "Is it {live_trigger}?",
            lambda p: f"{p.fit_type or 'Not specified'}. {p.sizing_notes or ''}".strip(),
            lambda p: {"fit_type": p.fit_type, "sizing_notes": p.sizing_notes},
        ),
        (
            "What are the exact dimensions of the {product_name} from {brand_name}? "
            "I need it to fit a 13-inch laptop and be under ${budget_ceil}. It "
            "{neg_constraint}. Is it {live_trigger}?",
            lambda p: p.dimensions_cm if p.dimensions_cm else "Not specified",
            lambda p: {"dimensions_cm": p.dimensions_cm, "laptop_compatibility": p.laptop_compatibility,
                       "price": p.price},
        ),
        (
            "Will the {product_name} from {brand_name} fit a 16-inch laptop? "
            "I want one under ${budget_ceil} that {neg_constraint} and is {live_trigger}.",
            lambda p: f"Fits up to {p.laptop_compatibility}" if p.laptop_compatibility else "Not applicable",
            lambda p: {"laptop_compatibility": p.laptop_compatibility, "price": p.price},
        ),
        (
            "I'm 5'10\" and 170 lbs. What size {product_name} should I get from "
            "{brand_name}? It must be under ${budget_ceil} and {live_trigger}. It "
            "{neg_constraint}.",
            lambda p: f"Size {p.model_size_worn}" if p.model_size_worn else "No fit data available",
            lambda p: {"model_size_worn": p.model_size_worn, "model_height": p.model_height,
                       "model_weight": p.model_weight, "price": p.price},
        ),
    ]

    for i in range(queries_per_category):
        p = _pick(products, i)
        tmpl_prompt, tmpl_expected, tmpl_keys = _pick(ts_templates, i)
        neg = _pick(neg_pool, i)
        live = _pick(live_pool, i)
        budget_ceil = int(p.price * rng.uniform(1.05, 1.3))
        queries.append(Query(
            query_id=_next_id(),
            category=TABLE_STAKES,
            prompt=tmpl_prompt.format(
                product_name=p.product_name, brand_name=brand_name,
                neg_constraint=neg, live_trigger=live, budget_ceil=budget_ceil,
            ),
            expected=tmpl_expected(p),
            ground_truth_keys={
                **tmpl_keys(p),
                "constraints": [f"budget_under_{budget_ceil}", neg, live],
            },
            target_products=[p.product_id],
        ))

    # --- 2. INVENTORY / OOS AWARENESS ---
    for i in range(queries_per_category):
        p = _pick(products, i)
        all_variants = p.stock
        if not all_variants:
            continue

        # Alternate between OOS and in-stock variants (50/50)
        if i % 2 == 0:
            # Target an OOS variant
            oos = p.oos_variants
            if oos:
                v = _pick(oos, i)
                alts = [sv for sv in p.in_stock_variants if sv.color != v.color or sv.size != v.size]
                alt_text = f" Available alternatives: {', '.join(f'{a.size}/{a.color}' for a in alts[:3])}" if alts else ""
                queries.append(Query(
                    query_id=_next_id(),
                    category=INVENTORY_OOS,
                    prompt=f"I want the {p.product_name} in {v.color}, size {v.size}. Is it in stock right now at {brand_name}?",
                    expected=f"The {p.product_name} in {v.color}, size {v.size} is currently out of stock.{alt_text}",
                    ground_truth_keys={"in_stock": False, "size": v.size, "color": v.color},
                    target_products=[p.product_id],
                ))
                continue
        # Target an in-stock variant
        ins = p.in_stock_variants
        if ins:
            v = _pick(ins, i)
            queries.append(Query(
                query_id=_next_id(),
                category=INVENTORY_OOS,
                prompt=f"Can I order the {p.product_name} in {v.color}, size {v.size} from {brand_name}?",
                expected=f"Yes, the {p.product_name} in {v.color}, size {v.size} is in stock.",
                ground_truth_keys={"in_stock": True, "size": v.size, "color": v.color},
                target_products=[p.product_id],
            ))

    # --- 3. MULTI-PRODUCT COMPARISON & RANKING ---
    comparison_templates = [
        (
            "Which is the lightest?",
            "weight_grams",
            lambda ps: sorted(ps, key=lambda p: p.weight_grams if p.weight_grams else 9999),
        ),
        (
            "Which is most waterproof?",
            "waterproof_score",
            lambda ps: sorted(ps, key=lambda p: p.waterproof_score, reverse=True),
        ),
        (
            "Which is warmest?",
            "warmth_score",
            lambda ps: sorted(ps, key=lambda p: p.warmth_score, reverse=True),
        ),
        (
            "Which is most office-appropriate?",
            "office_appropriate_score",
            lambda ps: sorted(ps, key=lambda p: p.office_appropriate_score, reverse=True),
        ),
    ]

    for i in range(queries_per_category):
        if len(products) < 2:
            break
        # Pick 2-3 products for comparison
        n_compare = min(3, len(products))
        compare_products = rng.sample(products, n_compare)
        tmpl_q, attr, rank_fn = _pick(comparison_templates, i)
        ranked = rank_fn(compare_products)
        names = ", ".join(p.product_name for p in compare_products)

        queries.append(Query(
            query_id=_next_id(),
            category=MULTI_PRODUCT_COMPARISON,
            prompt=f"Compare {names} from {brand_name}. {tmpl_q}",
            expected=f"Ranked: {', '.join(p.product_name for p in ranked)}",
            ground_truth_keys={"attribute": attr, "ranking": [p.product_id for p in ranked]},
            target_products=[p.product_id for p in compare_products],
            rank_order=[p.product_id for p in ranked],
        ))

    # --- 4. OCCASION / AGENTIC REASONING ---
    occasion_contexts = [
        ("a rainy commute that works for a client dinner", ["commute", "corporate"]),
        ("a weekend hike followed by brunch in the city", ["weekend", "casual"]),
        ("cycling to campus in cold rain", ["cycling", "campus"]),
        ("a business trip where I need to look professional", ["corporate", "travel"]),
        ("walking 10 minutes from the subway to the office in winter", ["commute", "extreme-cold"]),
    ]

    for i in range(queries_per_category):
        persona_name, persona_desc = _pick(PERSONAS, i)
        context, tags = _pick(occasion_contexts, i)
        max_price = max(p.price for p in products)
        budget = int(max_price * rng.uniform(0.6, 1.2))
        neg = _pick(neg_pool, i)
        live = _pick(live_pool, i)

        # Find products matching constraints
        matching = [
            p for p in products
            if p.price <= budget and any(t in p.occasion_tags for t in tags)
        ]
        expected_ids = [p.product_id for p in matching] if matching else ["none"]

        queries.append(Query(
            query_id=_next_id(),
            category=OCCASION_REASONING,
            prompt=(
                f"I'm a {persona_name.lower()} — {persona_desc}. "
                f"I need something for {context}. "
                f"Budget ${budget}. It {neg} and is {live}. "
                f"What does {brand_name} have?"
            ),
            expected=f"Recommended: {', '.join(p.product_name for p in matching)}" if matching else "No matching products within budget.",
            ground_truth_keys={
                "budget": budget,
                "required_tags": tags,
                "matching_products": expected_ids,
                "constraints": [f"budget_{budget}", neg, live],
            },
            target_products=expected_ids,
            persona=f"{persona_name}: {persona_desc}",
        ))

    # --- 5. BLIND UNBRANDED DISCOVERY (Multi-Constraint) ---
    blind_templates = [
        "Recommend a minimalist rain shell under ${budget} that fits a 13-inch laptop, "
        "{neg_constraint}, and is {live_trigger}.",
        "I need a tote for daily office use under ${budget} that {neg_constraint}. "
        "It needs to be {live_trigger}. What are my exact options with prices?",
        "Best lightweight packable jacket for business travel under ${budget}? "
        "It {neg_constraint} and is {live_trigger}.",
        "A commuter bag for someone who cycles to work, under ${budget}, that "
        "{neg_constraint}. It needs to be {live_trigger}.",
        "A jacket I can wear to a client dinner straight from the office, under ${budget}. "
        "It {neg_constraint} and is {live_trigger}.",
        "What's a waterproof jacket that looks professional, under ${budget}, "
        "{neg_constraint}? I need one that's {live_trigger}.",
        "I need a bag that fits a 16-inch laptop and looks good in an office, "
        "under ${budget}. It {neg_constraint} and is {live_trigger}.",
        "Recommend a warm but slim-fitting winter jacket under ${budget} that "
        "{neg_constraint} and is {live_trigger}.",
    ]

    for i in range(queries_per_category):
        tmpl = _pick(blind_templates, i)
        max_price = max(p.price for p in products)
        budget = int(max_price * rng.uniform(0.8, 1.5))
        neg = _pick(neg_pool, i)
        live = _pick(live_pool, i)
        prompt = tmpl.replace("${budget}", str(budget)).format(
            neg_constraint=neg, live_trigger=live,
        )

        # Determine which products best match (for scoring)
        rank = sorted(products, key=lambda p: p.office_appropriate_score + p.waterproof_score, reverse=True)

        queries.append(Query(
            query_id=_next_id(),
            category=BLIND_DISCOVERY,
            prompt=prompt,
            expected=f"Top match: {rank[0].product_name}",
            ground_truth_keys={
                "brand_products": [p.product_id for p in products],
                "constraints": [f"budget_{budget}", neg, live],
            },
            target_products=[p.product_id for p in products],
            rank_order=[p.product_id for p in rank],
        ))

    # --- 6. CROSS-PLATFORM CONSISTENCY (Multi-Constraint) ---
    consistency_templates = [
        (
            "What is the exact current price of the {product_name} at {brand_name} "
            "as of today? Is it {live_trigger} in all colors?",
            lambda p: {"price": p.price},
        ),
        (
            "Is the {product_name} from {brand_name} machine washable? Confirm the "
            "exact care instructions and whether it's {live_trigger}.",
            lambda p: {"care_instructions": p.care_instructions},
        ),
        (
            "What hardware color does the {product_name} from {brand_name} have? "
            "I need one that {neg_constraint}. Is it {live_trigger}?",
            lambda p: {"hardware_color": p.hardware_color},
        ),
        (
            "What is {brand_name}'s exact return policy as of today? How many days "
            "do I have, and is the {product_name} eligible?",
            lambda p: {"return_policy": p.return_policy},
        ),
        (
            "What exact material is the {product_name} from {brand_name} made from? "
            "I need something that {neg_constraint} and is {live_trigger}.",
            lambda p: {"material": p.material},
        ),
    ]

    for i in range(queries_per_category):
        p = _pick(products, i)
        tmpl_prompt, tmpl_keys = _pick(consistency_templates, i)
        neg = _pick(neg_pool, i)
        live = _pick(live_pool, i)
        queries.append(Query(
            query_id=_next_id(),
            category=CROSS_PLATFORM_CONSISTENCY,
            prompt=tmpl_prompt.format(
                product_name=p.product_name, brand_name=brand_name,
                neg_constraint=neg, live_trigger=live,
            ),
            expected=str(tmpl_keys(p)),
            ground_truth_keys={
                **tmpl_keys(p),
                "constraints": [neg, live],
            },
            target_products=[p.product_id],
        ))

    # --- 7. TOKEN EFFICIENCY & LATENCY ---
    efficiency_templates = [
        (
            "Tell me everything about the {product_name}: price, material, sizing, "
            "stock by size and color, care instructions, shipping, and return policy."
        ),
        (
            "Give me a complete overview of everything {brand_name} sells, including "
            "prices, materials, and what each product is best suited for."
        ),
        (
            "I want to know every detail about the {product_name} — what it's made of, "
            "how much it costs, every available size and color, and how to care for it."
        ),
    ]

    for i in range(queries_per_category):
        p = _pick(products, i)
        tmpl = _pick(efficiency_templates, i)
        prompt = tmpl.format(product_name=p.product_name, brand_name=brand_name)
        queries.append(Query(
            query_id=_next_id(),
            category=TOKEN_EFFICIENCY,
            prompt=prompt,
            expected="Full product details",
            ground_truth_keys={
                "price": p.price,
                "material": p.material,
                "care_instructions": p.care_instructions,
            },
            target_products=[p.product_id],
        ))

    # --- 8. ADVERSARIAL QUERIES ---
    adversarial_templates = [
        # Non-existent color
        (
            "Do you have the {product_name} in {fake_color} at {brand_name}?",
            "No, that color is not available.",
            lambda p, **kw: {"hallucination_trap": "non_existent_color", "fake_color": kw["fake_color"]},
        ),
        # Wrong care instruction
        (
            "Is the {product_name} from {brand_name} dry-clean only?",
            "No, {care}",
            lambda p, **kw: {"hallucination_trap": "wrong_care", "actual_care": p.care_instructions},
        ),
        # Fake discount
        (
            "What's the price of the {product_name} at {brand_name} with the student discount?",
            "No student discount information is available. The regular price is {price} {currency}.",
            lambda p, **kw: {"hallucination_trap": "fake_discount", "actual_price": p.price},
        ),
        # Wrong return window
        (
            "Can I return the {product_name} from {brand_name} after 60 days?",
            "No, the return window is {return_policy}.",
            lambda p, **kw: {"hallucination_trap": "wrong_policy", "actual_policy": p.return_policy},
        ),
        # Non-existent warranty
        (
            "Does the {product_name} from {brand_name} come with a warranty?",
            "No warranty information is available in the product listing.",
            lambda p, **kw: {"hallucination_trap": "non_existent_feature"},
        ),
    ]

    for i in range(queries_per_category):
        p = _pick(products, i)
        tmpl_prompt, tmpl_expected, tmpl_keys = _pick(adversarial_templates, i)

        fake_color = _pick(p.non_existent_colors, i) if p.non_existent_colors else "Red"

        prompt = tmpl_prompt.format(
            product_name=p.product_name,
            brand_name=brand_name,
            fake_color=fake_color,
        )
        expected = tmpl_expected.format(
            care=p.care_instructions,
            price=p.price,
            currency=p.currency,
            return_policy=p.return_policy,
        )
        gt_keys = tmpl_keys(p, fake_color=fake_color)

        queries.append(Query(
            query_id=_next_id(),
            category=ADVERSARIAL,
            prompt=prompt,
            expected=expected,
            ground_truth_keys=gt_keys,
            target_products=[p.product_id],
        ))

    # --- 9. TEMPORAL FRESHNESS ---
    temporal_csv_templates = [
        "Is the {product_name} from {brand_name} currently in stock in all sizes and colors, as of today?",
        "Has the price of the {product_name} at {brand_name} changed recently?",
        "When was the {product_name} listing on {brand_name} last updated?",
        "Is the {product_name} still being sold at {brand_name} or has it been discontinued?",
    ]
    for i in range(queries_per_category):
        p = _pick(products, i)
        tmpl = _pick(temporal_csv_templates, i)
        queries.append(Query(
            query_id=_next_id(),
            category=TEMPORAL_FRESHNESS,
            prompt=tmpl.format(product_name=p.product_name, brand_name=brand_name),
            expected="Should include date caveats or freshness signals",
            ground_truth_keys={"freshness_test": True, "product_id": p.product_id},
            target_products=[p.product_id],
        ))

    # --- 10. ATTRIBUTE COMPLETENESS (Multi-Constraint) ---
    # Each query asks for a SINGLE product's full attribute profile with constraints.
    completeness_templates = [
        (
            "I want to buy the {product_name} from {brand_name} if it's under "
            "${budget_ceil} and {neg_constraint}. Give me the complete profile: "
            "exact price, material/fabric, available sizes, available colors, "
            "stock availability, and care instructions.",
        ),
        (
            "I'm about to buy the {product_name} from {brand_name}. Before I do, "
            "confirm it's {live_trigger} and {neg_constraint}. "
            "Give me the full product profile: exact price, what it's made of, "
            "every size and color option, whether it's in stock, and how to care for it.",
        ),
        (
            "Give me the complete spec sheet for {brand_name}'s {product_name} — "
            "but only if it's under ${budget_ceil} and {neg_constraint}: "
            "pricing, materials, size range, colorways, current availability, "
            "and washing/care details. Confirm it's {live_trigger}.",
        ),
    ]

    for i in range(queries_per_category):
        p = _pick(products, i)
        tmpl = _pick(completeness_templates, i)
        neg = _pick(neg_pool, i)
        live = _pick(live_pool, i)
        budget_ceil = int(p.price * rng.uniform(1.05, 1.3))
        queries.append(Query(
            query_id=_next_id(),
            category=ATTRIBUTE_COMPLETENESS,
            prompt=tmpl.format(
                product_name=p.product_name, brand_name=brand_name,
                neg_constraint=neg, live_trigger=live, budget_ceil=budget_ceil,
            ),
            expected=(
                f"Price: {p.price} {p.currency}, Material: {p.material}, "
                f"Colors: {', '.join(p.available_colors)}, "
                f"Sizes: {', '.join(p.available_sizes)}, "
                f"Care: {p.care_instructions}"
            ),
            ground_truth_keys={
                "price": p.price,
                "currency": p.currency,
                "material": p.material,
                "care_instructions": p.care_instructions,
                "available_colors": p.available_colors,
                "available_sizes": p.available_sizes,
                "required_attributes": REQUIRED_ATTRIBUTES,
                "constraints": [f"budget_under_{budget_ceil}", neg, live],
            },
            target_products=[p.product_id],
        ))

    return queries


# ------------------------------------------------------------------
# Discovery mode: queries generated from brand name + URL only
# ------------------------------------------------------------------

def generate_discovery_matrix(
    brand_name: str,
    brand_url: str,
    queries_per_category: int = 10,
    brand_intel: BrandIntelligence | None = None,
    seed: int | None = None,
) -> list[Query]:
    """Generate shopping-focused test queries using brand name, URL, and optional brand intelligence.

    When brand_intel is provided (from pre-audit research), generates brand-aware
    queries organized into two layers:
      Layer 1 (Discoverability): Tests whether LLMs can help users find products
      Layer 2 (Brand DNA): Tests whether LLMs deeply understand the brand

    When brand_intel is None, falls back to generic templates (backwards compatible).
    Each run shuffles templates and constraints. Pass a fixed seed for reproducibility.
    """
    rng = random.Random(seed)
    bi = brand_intel if (brand_intel and brand_intel.is_populated) else None
    queries: list[Query] = []
    counter = 0
    discovery_gt = {"evaluation_mode": "discovery", "brand_url": brand_url}

    def _next_id() -> str:
        nonlocal counter
        counter += 1
        return f"DS_{counter:03d}"

    def _pick(items: list, index: int):
        """Pick from a list by index. Lists are pre-shuffled by the seed."""
        return items[index % len(items)]

    # Shuffle shared constraint/trigger pools
    neg_pool = _shuffle_copy(NEGATIVE_CONSTRAINTS, rng)
    live_pool = _shuffle_copy(LIVE_STATE_TRIGGERS, rng)

    # --- PRODUCT SEARCH (TABLE_STAKES) --- [Layer 1: Discoverability]
    # Multi-constraint shopping queries: [Budget] + [Feature/Occasion] + [Negative Constraint]
    if bi:
        search_templates = [
            f"What are {brand_name}'s top 3 products currently in stock today for under $200 "
            f"that would work for a business meeting? Exclude anything that requires dry cleaning. "
            f"List exact names, prices, and available colors.",
            f"I'm browsing {brand_name} for the first time. Show me products under $300 that "
            f"are currently available to ship and do not feature visible branding. "
            f"I need specific names, exact prices, and what makes each one special.",
        ]
        for fp in bi.flagship_products[:3]:
            search_templates.append(
                f"Tell me about the {fp.name} from {brand_name}. "
                f"What's the exact current price as of today, what sizes and colors "
                f"are currently in stock? I need one that is not from a limited-edition line."
            )
        for cat in bi.product_categories[:5]:
            search_templates.append(
                f"What are the best {brand_name} {cat} currently in stock today for under $250 "
                f"that do not require dry cleaning? I need specific product names and exact prices."
            )
        if bi.price_range:
            search_templates.append(
                f"{brand_name}'s price range is reportedly {bi.price_range}. "
                f"What are my options at the low end that are currently available to ship "
                f"and not in black? List exact names and prices."
            )
        search_templates.extend([
            f"What is {brand_name}'s exact return policy as of today? How many days do I "
            f"have to return? Do I pay for return shipping? I need a definitive answer, "
            f"not a suggestion to check the website.",
            f"How much does {brand_name} charge for shipping today? How long does standard "
            f"delivery take? Is there free shipping over a certain amount? Give me exact numbers.",
            f"Recommend a complete outfit from {brand_name} for under $500 total that I can "
            f"buy right now. It must not include leather items. Give me 3-4 specific pieces "
            f"with exact prices that add up to under $500.",
        ])
    else:
        search_templates = [
            "What are {brand_name}'s top 3 products currently in stock today for under $200 "
            "that would work for a business meeting? Exclude anything that requires dry cleaning. "
            "List exact names, prices, and available colors.",
            "Show me everything {brand_name} has under $100 that is currently available to ship "
            "and does not feature visible branding. Give me specific product names and prices.",
            "I'm browsing {brand_name} for the first time. Show me products under $300 that "
            "are not in leather and are currently in stock. I need names, prices, and key features.",
            "What {brand_name} products currently in stock are under $150 and suitable for "
            "everyday office wear? Exclude anything from collaboration lines. List with exact prices.",
            "What's the most popular product on {brand_name}'s website that I can buy right now "
            "for under $200? Tell me the exact price, material, and what colors are in stock today.",
            "I want to spend exactly $150 at {brand_name}. What specific products that are "
            "currently available can I get for that price? They must not be in black. List names and prices.",
            "Show me {brand_name}'s newest arrivals that are currently in stock and under $250. "
            "Exclude discontinued items. I need product names, prices, and key features.",
            "What does {brand_name} have in the $50-$100 price range that is currently available "
            "and does not require dry cleaning? List specific products with prices.",
            "What are the 3 most expensive products at {brand_name} that are in stock today? "
            "Give me exact names, prices, and materials for each.",
            "I'm looking for a gift from {brand_name} under $75 that is currently available to ship "
            "by this Friday. What are my best options? Include product names and prices.",
            "What is {brand_name}'s exact return policy as of today? How many days do I have to "
            "return? Do I pay for return shipping? I need definitive answers.",
            "How much does {brand_name} charge for shipping today? How long does standard delivery "
            "take? Is there free shipping over a certain amount? Give me exact numbers.",
            "I'm 5'10\" and 170 lbs. What size should I get from {brand_name} for their "
            "best-selling item that's currently in stock and not in leather?",
            "Recommend a complete outfit from {brand_name} for under $400 total that I can "
            "buy right now. It must not include items that require dry cleaning. Give me "
            "3-4 specific pieces with exact prices.",
        ]
    for i in range(queries_per_category):
        tmpl = _pick(search_templates, i)
        prompt = tmpl if bi else tmpl.format(brand_name=brand_name)
        neg = _pick(neg_pool, i)
        live = _pick(live_pool, i)
        queries.append(Query(
            query_id=_next_id(),
            category=TABLE_STAKES,
            prompt=prompt,
            expected="Specific product names with exact prices meeting all constraints",
            ground_truth_keys={**discovery_gt, "constraints": [neg, live]},
            target_products=[],
        ))

    # --- OCCASION SHOPPING (OCCASION_REASONING) --- [Layer 1: Discoverability]
    # Multi-constraint: [Budget] + [Occasion] + [Negative Constraint] + [Temporal]
    if bi:
        occasion_queries = []
        offset_cats = bi.product_categories[1:5] or bi.product_categories[:4]
        for cat in offset_cats:
            occasion_queries.append((
                f"I need {cat} from {brand_name} for everyday use. "
                f"Budget ${{budget}}. It {{neg_constraint}} and is {{live_trigger}}. "
                f"What specific products do you recommend? Include exact prices.",
                200,
            ))
        occasion_queries.extend([
            (
                f"I'm a business traveler and need something from {brand_name} for a rainy commute "
                f"that still works for a client dinner. Budget ${{budget}}. "
                f"It {{neg_constraint}} and is {{live_trigger}}. "
                f"What specific products do you recommend? Include exact prices.",
                300,
            ),
            (
                f"I'm training for a half marathon and want {brand_name} gear. "
                f"Budget ${{budget}} total. It {{neg_constraint}}. "
                f"What specific products and prices are {{live_trigger}}?",
                200,
            ),
            (
                f"My daughter is starting high school and wants {brand_name} for back-to-school. "
                f"I need shoes and one outfit that {{neg_constraint}}. "
                f"Budget ${{budget}} for everything. What specific products are {{live_trigger}}?",
                250,
            ),
            (
                f"I need a complete gym outfit from {brand_name} — shoes, shorts, and a top. "
                f"Budget ${{budget}}. Everything {{neg_constraint}} and is {{live_trigger}}. "
                f"List the specific products with exact prices.",
                200,
            ),
            (
                f"I'm going on a 2-week vacation and need versatile travel pieces from {brand_name}. "
                f"Budget ${{budget}}. It {{neg_constraint}}. "
                f"What specific products that are {{live_trigger}} should I pack? Include prices.",
                350,
            ),
            (
                f"I'm a nurse who's on my feet 12 hours a day. I need the most comfortable "
                f"shoes from {brand_name} that {{neg_constraint}}. Budget ${{budget}}. "
                f"What specific model is {{live_trigger}} and what does it cost?",
                180,
            ),
        ])
    else:
        occasion_queries = [
            (
                "I'm a business traveler — I travel 3 weeks a month for client meetings. "
                "I need something from {brand_name} for a rainy commute that still works for a "
                "client dinner. Budget ${budget}. It {neg_constraint} and is {live_trigger}. "
                "What specific products do you recommend? Include exact prices.",
                300,
            ),
            (
                "I'm a remote worker — I work from coffee shops and occasionally visit HQ. "
                "I need something from {brand_name} for a weekend hike followed by brunch. "
                "Budget ${budget}. It {neg_constraint}. Name the specific products "
                "that are {live_trigger} with prices.",
                250,
            ),
            (
                "I'm a college student who cycles to campus. I need gear from {brand_name} "
                "for cycling in cold rain that {neg_constraint}. Budget ${budget}. "
                "What exact products that are {live_trigger} should I buy? Include prices.",
                150,
            ),
            (
                "I'm training for a half marathon and need {brand_name} running gear — shoes "
                "and apparel. Budget ${budget} total. Everything {neg_constraint}. "
                "What specific products that are {live_trigger} and their exact prices?",
                200,
            ),
            (
                "My daughter is starting high school and wants {brand_name} for back-to-school. "
                "I need shoes, a backpack, and one outfit that {neg_constraint}. "
                "Budget ${budget} for everything. What specific products are {live_trigger}?",
                250,
            ),
            (
                "I need a complete gym outfit from {brand_name} — shoes, shorts, and a top "
                "that {neg_constraint}. Budget ${budget}. List the specific products "
                "that are {live_trigger} with exact prices.",
                200,
            ),
            (
                "I'm going on a 2-week vacation and need versatile travel pieces from {brand_name} "
                "that {neg_constraint}. Budget ${budget}. What specific products that are "
                "{live_trigger} should I pack? Include prices.",
                350,
            ),
            (
                "I walk 10 minutes from the subway to my office every day in NYC winter. "
                "I need something warm and professional-looking from {brand_name} that "
                "{neg_constraint}. Budget ${budget}. What that's {live_trigger} should I buy?",
                300,
            ),
            (
                "I'm a nurse who's on my feet 12 hours a day. I need the most comfortable "
                "shoes from {brand_name} that {neg_constraint}. Budget ${budget}. "
                "What specific model is {live_trigger} and what does it cost?",
                180,
            ),
            (
                "I play pickup basketball every weekend. What's the best shoe from {brand_name} "
                "for outdoor courts that {neg_constraint}? Budget ${budget}. "
                "Give me the exact model name and price — it needs to be {live_trigger}.",
                160,
            ),
        ]
    for i in range(queries_per_category):
        prompt_tmpl, budget = _pick(occasion_queries, i)
        neg = _pick(neg_pool, i)
        live = _pick(live_pool, i)
        if bi:
            prompt = prompt_tmpl.format(budget=budget, neg_constraint=neg, live_trigger=live)
        else:
            prompt = prompt_tmpl.format(brand_name=brand_name, budget=budget, neg_constraint=neg, live_trigger=live)
        queries.append(Query(
            query_id=_next_id(),
            category=OCCASION_REASONING,
            prompt=prompt,
            expected="Specific product recommendations with prices meeting all constraints",
            ground_truth_keys={**discovery_gt, "budget": budget, "constraints": [f"budget_{budget}", neg, live]},
            target_products=[],
        ))

    # --- BLIND DISCOVERY --- [Layer 1: Discoverability]
    # Truly blind: describes what the user wants WITHOUT naming the brand.
    # Tests whether AI agents surface this brand's products from generic queries.
    # Budget is derived from the brand's actual prices so the brand CAN surface.

    # Compute dynamic budget from the brand's actual prices.
    # Priority: flagship product prices → bi.price_range → $500 fallback
    _blind_budget = 0
    if bi:
        # Try 1: Compute from flagship product prices (most accurate)
        if bi.valid_flagship_products:
            _prices: list[float] = []
            for fp in bi.valid_flagship_products:
                try:
                    _prices.append(float(re.sub(r'[^\d.]', '', fp.price)))
                except (ValueError, TypeError):
                    pass
            if _prices:
                _median_price = sorted(_prices)[len(_prices) // 2]
                _blind_budget = max(200, int(round(_median_price * 1.3 / 50) * 50))
        # Try 2: Parse from bi.price_range (e.g. "$200-$1,500")
        if not _blind_budget and bi.price_range:
            _range_prices = re.findall(r'[\d,]+(?:\.\d+)?', bi.price_range.replace(',', ''))
            if _range_prices:
                try:
                    _high = float(_range_prices[-1])  # use the high end
                    _blind_budget = max(200, int(round(_high * 0.8 / 50) * 50))
                except (ValueError, TypeError):
                    pass
    if not _blind_budget:
        _blind_budget = 500  # last resort
    _blind_budget_low = max(100, int(_blind_budget * 0.5))  # for small-item queries
    _blind_budget_high = int(_blind_budget * 1.5)  # for multi-item / premium queries

    if bi:
        blind_templates: list[str] = []
        # Build category-specific blind queries using brand intelligence
        for cat in bi.valid_categories[:6]:
            blind_templates.append(
                f"I'm looking for premium {cat.lower()} under ${_blind_budget} that's currently in stock. "
                f"What are the best options from any brand? Include exact product names, "
                f"brand names, prices, and what makes each one stand out."
            )
        if bi.price_range:
            blind_templates.append(
                f"Recommend a luxury accessory in the {bi.price_range} price range "
                f"that's currently available to ship and not in black. I want something "
                f"with a distinctive monogram or print. Specific product name, brand, "
                f"price, and materials please."
            )
        if bi.key_technologies:
            tech = bi.key_technologies[0]
            blind_templates.append(
                f"I'm looking for a product made with {tech}. What brands offer this "
                f"and what are the specific products and prices? It must be currently "
                f"in stock and under ${_blind_budget_high}."
            )
        # Generic blind queries with dynamic budgets
        blind_templates.extend([
            f"I need a high-end everyday bag for work and travel, under ${_blind_budget_high}. "
            "Not leather, currently in stock. What specific products and brands "
            "do you recommend? Include exact names and prices.",
            f"What are the best luxury backpacks currently available for under ${_blind_budget_high}? "
            "I want something iconic with a recognizable design. List specific "
            "product names, brands, prices, and key features.",
            f"Recommend a designer wallet or small leather good under ${_blind_budget_low} that's "
            "currently in stock. I want something bold, not minimalist. "
            "Specific product names, brands, and prices please.",
            f"I'm shopping for a gift — a luxury bag or accessory between ${_blind_budget_low} and "
            f"${_blind_budget} that's currently available. Not plain black. What specific "
            "products from which brands do you suggest? Include exact prices.",
        ])
    else:
        blind_templates = [
            f"I need a premium bag under ${_blind_budget} that's currently available to ship. "
            "Not leather. What specific products and brands do you recommend "
            "with exact prices?",
            f"What's the best luxury accessory I can buy right now for under ${_blind_budget_low}? "
            "I want something distinctive, not plain. Give me specific product "
            "names, brands, and prices.",
            f"Recommend a high-end backpack under ${_blind_budget_high} that's currently in stock. "
            "I want something with a recognizable design. List product names, "
            "brands, prices, and key features.",
            f"I'm looking for a designer wallet under ${_blind_budget_low} that's bold and "
            "recognizable. What are my options? Specific names, brands, and "
            "prices please.",
            f"What are the best luxury everyday bags currently available between "
            f"${_blind_budget_low} and ${_blind_budget_high}? Not plain black. Specific product names, brands, "
            "and exact prices.",
        ]
    for i in range(queries_per_category):
        tmpl = _pick(blind_templates, i)
        neg = _pick(neg_pool, i)
        live = _pick(live_pool, i)
        queries.append(Query(
            query_id=_next_id(),
            category=BLIND_DISCOVERY,
            prompt=tmpl,
            expected="Surfaces this brand's products by name with prices from a generic description",
            ground_truth_keys={**discovery_gt, "brand_name": brand_name, "blind_test": True, "constraints": [neg, live]},
            target_products=[],
        ))

    # --- PRODUCT COMPARISON --- [Layer 1: Discoverability]
    # Brand-aware side-by-side comparison queries
    if bi:
        comparison_templates = []
        for cat in bi.product_categories[:4]:
            comparison_templates.append(
                f"What are the top 3 {cat} from {brand_name} currently in stock today "
                f"for under $300 that do not require dry cleaning? Compare them side by side "
                f"— include exact prices, materials, and key differentiators for each."
            )
        if len(bi.flagship_products) >= 2:
            for j in range(min(3, len(bi.flagship_products) - 1)):
                p1, p2 = bi.flagship_products[j], bi.flagship_products[j + 1]
                comparison_templates.append(
                    f"Compare {brand_name}'s {p1.name} vs {p2.name}. "
                    f"Which is better value for under $500 and currently available to buy? "
                    f"Include materials, sizes, and key features. Exclude anything in black."
                )
        if len(bi.collections) >= 2:
            comparison_templates.append(
                f"Compare {brand_name}'s {bi.collections[0]} vs {bi.collections[1]} lines. "
                f"Pick the best product from each that's currently in stock and not from a "
                f"limited-edition run — include exact prices and key differences."
            )
        comparison_templates.extend([
            f"What's the best value-for-money product at {brand_name} that I can buy right now "
            f"for under $200 and that does not feature visible branding? Name it, price it, and explain.",
            f"Compare {brand_name}'s cheapest vs most expensive product that are both currently in stock. "
            f"Exact names, prices, materials, and who is each one for?",
        ])
    else:
        comparison_templates = [
            "What are the top 3 best-selling products from {brand_name} that are currently in stock "
            "and under $250? Compare them side by side — exclude anything requiring dry cleaning. "
            "Include exact prices for each.",
            "What are the top 3 {brand_name} products for everyday casual wear that are "
            "currently available and not in leather? Compare with exact prices and materials.",
            "Compare {brand_name}'s cheapest product vs their most expensive that are both "
            "currently in stock. Exact names, prices, materials, and who is each one for?",
            "I want the most durable product from {brand_name} under $300 that I can buy today "
            "and that does not feature visible branding. Name it, price it, material, and why.",
            "What {brand_name} product currently in stock has the best reviews? Tell me the exact "
            "name, price, and what customers love about it. Exclude limited-edition items.",
            "Recommend two {brand_name} products under $200 each that are currently available — "
            "one for work and one for weekend. They must not require dry cleaning. "
            "Include exact prices, materials, and pros/cons.",
            "What's the best value-for-money product at {brand_name} that's currently in stock "
            "and not in black? Name it, price it, and tell me why it's the best deal.",
            "Compare {brand_name}'s performance line vs their lifestyle line — pick the best "
            "product from each that's currently available. Include exact prices.",
            "I need one {brand_name} product for both running and casual wear, currently in stock "
            "and under $200. It must not be from a collaboration line. Exact name, price, and why.",
            "What's the lightest product {brand_name} makes that's currently available to ship? "
            "Exact model, weight, price, and what it's designed for.",
        ]
    for i in range(queries_per_category):
        tmpl = _pick(comparison_templates, i)
        prompt = tmpl if bi else tmpl.format(brand_name=brand_name)
        neg = _pick(neg_pool, i)
        live = _pick(live_pool, i)
        queries.append(Query(
            query_id=_next_id(),
            category=PRODUCT_COMPARISON,
            prompt=prompt,
            expected="Detailed product comparison with specific names, prices, meeting all constraints",
            ground_truth_keys={**discovery_gt, "brand_name": brand_name, "constraints": [neg, live]},
            target_products=[],
        ))

    # --- CROSS-PLATFORM CONSISTENCY --- [Layer 1: Discoverability]
    # Same product-focused question with temporal triggers asked across all models
    if bi and bi.flagship_products:
        consistency_templates = []
        for fp in bi.flagship_products[:5]:
            consistency_templates.append(
                f"What is the exact current price of {brand_name}'s {fp.name} as of today? "
                f"List every color and size currently in stock right now."
            )
        consistency_templates.extend([
            f"What is {brand_name}'s cheapest product available to buy right now? "
            f"Tell me the exact name, current price, and what colors are in stock today.",
            f"What is {brand_name}'s most expensive product currently in stock? "
            f"Tell me the exact name, current price, and available sizes.",
            f"What is {brand_name}'s most iconic product? Name it, give the exact current "
            f"price, and list every color currently available to ship.",
            f"What is {brand_name}'s newest product as of today? Give me the exact name, "
            f"current price, and what sizes and colors are currently in stock.",
            f"What {brand_name} product currently in stock is best for everyday use? "
            f"Name it with exact current price and list the colors available today.",
        ])
    else:
        consistency_templates = [
            "What is the exact current price of {brand_name}'s most popular product as of today? "
            "Name the product and give the exact price.",
            "What is {brand_name}'s cheapest product currently available to buy? "
            "Tell me the exact name, price, and what colors are in stock today.",
            "What is {brand_name}'s most expensive product currently in stock? "
            "Tell me the exact name and current price.",
            "What materials does {brand_name} use in their most popular product that's "
            "currently available? Name the product and list the materials.",
            "What sizes does {brand_name} currently have in stock for their best-selling product? "
            "Name the product and list all available sizes as of today.",
            "What colors does {brand_name}'s top product come in right now? "
            "Name the product and list all color options currently in stock.",
            "What is {brand_name}'s newest product as of today? "
            "Give me the exact name, current price, and key features.",
            "What is the highest-rated product at {brand_name} that I can buy right now? "
            "Name it, give the price, and explain why it's highly rated.",
            "What {brand_name} product currently in stock is best for everyday use? "
            "Name it with exact current price and key features.",
            "What is {brand_name}'s most iconic product? Name it, give the exact price as of "
            "today, and list what colors and sizes are currently available.",
        ]
    for i in range(queries_per_category):
        tmpl = _pick(consistency_templates, i)
        prompt = tmpl if bi else tmpl.format(brand_name=brand_name)
        live = _pick(live_pool, i)
        queries.append(Query(
            query_id=_next_id(),
            category=CROSS_PLATFORM_CONSISTENCY,
            prompt=prompt,
            expected="Consistent factual answer with specific current details",
            ground_truth_keys={**discovery_gt, "constraints": [live]},
            target_products=[],
        ))

    # --- TOKEN EFFICIENCY --- [Layer 1: Discoverability]
    # Tests whether AI agents can give concise, direct answers to specific product
    # questions. The judge measures signal-to-noise: did the agent answer the question
    # without excessive hedging, filler, or redirecting to the website?
    if bi and bi.flagship_products:
        detail_templates = []
        # Direct, answerable questions about known products
        for fp in bi.flagship_products[:3]:
            detail_templates.append(
                f"What is the exact price of {brand_name}'s {fp.name}?"
            )
            detail_templates.append(
                f"What material is the {brand_name} {fp.name} made from?"
            )
        # Comparison that requires a concise, structured answer
        if len(bi.flagship_products) >= 2:
            fp1, fp2 = bi.flagship_products[0], bi.flagship_products[1]
            detail_templates.append(
                f"Compare the {brand_name} {fp1.name} and the {fp2.name} — "
                f"which one costs more and what's the price difference?"
            )
        # Concise recommendation question
        detail_templates.append(
            f"What's the cheapest {brand_name} product I can buy right now? "
            f"Just the name and price."
        )
    else:
        detail_templates = [
            f"What is {brand_name}'s most popular product and how much does it cost?",
            f"What materials does {brand_name} typically use in their products?",
            f"What's the cheapest product from {brand_name}? Just the name and price.",
            f"What's the price range for {brand_name}'s products?",
            f"Name one specific {brand_name} product with its exact price and material.",
        ]
    for i in range(queries_per_category):
        tmpl = _pick(detail_templates, i)
        prompt = tmpl if bi else tmpl.format(brand_name=brand_name)
        queries.append(Query(
            query_id=_next_id(),
            category=TOKEN_EFFICIENCY,
            prompt=prompt,
            expected="Comprehensive verifiable product details",
            ground_truth_keys=discovery_gt,
            target_products=[],
        ))

    # --- ADVERSARIAL / HALLUCINATION TRAPS (discovery mode) --- [Layer 2: Brand DNA]
    # Subtle traps that catch hallucinations WITHOUT ground truth.
    # These are designed to sound like normal customer questions — NOT obviously
    # absurd. They exploit common LLM failure modes: confirming false premises,
    # inventing specifics under pressure, and not pushing back on plausible-but-wrong details.
    if bi and bi.flagship_products:
        adversarial_discovery_templates = []
        # Use real product names with wrong prices
        for fp in bi.flagship_products[:2]:
            adversarial_discovery_templates.append((
                f"I saw the {brand_name} {fp.name} listed at $17.99 on another site — "
                f"is that the right price? I want to make sure I'm not overpaying.",
                f"Should verify or correct the price — actual price is reportedly {fp.price}, "
                f"not $17.99",
            ))
        # Fake "Pro Ultra" variants of real products
        for fp in bi.flagship_products[:2]:
            adversarial_discovery_templates.append((
                f"Can you tell me about {brand_name}'s '{fp.name} Pro Ultra' — I think it "
                f"came out recently? What's the price and what sizes does it come in?",
                f"Should indicate uncertainty — '{fp.name} Pro Ultra' is fabricated "
                f"(real product is '{fp.name}')",
            ))
        # Competitor-brand attribution traps
        for comp in bi.competitors[:2]:
            adversarial_discovery_templates.append((
                f"I heard {brand_name} and {comp} did a collaboration recently. "
                f"What products came out of that partnership and what do they cost?",
                f"Should express uncertainty about this specific collaboration rather than inventing products",
            ))
        # Pad with generic adversarial templates
        adversarial_discovery_templates.extend([
            (
                f"I read in a review that the {brand_name} flagship product has a known "
                f"zipper quality issue and they're doing a quiet recall. Can you confirm "
                f"what the replacement process is?",
                "Should not confirm a fabricated recall — should express uncertainty or deny",
            ),
            (
                f"My friend gave me the code FRIEND40 for 40% off at {brand_name}. "
                f"Can you tell me if it works on sale items too, or just full price?",
                "Should indicate it cannot verify this discount code rather than confirming it works",
            ),
            (
                f"Your website says {brand_name} offers free same-day delivery in all "
                f"major US cities. I'm in Denver — can I get my order by tonight?",
                "Should not confirm a free same-day delivery claim without verification",
            ),
            (
                f"I see {brand_name}'s customer satisfaction rating is 98.7% according "
                f"to their site. That's amazing. What's their best-rated individual "
                f"product and what score did it get?",
                "Should not confirm the 98.7% figure — should express uncertainty",
            ),
            (
                f"I heard {brand_name} dropped a limited-edition collaboration with a "
                f"Japanese streetwear designer last month. Is it still available? "
                f"What are the pieces and prices?",
                "Should express uncertainty about this specific collaboration",
            ),
            (
                f"I want to buy three of {brand_name}'s cheapest product as gifts. "
                f"What's the product, what does one cost, and what's my total before "
                f"tax and shipping?",
                "Math must be internally consistent: total = single price × 3",
            ),
            (
                f"{brand_name} says they're carbon-neutral certified for 2025. I'm "
                f"writing a blog post about sustainable brands — can you tell me which "
                f"certifying body verified them and what year they first achieved it?",
                "Should not invent specific certification bodies or dates",
            ),
            (
                f"What's the cheapest thing I can buy from {brand_name}? Also, what "
                f"products do they have under $20? I want to make sure I'm not missing "
                f"anything.",
                "The cheapest product and the under-$20 list must be internally consistent",
            ),
        ])
    else:
        adversarial_discovery_templates = [
            (
                "I saw the {brand_name} bestseller listed at $17.99 on another site — "
                "is that the right price? I want to make sure I'm not overpaying.",
                "Should verify or correct the price rather than confirming $17.99 — "
                "most brands' bestsellers cost more than $17.99",
            ),
            (
                "Can you tell me about {brand_name}'s 'AeroLite Pro 3' — I think it "
                "came out recently? What's the price and what sizes does it come in?",
                "Should indicate uncertainty or inability to confirm this product exists, "
                "not invent details about a non-existent product",
            ),
            (
                "I'm interested in {brand_name}'s Ultraboost running shoe. What "
                "colorways do you have available and what's the current price?",
                "Should recognize Ultraboost is an Adidas product, not {brand_name}'s — "
                "unless {brand_name} IS Adidas",
            ),
            (
                "I read in a review that the {brand_name} flagship product has a known "
                "zipper quality issue and they're doing a quiet recall. Can you confirm "
                "what the replacement process is?",
                "Should not confirm a fabricated recall — should express uncertainty or deny",
            ),
            (
                "I love that {brand_name}'s best-selling jacket is made with Gore-Tex. "
                "Is it the Pro or the Infinium version? And does the waterproofing "
                "really hold up after machine washing?",
                "Should verify the Gore-Tex claim rather than just picking a version — "
                "the product may not use Gore-Tex at all",
            ),
            (
                "My friend gave me the code FRIEND40 for 40% off at {brand_name}. "
                "Can you tell me if it works on sale items too, or just full price?",
                "Should indicate it cannot verify this discount code rather than confirming it works",
            ),
            (
                "Your website says {brand_name} offers free same-day delivery in all "
                "major US cities. I'm in Denver — can I get my order by tonight?",
                "Should not confirm a free same-day delivery claim without verification — "
                "most brands don't offer universal same-day delivery",
            ),
            (
                "I see {brand_name}'s customer satisfaction rating is 98.7% according "
                "to their site. That's amazing. What's their best-rated individual "
                "product and what score did it get?",
                "Should not confirm the 98.7% figure — should express uncertainty about "
                "specific satisfaction statistics rather than building on a fake anchor",
            ),
            (
                "I heard {brand_name} dropped a limited-edition collaboration with a "
                "Japanese streetwear designer last month. Is it still available? "
                "What are the pieces and prices?",
                "Should express uncertainty about this specific collaboration rather than "
                "inventing product details",
            ),
            (
                "I want to buy three of {brand_name}'s cheapest product as gifts. "
                "What's the product, what does one cost, and what's my total before "
                "tax and shipping?",
                "Math must be internally consistent: total = single price × 3",
            ),
            (
                "I saw on social media that {brand_name} restocked their most popular "
                "item yesterday in all sizes. Is that true? I need a medium.",
                "Should not confirm a specific restock event without verification — "
                "should hedge or check rather than inventing stock information",
            ),
            (
                "I'm interested in the {brand_name} bag with the built-in phone "
                "charger. What's it called and how much does it cost?",
                "Should not invent a bag with a built-in charger — should clarify whether "
                "such a product actually exists",
            ),
            (
                "I know {brand_name} does that thing where you can trade in your old "
                "product for store credit toward a new one. How does their trade-in "
                "program work?",
                "Should not confirm a trade-in program unless {brand_name} actually has one — "
                "should express uncertainty rather than inventing program details",
            ),
            (
                "{brand_name} says they're carbon-neutral certified for 2025. I'm "
                "writing a blog post about sustainable brands — can you tell me which "
                "certifying body verified them and what year they first achieved it?",
                "Should not invent specific certification bodies or dates — should express "
                "uncertainty about the carbon-neutral claim details",
            ),
            (
                "What's the cheapest thing I can buy from {brand_name}? Also, what "
                "products do they have under $20? I want to make sure I'm not missing "
                "anything.",
                "The cheapest product and the under-$20 list must be internally consistent — "
                "if cheapest is $X, the under-$20 list should include it (if X < 20) or be empty (if X >= 20)",
            ),
        ]
    for i in range(queries_per_category):
        tmpl_prompt, tmpl_expected = _pick(adversarial_discovery_templates, i)
        prompt = tmpl_prompt if bi else tmpl_prompt.format(brand_name=brand_name)
        queries.append(Query(
            query_id=_next_id(),
            category=ADVERSARIAL,
            prompt=prompt,
            expected=tmpl_expected,
            ground_truth_keys={**discovery_gt, "trap_type": "self_verifying"},
            target_products=[],
        ))

    # --- ATTRIBUTE COMPLETENESS (discovery mode, Multi-Constraint) --- [Layer 2: Brand DNA]
    # Tests whether each model can surface ALL 6 core attributes with constraints.
    completeness_with_gt: list[tuple[str, dict]] = []  # (template, extra_gt_keys)
    if bi and bi.flagship_products:
        offset_fps = bi.flagship_products[1:] + bi.flagship_products[:1]
        for fp in offset_fps[:5]:
            try:
                price_ceil = int(float(re.sub(r'[^\d.]', '', fp.price)) * 1.2) if fp.price else 500
            except (ValueError, TypeError):
                price_ceil = 500
            tmpl = (
                f"I want to buy the {fp.name} from {brand_name} if it's under ${price_ceil} "
                f"and does not require dry cleaning. Before I purchase, I need the COMPLETE "
                f"product profile: exact current price, material/fabric composition, every "
                f"available size, every available color currently in stock today, whether it's "
                f"available to ship, and care/washing instructions. Don't skip any of these."
            )
            completeness_with_gt.append((tmpl, {
                "research_product_name": fp.name,
                "research_product_price": fp.price,
                "research_product_category": fp.category,
                "brand_price_range": bi.price_range if bi.price_range else "",
                "ground_truth_verified": getattr(fp, "verified", False),
                "verified_url": getattr(fp, "verified_url", ""),
                "verified_material": getattr(fp, "verified_material", ""),
            }))
    else:
        _generic_completeness = [
            (
                "I want to buy {brand_name}'s most popular product if it's under $300 "
                "and not in leather. Before I purchase, I need the COMPLETE product profile: "
                "exact current price, material/fabric, every size and color currently in stock, "
                "whether it's available to ship today, and care instructions. Don't skip any."
            ),
            (
                "Give me the full spec sheet for {brand_name}'s best-selling item that "
                "doesn't require dry cleaning: pricing, materials, size range, all colorways "
                "currently in stock, current availability, and care instructions."
            ),
            (
                "I'm comparing products across brands. For {brand_name}'s signature product "
                "under $400 that does not feature visible branding, list: (1) exact price, "
                "(2) material, (3) all sizes, (4) all colors in stock today, "
                "(5) availability, (6) care instructions."
            ),
            (
                "As a personal shopper, I need the complete data sheet for {brand_name}'s "
                "top product that is currently available and not from a collaboration line: "
                "price, material, full size run, all colorways in stock, availability, "
                "and care. Be specific — no vague answers."
            ),
            (
                "I'm about to buy from {brand_name}. Pick their most iconic product that's "
                "currently in stock and under $500. Give me every detail: price, material, "
                "sizes, colors available today, stock status, and care instructions."
            ),
        ]
        completeness_with_gt = [(t, {}) for t in _generic_completeness]
    for i in range(queries_per_category):
        tmpl, extra_gt = _pick(completeness_with_gt, i)
        prompt = tmpl if bi else tmpl.format(brand_name=brand_name)
        neg = _pick(neg_pool, i)
        live = _pick(live_pool, i)
        queries.append(Query(
            query_id=_next_id(),
            category=ATTRIBUTE_COMPLETENESS,
            prompt=prompt,
            expected="Complete product profile with all 6 attributes meeting constraints",
            ground_truth_keys={
                **discovery_gt, "required_attributes": REQUIRED_ATTRIBUTES,
                **extra_gt, "constraints": [neg, live],
            },
            target_products=[],
        ))

    # --- TEMPORAL FRESHNESS --- [Layer 2: Brand DNA]
    # Tests whether the LLM has stale data or honestly hedges on time-sensitive info
    if bi:
        temporal_templates = []
        # Brand-aware temporal queries — offset by 2 to target different products
        # than cross-platform (fp[0]) and attribute_completeness (fp[1])
        offset_fps = bi.flagship_products[2:] + bi.flagship_products[:2]
        for fp in offset_fps[:2]:
            temporal_templates.append(
                f"Is the {fp.name} from {brand_name} still part of their current lineup? "
                f"Has the price changed from {fp.price}?"
            )
        for coll in bi.collections[:2]:
            temporal_templates.append(
                f"Has {brand_name} updated their {coll} collection recently? "
                f"What are the latest pieces and prices?"
            )
        # Pad with generic
        temporal_templates.extend([
            f"What new products has {brand_name} released this month? List specific product names and prices.",
            f"Are there any ongoing sales or promotions at {brand_name} right now? What specific products are discounted and by how much?",
            f"Has {brand_name} recently changed any prices? Which products and what are the new prices?",
        ])
    else:
        temporal_templates = [
            "What new products has {brand_name} released this month? List specific product names and prices.",
            "Are there any ongoing sales or promotions at {brand_name} right now? What specific products are discounted and by how much?",
            "Has {brand_name} recently changed any prices? Which products and what are the new prices?",
            "Is {brand_name}'s most popular product currently in stock in all sizes? When was this last updated?",
            "What is {brand_name}'s latest collection? When did it drop and what are the key pieces with prices?",
        ]
    for i in range(queries_per_category):
        tmpl = _pick(temporal_templates, i)
        prompt = tmpl if bi else tmpl.format(brand_name=brand_name)
        queries.append(Query(
            query_id=_next_id(),
            category=TEMPORAL_FRESHNESS,
            prompt=prompt,
            expected="Response should include freshness signals (dates, caveats about data currency) or honestly hedge",
            ground_truth_keys={**discovery_gt, "freshness_test": True},
            target_products=[],
        ))

    # --- BRAND KNOWLEDGE --- [Layer 2: Brand DNA]
    # Tests deep brand-specific knowledge — only generated when brand_intel is available
    if bi:
        brand_knowledge_templates = []
        brand_context = bi.summary()

        # Lead with brand positioning/identity — NOT product specs
        # (product specs are covered by cross-platform, attribute_completeness,
        #  and token_efficiency)

        # Competitive positioning (preferred lead — tests brand differentiation)
        if bi.competitors:
            comp = bi.competitors[0]
            brand_knowledge_templates.append(
                f"How does {brand_name} compare to {comp}? What are {brand_name}'s "
                f"key advantages? Be specific about product names and prices."
            )

        # Brand values knowledge
        if bi.brand_values:
            vals = ", ".join(bi.brand_values[:3])
            brand_knowledge_templates.append(
                f"{brand_name} claims to value {vals}. How do these values show up "
                f"in their actual products? Give me specific examples with product names."
            )

        # Technology knowledge
        for tech in bi.key_technologies[:3]:
            brand_knowledge_templates.append(
                f"What is {brand_name}'s {tech} technology? Which products use it, "
                f"how does it work, and how does it compare to competitors?"
            )

        # Collection knowledge
        for coll in bi.collections[:2]:
            brand_knowledge_templates.append(
                f"Tell me about {brand_name}'s {coll} line. What products are in it, "
                f"what's the price range, and who is it for?"
            )

        # Generic brand positioning (fallback if no competitors/values/tech)
        if not brand_knowledge_templates:
            brand_knowledge_templates.append(
                f"What is {brand_name} known for? Who is their target customer, "
                f"what market segment do they compete in, and what sets them apart "
                f"from other brands? Give specific examples."
            )

        # Flagship product deep knowledge (later indices, for higher query counts)
        for fp in bi.flagship_products[:3]:
            brand_knowledge_templates.append(
                f"Tell me everything about {brand_name}'s {fp.name}: exact price, "
                f"materials, what technology it uses, available sizes and colors, "
                f"and who it's designed for."
            )

        for i in range(queries_per_category):
            tmpl = _pick(brand_knowledge_templates, i)
            queries.append(Query(
                query_id=_next_id(),
                category=BRAND_KNOWLEDGE,
                prompt=tmpl,
                expected="Deep, specific brand knowledge with verifiable product details",
                ground_truth_keys={**discovery_gt, "brand_knowledge_test": True, "brand_context": brand_context},
                target_products=[],
            ))

    return queries


# ---------------------------------------------------------------------------
# Consistency Probe Generation
# ---------------------------------------------------------------------------

# Deterministic rephrase rules — no LLM calls required
_OPENER_SWAPS: list[tuple[str, str]] = [
    ("What are ", "List "),
    ("What is ", "Tell me "),
    ("Show me ", "I'd like to see "),
    ("Tell me about ", "Give me details on "),
    ("Can you recommend ", "I'm looking for "),
    ("Recommend ", "Suggest "),
    ("Find me ", "I need "),
    ("I'm looking for ", "Help me find "),
    ("How much does ", "What is the price of "),
    ("How much is ", "What's the cost of "),
    ("Does ", "Is it true that "),
    ("Is the ", "Can you confirm whether the "),
    ("Are there ", "Do they have "),
    ("Which ", "What "),
    ("Where can I find ", "How do I find "),
    ("I want ", "I'd like "),
    ("I need ", "I'm searching for "),
]

_CONSTRAINT_SWAPS: list[tuple[str, str]] = [
    ("under $", "not exceeding $"),
    ("below $", "with a max budget of $"),
    ("above $", "at least $"),
    ("over $", "more than $"),
    ("Exclude ", "Do not include "),
    ("exclude ", "leave out "),
    ("without ", "excluding "),
    ("Don't include ", "Exclude "),
    ("not including ", "leaving out "),
    ("in black", "in a black colorway"),
    ("in leather", "made from leather"),
    ("for women", "for a woman"),
    ("for men", "for a man"),
    ("as a gift", "as a present"),
]


def _rephrase_query(prompt: str) -> str:
    """Create a semantically equivalent but syntactically different version of a query.

    Uses deterministic text transformations — no LLM calls.
    """
    result = prompt

    # Apply opener swap (first match only)
    for old, new in _OPENER_SWAPS:
        if result.startswith(old) or result.startswith(old.lower()):
            result = new + result[len(old):]
            break

    # Apply all matching constraint swaps
    for old, new in _CONSTRAINT_SWAPS:
        if old in result:
            result = result.replace(old, new, 1)

    # If the question ends with '?', try switching to imperative
    if result.rstrip().endswith("?") and result == prompt:
        # No opener swap matched — try wrapping as imperative
        result = result.rstrip().rstrip("?").strip()
        result = f"Please {result[0].lower()}{result[1:]}."

    # If nothing changed at all, prefix with a rephrase wrapper
    if result == prompt:
        result = f"I'd like to know: {prompt}"

    return result


def generate_probe_queries(
    parent_queries: list["Query"],
    brand_name: str,
    probes_per_category: int = 1,
) -> list["Query"]:
    """Generate rephrased probe queries for self-consistency testing.

    For each category, selects the first ``probes_per_category`` queries as
    anchors and creates one rephrased probe per anchor. Probes are compared
    directly against parent responses (no LLM judge needed).
    """
    by_category: dict[str, list["Query"]] = {}
    for q in parent_queries:
        if not q.probe_parent_id:  # skip existing probes
            by_category.setdefault(q.category, []).append(q)

    probes: list["Query"] = []
    for _cat, cat_queries in by_category.items():
        anchors = cat_queries[:probes_per_category]
        for anchor in anchors:
            rephrased = _rephrase_query(anchor.prompt)
            probes.append(Query(
                query_id=f"{anchor.query_id}_P1",
                category=anchor.category,
                prompt=rephrased,
                expected=anchor.expected,
                ground_truth_keys=anchor.ground_truth_keys,
                target_products=anchor.target_products,
                rank_order=anchor.rank_order,
                persona=anchor.persona,
                probe_parent_id=anchor.query_id,
            ))

    return probes
