#!/usr/bin/env python3
"""CLI entrypoint for the Unstructrd Sandbox audit."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from colorama import Fore, Style, init as colorama_init
from tqdm import tqdm

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
from src.data_loader import load_products, build_structured_feed
from src.query_generator import generate_query_matrix, ALL_CATEGORIES, Query
from src.agents.agent_a import AgentA
from src.agents.agent_b import AgentB
from src.agents.base_agent import AgentResponse, reset_provider_semaphores
from src.judge import QueryResult, JudgeResult, run_judge_batch, make_error_judge_result, run_cross_model_comparisons, extract_catalog_products
from src.scorer import compute_scores
from src.reporter import render_report, save_raw_json, save_transcript, MODEL_DISPLAY_NAMES

logger = logging.getLogger("unstructrd_sandbox")
colorama_init()

MODEL_KEYS = {
    "claude": "claude",
    "gpt": "gpt",
    "gemini": "gemini",
    "perplexity": "perplexity",
}


def _has_api_key(model_key: str) -> bool:
    """Check if the API key for a model is configured."""
    keys = {
        "claude": cfg.ANTHROPIC_API_KEY,
        "gpt": cfg.OPENAI_API_KEY,
        "gemini": cfg.GOOGLE_API_KEY,
        "perplexity": cfg.PERPLEXITY_API_KEY,
    }
    return bool(keys.get(model_key, "").strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unstructrd Sandbox — LLM Ecommerce Audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--brand-name", required=True, help="Display name for report header")
    parser.add_argument("--brand-url", required=True, help="Brand website URL")
    parser.add_argument("--ground-truth", required=False, default=None,
                        help="Path to ground truth CSV (omit for discovery mode)")
    parser.add_argument("--unstructrd-url", default="", help="Live Unstructrd endpoint URL (fetches feed)")
    parser.add_argument("--structured-data", default="", help="Path to structured product JSON feed for Agent B")
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_KEYS.keys()),
        choices=list(MODEL_KEYS.keys()),
        help="Models to test (default: all)",
    )
    parser.add_argument("--queries-per-category", type=int, default=cfg.QUERIES_PER_CATEGORY)
    parser.add_argument(
        "--probe-consistency", action=argparse.BooleanOptionalAction,
        default=True,
        help="Run self-consistency probes for reliability scoring (adds ~36 API calls). "
             "Use --no-probe-consistency to disable.",
    )
    parser.add_argument(
        "--probes-per-category", type=int, default=1,
        help="Number of probe queries per category for self-consistency testing (default: 1)",
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for query generation (default: random each run, set for reproducibility)")
    return parser.parse_args()


async def run_agent_queries(
    agent_cls,
    agent_kwargs: dict,
    queries: list,
    models: list[str],
    label: str,
    pbar_desc: str,
) -> dict[str, AgentResponse]:
    """Run all queries for an agent across specified models."""
    from src.agents.base_agent import _get_provider_semaphore

    responses: dict[str, AgentResponse] = {}
    tasks = []

    # Track consecutive failures per provider — skip the rest after 3 in a row
    provider_failures: dict[str, int] = {m: 0 for m in models}
    dead_providers: set[str] = set()

    from src.query_generator import BLIND_DISCOVERY
    for q in queries:
        query_models = [m for m in models if _has_api_key(m)]
        for model_key in query_models:
            agent = agent_cls(model_key=model_key, **agent_kwargs)
            if q.category == BLIND_DISCOVERY:
                agent.blind = True
            tasks.append((f"{q.query_id}_{model_key}", agent, q.prompt))

    pbar = tqdm(total=len(tasks), desc=pbar_desc, ncols=80)

    def _skip(key: str, mk: str, reason: str):
        responses[key] = AgentResponse(
            text=f"ERROR: {mk} skipped ({reason})",
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            latency_seconds=0.0, model=mk, agent=label,
        )
        pbar.update(1)

    async def _run_one(key: str, agent, prompt: str):
        mk = agent.model_key
        # Early check — skip if provider already died before we even queue
        if mk in dead_providers:
            _skip(key, mk, "too many failures")
            return

        # Acquire the per-provider semaphore here (not inside agent.run)
        # so we can re-check dead_providers after waiting in the queue.
        sem = _get_provider_semaphore(mk)
        async with sem:
            # Re-check after waiting — provider may have died while queued
            if mk in dead_providers:
                _skip(key, mk, "too many failures")
                return

            try:
                resp = await agent._call_with_retry(prompt)
                responses[key] = resp
                provider_failures[mk] = 0  # reset on success
            except Exception as e:
                logger.warning(f"Agent {label} failed on {key}: {e}")
                responses[key] = AgentResponse(
                    text=f"ERROR: {e}",
                    prompt_tokens=0, completion_tokens=0, total_tokens=0,
                    latency_seconds=0.0, model=mk, agent=label,
                )
                provider_failures[mk] = provider_failures.get(mk, 0) + 1
                if provider_failures[mk] >= 3:
                    dead_providers.add(mk)
                    logger.warning(f"{mk}: 3 consecutive failures — skipping remaining queries")
                    print(f"\n  ⚠ {mk} failed 3x in a row — skipping remaining {mk} queries")
            finally:
                pbar.update(1)

    await asyncio.gather(*[_run_one(k, a, p) for k, a, p in tasks])
    pbar.close()
    return responses


async def main() -> None:
    args = parse_args()
    reset_provider_semaphores()
    discovery_mode = args.ground_truth is None

    # Determine structured data source for Agent B
    structured_data = ""
    if args.structured_data:
        sd_path = Path(args.structured_data)
        if not sd_path.exists():
            print(f"\n{Fore.RED}ERROR: Structured data file not found: {sd_path}{Style.RESET_ALL}")
            sys.exit(1)
        structured_data = sd_path.read_text(encoding="utf-8")
    # CSV mode: auto-build structured feed from products (set after CSV load below)

    is_baseline = not args.unstructrd_url and not structured_data

    if discovery_mode and is_baseline:
        mode_label = "Discovery (Agent A only)"
    elif discovery_mode:
        mode_label = "Discovery A/B Test"
    elif is_baseline:
        mode_label = "Baseline (Agent A only)"
    else:
        mode_label = "A/B Test"

    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  Unstructrd Sandbox — Agentic Commerce Audit{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"  Brand:    {args.brand_name}")
    print(f"  URL:      {args.brand_url}")
    print(f"  Models:   {', '.join(args.models)}")
    print(f"  Queries:  {args.queries_per_category} per category")
    print(f"  Mode:     {mode_label}")
    if args.unstructrd_url:
        print(f"  Unstructrd: {args.unstructrd_url}")
    if structured_data:
        print(f"  Structured data: {'from file' if args.structured_data else 'from CSV'}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

    # 1. Load and validate CSV (or skip in discovery mode)
    brand_intel = None
    if discovery_mode:
        print(f"{Fore.YELLOW}[1/9]{Style.RESET_ALL} Discovery mode — no CSV provided")
        products = None

        # Pre-audit brand research
        print(f"{Fore.YELLOW}[2/9]{Style.RESET_ALL} Researching brand intelligence...")
        try:
            from src.brand_researcher import research_brand
            brand_intel = await research_brand(args.brand_name, args.brand_url)
            # Surface quality warnings before the populated check
            for warning in brand_intel.quality_warnings:
                print(f"  {Fore.YELLOW}Warning: {warning}{Style.RESET_ALL}")
            if brand_intel.is_populated:
                print(f"  {Fore.GREEN}Brand research complete:{Style.RESET_ALL}")
                for line in brand_intel.summary().split("\n"):
                    print(f"    {line}")
            else:
                print(f"  {Fore.YELLOW}Brand research incomplete — falling back to generic templates{Style.RESET_ALL}")
                brand_intel = None
        except Exception as e:
            print(f"  {Fore.YELLOW}Brand research failed ({e}) — using generic templates{Style.RESET_ALL}")
            brand_intel = None
    else:
        print(f"{Fore.YELLOW}[1/8]{Style.RESET_ALL} Loading ground truth CSV...")
        try:
            products = load_products(args.ground_truth)
            print(f"  Loaded {len(products)} products")
        except (FileNotFoundError, ValueError) as e:
            print(f"\n{Fore.RED}ERROR: {e}{Style.RESET_ALL}")
            sys.exit(1)
        # Auto-build structured feed for Agent B from CSV products
        if not structured_data and products:
            structured_data = build_structured_feed(products)
            is_baseline = False  # CSV mode always runs Agent B
            print(f"  Built structured feed for Agent B ({len(products)} products)")

    # 2. Health check Unstructrd endpoint + fetch feed
    if args.unstructrd_url:
        print(f"{Fore.YELLOW}[2/8]{Style.RESET_ALL} Pinging Unstructrd endpoint...")
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{args.unstructrd_url.rstrip('/')}/health", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        print(f"  {Fore.GREEN}Connected to {args.unstructrd_url}{Style.RESET_ALL}")
                    else:
                        print(f"\n{Fore.RED}ERROR: Unstructrd returned status {resp.status}{Style.RESET_ALL}")
                        sys.exit(1)

                # Fetch structured feed if no --structured-data was provided
                if not structured_data:
                    feed_url = f"{args.unstructrd_url.rstrip('/')}/feed"
                    print(f"  Fetching structured feed from {feed_url}...")
                    try:
                        async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=30)) as feed_resp:
                            if feed_resp.status == 200:
                                structured_data = await feed_resp.text()
                                print(f"  {Fore.GREEN}Fetched structured feed ({len(structured_data)} bytes){Style.RESET_ALL}")
                            else:
                                print(f"  {Fore.YELLOW}Warning: Could not fetch feed (status {feed_resp.status}), running baseline{Style.RESET_ALL}")
                    except Exception as fe:
                        print(f"  {Fore.YELLOW}Warning: Feed fetch failed ({fe}), running baseline{Style.RESET_ALL}")
        except Exception as e:
            print(f"\n{Fore.RED}ERROR: Cannot reach {args.unstructrd_url} — {e}{Style.RESET_ALL}")
            sys.exit(1)

        # Recalculate baseline status after potential feed fetch
        is_baseline = not args.unstructrd_url and not structured_data
    else:
        print(f"{Fore.YELLOW}[2/8]{Style.RESET_ALL} Skipping Unstructrd health check (baseline mode)")

    # 3. Generate query matrix
    print(f"{Fore.YELLOW}[3/8]{Style.RESET_ALL} Generating query matrix...")
    if discovery_mode:
        from src.query_generator import generate_discovery_matrix, DISCOVERY_CATEGORIES
        queries = generate_discovery_matrix(
            args.brand_name, args.brand_url, args.queries_per_category,
            brand_intel=brand_intel, seed=args.seed,
        )
        mode_desc = "brand-aware" if brand_intel else "generic"
        print(f"  Generated {len(queries)} queries across {len(DISCOVERY_CATEGORIES)} categories ({mode_desc} discovery mode)")
    else:
        queries = generate_query_matrix(products, args.brand_name, args.queries_per_category, seed=args.seed)
        print(f"  Generated {len(queries)} queries across {len(ALL_CATEGORIES)} categories")

    # 3b. Generate consistency probes (discovery mode only)
    # Probes need multiple samples per category for statistical power.
    # At 1 query/category they add ~40 API calls for a weak signal — skip.
    probe_queries: list[Query] = []
    probes_enabled = args.probe_consistency and args.queries_per_category >= 3
    if discovery_mode and probes_enabled:
        from src.query_generator import generate_probe_queries
        probe_queries = generate_probe_queries(
            queries, args.brand_name,
            probes_per_category=args.probes_per_category,
        )
        print(f"  Generated {len(probe_queries)} self-consistency probes")
    elif discovery_mode and args.probe_consistency and args.queries_per_category < 3:
        print(f"  Skipping probes (need ≥3 queries/category for statistical power)")

    # 4. Run Agent A
    print(f"\n{Fore.YELLOW}[4/8]{Style.RESET_ALL} Running Agent A queries...")
    agent_a_kwargs = {"brand_name": args.brand_name, "brand_url": args.brand_url}
    agent_a_responses = await run_agent_queries(
        AgentA, agent_a_kwargs, queries, args.models, "A", "Agent A",
    )

    # 4b. Run self-consistency probes
    probe_responses: dict[str, AgentResponse] = {}
    if probe_queries:
        print(f"\n{Fore.YELLOW}[4b/8]{Style.RESET_ALL} Running self-consistency probes...")
        probe_responses = await run_agent_queries(
            AgentA, agent_a_kwargs, probe_queries, args.models, "A", "Probes",
        )

    # 5. Run Agent B (if not baseline)
    agent_b_responses: dict[str, AgentResponse] = {}
    if not is_baseline:
        print(f"\n{Fore.YELLOW}[5/8]{Style.RESET_ALL} Running Agent B queries (with structured data)...")
        agent_b_kwargs = {
            "brand_name": args.brand_name,
            "brand_url": args.brand_url,
            "structured_data": structured_data,
        }
        agent_b_responses = await run_agent_queries(
            AgentB, agent_b_kwargs, queries, args.models, "B", "Agent B",
        )
    else:
        print(f"{Fore.YELLOW}[5/8]{Style.RESET_ALL} Skipping Agent B (baseline mode)")

    # 6. Batch judge evaluation — judge EVERY (query, model) pair
    print(f"\n{Fore.YELLOW}[6/8]{Style.RESET_ALL} Running judge evaluation...")

    # Build cross-reference map for discovery mode: query_id -> {model_key: response_snippet}
    cross_ref_map: dict[str, dict[str, str]] = {}
    if discovery_mode:
        for q in queries:
            for mk in args.models:
                key = f"{q.query_id}_{mk}"
                resp = agent_a_responses.get(key)
                if resp and not resp.text.startswith("ERROR"):
                    cross_ref_map.setdefault(q.query_id, {})[mk] = resp.text[:500]

    query_results = []
    error_judge_results: list[JudgeResult] = []
    for q in queries:
        # All queries were sent to all selected models
        query_models = [m for m in args.models if _has_api_key(m)]

        for model_key in query_models:
            key = f"{q.query_id}_{model_key}"
            a_resp = agent_a_responses.get(key)
            b_resp = agent_b_responses.get(key)
            if not a_resp:
                continue
            # Skip judging error responses — create a pre-built result instead
            if a_resp.text.startswith("ERROR"):
                error_judge_results.append(make_error_judge_result(
                    query=q, error_text=a_resp.text, model_key=model_key,
                ))
                continue
            # Build cross-ref context excluding this model's own response
            cross_ref = ""
            if discovery_mode:
                other_resps = {
                    m: s for m, s in cross_ref_map.get(q.query_id, {}).items()
                    if m != model_key
                }
                if other_resps:
                    cross_ref = "\n".join(
                        f"[{MODEL_DISPLAY_NAMES.get(m, m)}]: {s[:300]}"
                        for m, s in other_resps.items()
                    )
            query_results.append(QueryResult(
                query=q, agent_a_response=a_resp, agent_b_response=b_resp,
                model_key=model_key,
                cross_reference_context=cross_ref,
            ))

    n_skipped = len(error_judge_results)
    if n_skipped:
        print(f"  Skipping judge for {n_skipped} error responses")

    judge_pbar = tqdm(total=len(query_results), desc="Judge", ncols=80)

    def _judge_progress(completed: int, total: int):
        judge_pbar.n = completed
        judge_pbar.refresh()

    judge_results = await run_judge_batch(
        query_results,
        progress_callback=_judge_progress,
        discovery_mode=discovery_mode,
        brand_name=args.brand_name,
        brand_url=args.brand_url,
        baseline_mode=is_baseline,
    )
    judge_pbar.close()

    # Merge error results back in
    judge_results = list(judge_results) + error_judge_results

    # 7. Compute scores
    print(f"\n{Fore.YELLOW}[7/8]{Style.RESET_ALL} Computing statistics...")

    # Build per-query cross-model response dicts for platform consistency
    # All queries now run on all models, so consistency is measured across everything
    consistency_a: dict[str, dict[str, AgentResponse]] = {}
    consistency_b: dict[str, dict[str, AgentResponse]] = {}
    for q in queries:
        for mk in args.models:
            key = f"{q.query_id}_{mk}"
            if key in agent_a_responses:
                consistency_a.setdefault(q.query_id, {})[mk] = agent_a_responses[key]
            if key in agent_b_responses:
                consistency_b.setdefault(q.query_id, {})[mk] = agent_b_responses[key]

    # Run cross-model comparison judge (produces summaries + key differences)
    # Scale comparisons with query count — 1 query/cat doesn't need 5 comparisons
    max_cross_comparisons = min(5, max(2, len(queries) // 3))
    cross_model_comparisons = []
    if consistency_a and len(args.models) >= 2:
        print(f"  Running cross-model comparisons ({max_cross_comparisons})...")
        cross_model_comparisons = await run_cross_model_comparisons(
            consistency_responses=consistency_a,
            brand_name=args.brand_name,
            brand_url=args.brand_url,
            queries=queries,
            max_comparisons=max_cross_comparisons,
        )

    # Extract clean product names via LLM for catalog coverage
    catalog_products = await extract_catalog_products(agent_a_responses, args.brand_name)

    scores = compute_scores(
        judge_results, agent_a_responses, agent_b_responses,
        consistency_a_responses=consistency_a or None,
        consistency_b_responses=consistency_b or None,
        queries=queries,
        discovery_mode=discovery_mode,
        cross_model_comparisons=cross_model_comparisons or None,
        brand_name=args.brand_name,
        catalog_products=catalog_products,
    )

    # 7b. Compute reliability scores (discovery mode)
    reliability_scores = None
    if discovery_mode:
        from src.reliability import compute_reliability
        reliability_scores = compute_reliability(
            judge_results=judge_results,
            agent_a_responses=agent_a_responses,
            consistency_responses=consistency_a or None,
            queries=queries,
            probe_queries=probe_queries or None,
            probe_responses=probe_responses or None,
            brand_url=args.brand_url,
            probes_enabled=bool(probe_queries),
        )
        scores.reliability_scores = reliability_scores
        print(f"  Reliability score: {reliability_scores.overall_score * 100:.1f}%")

    # 8. Render report + save raw JSON
    print(f"{Fore.YELLOW}[8/8]{Style.RESET_ALL} Rendering report...")

    # Build raw data for JSON
    raw_data = {
        "brand_name": args.brand_name,
        "brand_url": args.brand_url,
        "unstructrd_url": args.unstructrd_url,
        "models": args.models,
        "queries_per_category": args.queries_per_category,
        "is_baseline": is_baseline,
        "discovery_mode": discovery_mode,
        "queries": [
            {
                "query_id": q.query_id,
                "category": q.category,
                "prompt": q.prompt,
                "expected": q.expected,
            }
            for q in queries
        ],
        "agent_a_responses": {
            k: asdict(v) for k, v in agent_a_responses.items()
        },
        "agent_b_responses": {
            k: asdict(v) for k, v in agent_b_responses.items()
        },
        "judge_results": [asdict(jr) for jr in judge_results],
        "scores": {
            "readiness_score": scores.agentic_readiness_score,
            "readiness_level": scores.readiness_level,
            "token_stats": asdict(scores.token_stats),
            "category_scores": {
                k: {
                    "agent_a_mean": v.agent_a_mean,
                    "agent_b_mean": v.agent_b_mean,
                    "delta": v.delta,
                }
                for k, v in scores.category_scores.items()
            },
            "per_model_scores": scores.per_model_scores,
            "per_model_attribute_coverage": scores.per_model_attribute_coverage,
            "per_model_source_citation_rate": scores.per_model_source_citation_rate,
            "layer_scores": scores.layer_scores,
        },
        "brand_intelligence": brand_intel.summary() if brand_intel else None,
        "reliability": {
            "overall_score": reliability_scores.overall_score,
            "probes_enabled": reliability_scores.probes_enabled,
            "weights": reliability_scores.weights,
            "per_model": {
                mk: {
                    "composite": mr.composite_score,
                    "self_consistency": mr.self_consistency.score,
                    "convergence": mr.cross_model_convergence.score,
                    "source_grounding": mr.source_grounding.score,
                    "specificity": mr.specificity.score,
                }
                for mk, mr in reliability_scores.per_model.items()
            },
        } if reliability_scores else None,
    }

    raw_json_path = save_raw_json(raw_data, args.brand_name, args.output_dir)
    report_path = render_report(
        scores=scores,
        brand_name=args.brand_name,
        models_used=args.models,
        total_queries=len(queries),
        is_baseline=is_baseline,
        output_dir=args.output_dir,
        raw_json_path=raw_json_path,
        discovery_mode=discovery_mode,
        brand_intel_summary=brand_intel.summary() if brand_intel else None,
        reliability_scores=reliability_scores,
    )
    transcript_path = save_transcript(
        brand_name=args.brand_name,
        brand_url=args.brand_url,
        queries=queries,
        agent_a_responses=agent_a_responses,
        agent_b_responses=agent_b_responses,
        judge_results=judge_results,
        scores=scores,
        models_used=args.models,
        is_baseline=is_baseline,
        discovery_mode=discovery_mode,
        output_dir=args.output_dir,
    )

    # Summary
    color = {
        "critical": Fore.RED,
        "at-risk": Fore.YELLOW,
        "competitive": Fore.GREEN,
    }[scores.readiness_level]

    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"  {color}AGENTIC READINESS: {scores.agentic_readiness_score:.1f}%{Style.RESET_ALL}")
    if reliability_scores:
        print(f"  RELIABILITY SCORE: {reliability_scores.overall_score * 100:.1f}%")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

    if scores.top_failures:
        print(f"\n  {Fore.RED}Top failures:{Style.RESET_ALL}")
        for f in scores.top_failures[:3]:
            print(f"    • [{f['failure_type']}] {f['failure'][:80]}")

    print(f"\n  {Fore.GREEN}Report saved to {report_path}{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}Raw JSON saved to {raw_json_path}{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}Transcript saved to {transcript_path}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Audit cancelled.{Style.RESET_ALL}")
        sys.exit(130)
    except Exception as e:
        # Log full traceback to file
        Path("output").mkdir(exist_ok=True)
        tb = traceback.format_exc()
        Path("output/error_log.txt").write_text(tb)
        print(f"\n{Fore.RED}ERROR: {e}{Style.RESET_ALL}")
        print(f"  Full traceback saved to output/error_log.txt")
        sys.exit(1)
