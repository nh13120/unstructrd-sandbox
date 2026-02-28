#!/usr/bin/env python3
"""Gradio web UI — primary entrypoint for the Unstructrd Sandbox."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

import config as cfg
from src.data_loader import load_products, build_structured_feed
from src.query_generator import generate_query_matrix, Query
from src.agents.agent_a import AgentA
from src.agents.agent_b import AgentB
from src.agents.base_agent import AgentResponse, _get_provider_semaphore, reset_provider_semaphores
from src.judge import QueryResult, JudgeResult, run_judge_batch, make_error_judge_result, run_cross_model_comparisons, extract_catalog_products
from src.scorer import compute_scores
from src.reporter import render_report, save_raw_json, save_transcript, MODEL_DISPLAY_NAMES

logger = logging.getLogger("unstructrd_sandbox")

CONFIG_FILE = Path("config.json")
MODEL_KEYS = {"claude": "claude", "gpt": "gpt", "gemini": "gemini", "perplexity": "perplexity"}


def _load_persisted_config() -> dict:
    """Load last-used brand name and URL from config.json."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_persisted_config(brand_name: str, brand_url: str) -> None:
    """Persist last-used brand name and URL."""
    CONFIG_FILE.write_text(json.dumps({"brand_name": brand_name, "brand_url": brand_url}, indent=2))


def _has_api_key(model_key: str) -> bool:
    keys = {
        "claude": cfg.ANTHROPIC_API_KEY,
        "gpt": cfg.OPENAI_API_KEY,
        "gemini": cfg.GOOGLE_API_KEY,
        "perplexity": cfg.PERPLEXITY_API_KEY,
    }
    return bool(keys.get(model_key, "").strip())


async def _test_connection(url: str) -> str:
    """Ping Unstructrd health endpoint."""
    if not url.strip():
        return ""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{url.rstrip('/')}/health",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    return f"Connected to {url}"
                return f"Endpoint returned status {resp.status}"
    except Exception as e:
        return f"Cannot reach {url} — {e}"


async def _run_agent_queries(
    tasks: list[tuple],
    label: str,
    models: list[str],
    log_callback,
) -> dict[str, AgentResponse]:
    """Run all agent queries with per-provider concurrency and circuit breaker.

    Args:
        tasks: List of (key, agent, prompt) tuples.
        label: Agent label ("A" or "B") for logging.
        models: List of model keys for failure tracking.
        log_callback: Function to call with progress messages.
    """
    responses: dict[str, AgentResponse] = {}
    provider_failures: dict[str, int] = {m: 0 for m in models}
    dead_providers: set[str] = set()
    completed = 0
    total = len(tasks)

    async def _run_one(key: str, agent, prompt: str) -> None:
        nonlocal completed
        mk = agent.model_key

        def _skip(reason: str) -> None:
            nonlocal completed
            responses[key] = AgentResponse(
                text=f"ERROR: {mk} skipped ({reason})", prompt_tokens=0,
                completion_tokens=0, total_tokens=0, latency_seconds=0.0,
                model=mk, agent=label,
            )
            completed += 1
            log_callback(f"  Agent {label}: {completed}/{total}")

        if mk in dead_providers:
            _skip("too many failures")
            return

        sem = _get_provider_semaphore(mk)
        async with sem:
            if mk in dead_providers:
                _skip("too many failures")
                return

            try:
                resp = await agent._call_with_retry(prompt)
                responses[key] = resp
                provider_failures[mk] = 0
            except Exception as e:
                responses[key] = AgentResponse(
                    text=f"ERROR: {e}", prompt_tokens=0, completion_tokens=0,
                    total_tokens=0, latency_seconds=0.0, model=mk, agent=label,
                )
                provider_failures[mk] = provider_failures.get(mk, 0) + 1
                if provider_failures[mk] >= 3:
                    dead_providers.add(mk)
                    log_callback(f"  ⚠ {mk} failed 3x in a row — skipping remaining")
            completed += 1
            log_callback(f"  Agent {label}: {completed}/{total}")

    await asyncio.gather(*[_run_one(k, a, p) for k, a, p in tasks])
    return responses


async def _run_audit_async(
    brand_name: str,
    brand_url: str,
    csv_path: str | None,
    unstructrd_url: str,
    models: list[str],
    queries_per_category: int,
    log_callback,
    seed: int | None = None,
) -> dict:
    """Core audit pipeline — runs asynchronously."""
    # Clear stale semaphores from any previous asyncio.run() invocation
    reset_provider_semaphores()

    discovery_mode = not csv_path
    structured_data = ""

    # 1. Load CSV (or skip in discovery mode)
    if discovery_mode:
        log_callback("Discovery mode — no CSV provided")
        products = None
    else:
        log_callback("Loading ground truth CSV...")
        products = load_products(csv_path)
        log_callback(f"  Loaded {len(products)} products")
        # Auto-build structured feed for Agent B
        structured_data = build_structured_feed(products)
        log_callback(f"  Built structured feed for Agent B ({len(products)} products)")

    is_baseline = not unstructrd_url.strip() and not structured_data

    # 2. Health check + fetch feed from live endpoint
    if unstructrd_url.strip():
        log_callback("Testing Unstructrd endpoint...")
        msg = await _test_connection(unstructrd_url)
        log_callback(f"  {msg}")
        if "Cannot reach" in msg:
            raise ConnectionError(msg)

        # Fetch structured feed if we don't already have one (from CSV)
        if not structured_data:
            feed_url = f"{unstructrd_url.rstrip('/')}/feed"
            log_callback(f"  Fetching structured feed from {feed_url}...")
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=30)) as feed_resp:
                        if feed_resp.status == 200:
                            structured_data = await feed_resp.text()
                            log_callback(f"  Fetched structured feed ({len(structured_data)} bytes)")
                        else:
                            log_callback(f"  Warning: Could not fetch feed (status {feed_resp.status}), running baseline")
            except Exception as fe:
                log_callback(f"  Warning: Feed fetch failed ({fe}), running baseline")

        # Recalculate baseline status after potential feed fetch
        is_baseline = not structured_data

    # 3. Generate queries (with brand research in discovery mode)
    brand_intel = None
    if discovery_mode:
        log_callback("Researching brand intelligence...")
        try:
            from src.brand_researcher import research_brand
            brand_intel = await research_brand(brand_name, brand_url)
            # Surface data quality warnings before the is_populated check
            for w in brand_intel.quality_warnings:
                log_callback(f"  ⚠ {w}")
            if brand_intel.is_populated:
                log_callback(f"  Brand research complete: {len(brand_intel.valid_flagship_products)} products identified")
                log_callback(f"  Categories: {', '.join(brand_intel.valid_categories[:5])}")
            else:
                log_callback("  Brand research incomplete — using generic templates")
                brand_intel = None
        except Exception as e:
            log_callback(f"  Brand research failed ({e}) — using generic templates")
            brand_intel = None

    log_callback("Generating query matrix...")
    if discovery_mode:
        from src.query_generator import generate_discovery_matrix
        queries = generate_discovery_matrix(
            brand_name, brand_url, queries_per_category,
            brand_intel=brand_intel, seed=seed,
        )
        mode_desc = "brand-aware" if brand_intel else "generic"
        log_callback(f"  Generated {len(queries)} queries ({mode_desc} discovery mode)")
    else:
        queries = generate_query_matrix(products, brand_name, queries_per_category, seed=seed)
        log_callback(f"  Generated {len(queries)} queries")

    # Generate consistency probes (discovery mode only, need ≥3 queries/cat)
    probe_queries: list[Query] = []
    if discovery_mode and queries_per_category >= 3:
        from src.query_generator import generate_probe_queries
        probe_queries = generate_probe_queries(queries, brand_name, probes_per_category=1)
        log_callback(f"  Generated {len(probe_queries)} self-consistency probes")
    elif discovery_mode:
        log_callback("  Skipping probes (need ≥3 queries/category)")

    # 4. Agent A
    log_callback("Running Agent A queries...")
    from src.query_generator import BLIND_DISCOVERY
    a_kwargs = {"brand_name": brand_name, "brand_url": brand_url}
    tasks_a = []
    for q in queries:
        for mk in [m for m in models if _has_api_key(m)]:
            agent = AgentA(model_key=mk, **a_kwargs)
            if q.category == BLIND_DISCOVERY:
                agent.blind = True
            tasks_a.append((f"{q.query_id}_{mk}", agent, q.prompt))
    agent_a_responses = await _run_agent_queries(tasks_a, "A", models, log_callback)

    # 4b. Run self-consistency probes
    probe_responses: dict[str, AgentResponse] = {}
    if probe_queries:
        log_callback("Running self-consistency probes...")
        tasks_p = []
        for q in probe_queries:
            for mk in [m for m in models if _has_api_key(m)]:
                tasks_p.append((f"{q.query_id}_{mk}", AgentA(model_key=mk, **a_kwargs), q.prompt))
        probe_responses = await _run_agent_queries(tasks_p, "A", models, log_callback)

    # 5. Agent B
    agent_b_responses: dict[str, AgentResponse] = {}
    if not is_baseline:
        log_callback("Running Agent B queries...")
        b_kwargs = {"brand_name": brand_name, "brand_url": brand_url, "structured_data": structured_data}
        tasks_b = []
        for q in queries:
            for mk in [m for m in models if _has_api_key(m)]:
                agent = AgentB(model_key=mk, **b_kwargs)
                if q.category == BLIND_DISCOVERY:
                    agent.blind = True
                tasks_b.append((f"{q.query_id}_{mk}", agent, q.prompt))
        agent_b_responses = await _run_agent_queries(tasks_b, "B", models, log_callback)
    else:
        log_callback("Skipping Agent B (baseline mode)")

    # 6. Judge — judge EVERY (query, model) pair
    log_callback("Running judge evaluation...")

    # Build cross-reference map for discovery mode
    cross_ref_map: dict[str, dict[str, str]] = {}
    if discovery_mode:
        for q in queries:
            for mk in models:
                key = f"{q.query_id}_{mk}"
                resp = agent_a_responses.get(key)
                if resp and not resp.text.startswith("ERROR"):
                    cross_ref_map.setdefault(q.query_id, {})[mk] = resp.text[:500]

    query_results = []
    error_judge_results: list[JudgeResult] = []
    for q in queries:
        q_models = [m for m in models if _has_api_key(m)]
        for mk in q_models:
            key = f"{q.query_id}_{mk}"
            a_resp = agent_a_responses.get(key)
            b_resp = agent_b_responses.get(key)
            if not a_resp:
                continue
            # Skip judging error responses — create a pre-built result instead
            if a_resp.text.startswith("ERROR"):
                error_judge_results.append(make_error_judge_result(
                    query=q, error_text=a_resp.text, model_key=mk,
                ))
                continue
            # Build cross-ref context excluding this model's own response
            cross_ref = ""
            if discovery_mode:
                other_resps = {
                    m: s for m, s in cross_ref_map.get(q.query_id, {}).items()
                    if m != mk
                }
                if other_resps:
                    cross_ref = "\n".join(
                        f"[{MODEL_DISPLAY_NAMES.get(m, m)}]: {s[:300]}"
                        for m, s in other_resps.items()
                    )
            query_results.append(QueryResult(
                query=q, agent_a_response=a_resp, agent_b_response=b_resp,
                model_key=mk,
                cross_reference_context=cross_ref,
            ))

    n_skipped = len(error_judge_results)
    if n_skipped:
        log_callback(f"  Skipping judge for {n_skipped} error responses")

    def _judge_progress(completed, total):
        log_callback(f"  Judge: {completed}/{total}")

    judge_results = await run_judge_batch(
        query_results, progress_callback=_judge_progress,
        discovery_mode=discovery_mode, brand_name=brand_name, brand_url=brand_url,
        baseline_mode=is_baseline,
    )

    # Merge error results back in
    judge_results = list(judge_results) + error_judge_results

    # 7. Score
    log_callback("Computing statistics...")

    # Build per-query cross-model response dicts for platform consistency
    # All queries now run on all models, so consistency is measured across everything
    consistency_a: dict[str, dict[str, AgentResponse]] = {}
    consistency_b: dict[str, dict[str, AgentResponse]] = {}
    for q in queries:
        for mk in models:
            key = f"{q.query_id}_{mk}"
            if key in agent_a_responses:
                consistency_a.setdefault(q.query_id, {})[mk] = agent_a_responses[key]
            if key in agent_b_responses:
                consistency_b.setdefault(q.query_id, {})[mk] = agent_b_responses[key]

    # Run cross-model comparison judge — scale with query count
    max_cross_comparisons = min(5, max(2, len(queries) // 3))
    cross_model_comparisons = []
    if consistency_a and len(models) >= 2:
        log_callback(f"Running cross-model comparisons ({max_cross_comparisons})...")
        cross_model_comparisons = await run_cross_model_comparisons(
            consistency_responses=consistency_a,
            brand_name=brand_name,
            brand_url=brand_url,
            queries=queries,
            max_comparisons=max_cross_comparisons,
        )

    # Extract clean product names via LLM for catalog coverage
    catalog_products = await extract_catalog_products(agent_a_responses, brand_name)

    scores = compute_scores(
        judge_results, agent_a_responses, agent_b_responses,
        consistency_a_responses=consistency_a or None,
        consistency_b_responses=consistency_b or None,
        queries=queries,
        discovery_mode=discovery_mode,
        cross_model_comparisons=cross_model_comparisons or None,
        brand_name=brand_name,
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
            brand_url=brand_url,
            probes_enabled=bool(probe_queries),
        )
        scores.reliability_scores = reliability_scores
        log_callback(f"  Reliability score: {reliability_scores.overall_score * 100:.1f}%")

    # 8. Report
    log_callback("Rendering report...")
    raw_data = {
        "brand_name": brand_name,
        "is_baseline": is_baseline,
        "discovery_mode": discovery_mode,
        "queries": [{"query_id": q.query_id, "category": q.category, "prompt": q.prompt} for q in queries],
        "agent_a_responses": {k: asdict(v) for k, v in agent_a_responses.items()},
        "agent_b_responses": {k: asdict(v) for k, v in agent_b_responses.items()},
        "judge_results": [asdict(jr) for jr in judge_results],
    }
    raw_json_path = save_raw_json(raw_data, brand_name)
    report_path = render_report(
        scores=scores, brand_name=brand_name, models_used=models,
        total_queries=len(queries), is_baseline=is_baseline,
        raw_json_path=raw_json_path, discovery_mode=discovery_mode,
        brand_intel_summary=brand_intel.summary() if brand_intel else None,
        reliability_scores=reliability_scores,
    )
    transcript_path = save_transcript(
        brand_name=brand_name, brand_url=brand_url, queries=queries,
        agent_a_responses=agent_a_responses, agent_b_responses=agent_b_responses,
        judge_results=judge_results, scores=scores, models_used=models,
        is_baseline=is_baseline, discovery_mode=discovery_mode,
    )

    _save_persisted_config(brand_name, brand_url)
    log_callback(f"Report saved to {report_path}")
    log_callback(f"Transcript saved to {transcript_path}")

    return {
        "report_path": report_path,
        "raw_json_path": raw_json_path,
        "transcript_path": transcript_path,
        "readiness_score": scores.agentic_readiness_score,
        "readiness_level": scores.readiness_level,
        "top_failures": scores.top_failures[:3],
    }


def build_ui():
    """Build and return the Gradio Blocks interface."""
    saved = _load_persisted_config()

    with gr.Blocks(
        title="Unstructrd Sandbox",
        theme=gr.themes.Base(
            primary_hue="indigo",
            neutral_hue="slate",
        ),
    ) as demo:
        gr.Markdown("# Unstructrd Sandbox\n### Agentic Commerce Audit Tool")

        # --- State ---
        log_state = gr.State([])

        # --- PANEL 1: BRAND SETUP ---
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Brand Setup")
                brand_name = gr.Textbox(
                    label="Brand Name",
                    placeholder="e.g. Keslo",
                    value=saved.get("brand_name", ""),
                )
                brand_url = gr.Textbox(
                    label="Brand Website URL",
                    placeholder="e.g. https://keslo.store",
                    value=saved.get("brand_url", ""),
                )
                csv_upload = gr.File(
                    label="Ground Truth CSV (optional — omit for Discovery Mode)",
                    file_types=[".csv"],
                    type="filepath",
                )
                discovery_notice = gr.Markdown(
                    "**Discovery Mode:** No CSV uploaded. The audit will evaluate response "
                    "quality and specificity rather than factual accuracy. Upload a CSV for "
                    "the full 8-category audit.",
                    visible=True,
                )
                csv_preview = gr.Dataframe(
                    label="CSV Preview (first 3 rows)",
                    visible=False,
                    interactive=False,
                )

                with gr.Accordion("CSV Format Guide", open=False):
                    gr.Markdown(
                        "See `ground_truth/template.csv` for full column reference.\n\n"
                        "**Required:** product_id, product_name, price, currency, material, "
                        "care_instructions, return_policy, shipping_standard_days, "
                        "shipping_express_days, occasion_tags, warmth_score, waterproof_score, "
                        "office_appropriate_score, non_existent_colors\n\n"
                        "**Stock columns:** `stock_{SIZE}_{COLOR}` = true/false"
                    )
                    template_btn = gr.Button("Download Blank Template", size="sm")
                    template_download = gr.File(label="Template", visible=False)

                gr.Markdown("---\n### Unstructrd Endpoint")
                unstructrd_url = gr.Textbox(
                    label="Unstructrd URL",
                    placeholder="e.g. https://unstructrd.keslo.store",
                    info="Leave blank to run Agent A baseline only",
                )
                connection_status = gr.Textbox(label="Connection Status", interactive=False, visible=False)
                test_btn = gr.Button("Test Connection", visible=True)

            # --- PANEL 2: RUN SETTINGS ---
            with gr.Column(scale=1):
                gr.Markdown("## Run Settings")
                model_checks = gr.CheckboxGroup(
                    label="Models",
                    choices=["claude", "gpt", "gemini", "perplexity"],
                    value=["claude", "gpt", "gemini", "perplexity"],
                )
                qpc_slider = gr.Slider(
                    label="Queries per Category",
                    minimum=3, maximum=30, step=1, value=10,
                )
                cost_label = gr.Markdown("*Standard — ~$10, ~10 min*")
                run_btn = gr.Button("RUN AUDIT", variant="primary", size="lg")

                # --- PANEL 3: PROGRESS + RESULTS ---
                gr.Markdown("## Progress")
                status_text = gr.Textbox(label="Status", interactive=False, lines=1)
                log_box = gr.Textbox(label="Live Log", interactive=False, lines=10, max_lines=20)

                gr.Markdown("## Results")
                result_text = gr.Markdown("")
                download_btn = gr.File(label="Download Report", visible=False)

        # --- Event Handlers ---
        def preview_csv(file_path):
            if not file_path:
                return gr.update(visible=False)
            import pandas as pd
            try:
                df = pd.read_csv(file_path, comment="#", nrows=3)
                return gr.update(value=df, visible=True)
            except Exception:
                return gr.update(visible=False)

        def update_cost_label(qpc):
            if qpc <= 5:
                return "*Quick demo — ~$2, ~3 min*"
            elif qpc <= 15:
                return "*Standard — ~$10, ~10 min*"
            else:
                return "*Statistically significant — ~$30, ~30 min*"

        def update_run_btn(brand_n, brand_u, csv_f, unstr_u):
            has_required = bool(brand_n and brand_u)
            if csv_f:
                label = "RUN BASELINE (Agent A only)" if not unstr_u else "RUN AUDIT"
            else:
                label = "RUN DISCOVERY (no CSV)" if not unstr_u else "RUN DISCOVERY + UNSTRUCTRD"
            return gr.update(value=label, interactive=has_required)

        def toggle_discovery_notice(csv_f):
            return gr.update(visible=not bool(csv_f))

        def test_connection_sync(url):
            if not url.strip():
                return gr.update(value="", visible=False)
            result = asyncio.run(_test_connection(url))
            return gr.update(value=result, visible=True)

        def serve_template():
            template_path = Path("ground_truth/template.csv")
            if template_path.exists():
                return gr.update(value=str(template_path), visible=True)
            return gr.update(visible=False)

        def validate_csv(file_path):
            """Validate that CSV has required columns."""
            if not file_path:
                return ""
            import csv
            required = {"product_id", "product_name", "price", "currency", "material",
                        "care_instructions", "return_policy", "shipping_standard_days",
                        "shipping_express_days", "occasion_tags"}
            try:
                with open(file_path) as f:
                    lines = [l for l in f if not l.startswith("#")]
                reader = csv.DictReader(lines)
                headers = set(reader.fieldnames or [])
                missing = required - headers
                if missing:
                    return f"Missing columns: {', '.join(sorted(missing))}"
                return ""
            except Exception as e:
                return f"CSV error: {e}"

        def run_audit_sync(brand_n, brand_u, csv_path, unstr_u, models, qpc):
            """Run the audit in a synchronous wrapper for Gradio."""
            logs = []

            def log_cb(msg):
                logs.append(msg)

            try:
                result = asyncio.run(_run_audit_async(
                    brand_name=brand_n,
                    brand_url=brand_u,
                    csv_path=csv_path or None,
                    unstructrd_url=unstr_u or "",
                    models=models,
                    queries_per_category=qpc,
                    log_callback=log_cb,
                    seed=None,
                ))

                log_text = "\n".join(logs)
                score = result["readiness_score"]
                level = result["readiness_level"]
                level_emoji = {"critical": "🔴", "at-risk": "🟡", "competitive": "🟢"}[level]

                failures_md = ""
                for f in result.get("top_failures", []):
                    failures_md += f"\n- **[{f['failure_type']}]** {f['failure'][:100]}"

                result_md = (
                    f"### {level_emoji} YOUR BRAND IS {score:.1f}% READY FOR AGENTIC COMMERCE\n\n"
                    f"**Top failures:**{failures_md}\n"
                )

                return (
                    "Audit complete!",
                    log_text,
                    result_md,
                    gr.update(value=result["report_path"], visible=True),
                )

            except Exception as e:
                tb = traceback.format_exc()
                Path("output").mkdir(exist_ok=True)
                Path("output/error_log.txt").write_text(tb)
                log_text = "\n".join(logs) + f"\n\nERROR: {e}"
                return (
                    f"Error: {e}",
                    log_text,
                    f"### Error\n\n```\n{e}\n```\n\nFull traceback saved to output/error_log.txt",
                    gr.update(visible=False),
                )

        # Wire up events
        csv_upload.change(preview_csv, inputs=[csv_upload], outputs=[csv_preview])
        csv_upload.change(toggle_discovery_notice, inputs=[csv_upload], outputs=[discovery_notice])
        qpc_slider.change(update_cost_label, inputs=[qpc_slider], outputs=[cost_label])

        for inp in [brand_name, brand_url, csv_upload, unstructrd_url]:
            inp.change(
                update_run_btn,
                inputs=[brand_name, brand_url, csv_upload, unstructrd_url],
                outputs=[run_btn],
            )

        test_btn.click(test_connection_sync, inputs=[unstructrd_url], outputs=[connection_status])
        template_btn.click(serve_template, outputs=[template_download])

        run_btn.click(
            run_audit_sync,
            inputs=[brand_name, brand_url, csv_upload, unstructrd_url, model_checks, qpc_slider],
            outputs=[status_text, log_box, result_text, download_btn],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=True)
