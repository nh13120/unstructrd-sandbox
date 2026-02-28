"""Renders a self-contained HTML audit dashboard."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from jinja2 import Template

from src.scorer import AuditScores
from src.query_generator import (
    TABLE_STAKES, INVENTORY_OOS, MULTI_PRODUCT_COMPARISON,
    OCCASION_REASONING, BLIND_DISCOVERY, PRODUCT_COMPARISON,
    CROSS_PLATFORM_CONSISTENCY,
    TOKEN_EFFICIENCY, ADVERSARIAL, ATTRIBUTE_COMPLETENESS,
    TEMPORAL_FRESHNESS, BRAND_KNOWLEDGE, REQUIRED_ATTRIBUTES,
    LAYER_MAPPING, LAYER_LABELS, LAYER_1_DISCOVERABILITY, LAYER_2_BRAND_DNA,
)

import config as cfg

CATEGORY_LABELS = {
    TABLE_STAKES: "Table Stakes",
    INVENTORY_OOS: "Inventory / OOS",
    MULTI_PRODUCT_COMPARISON: "Multi-Product Comparison",
    OCCASION_REASONING: "Occasion Reasoning",
    BLIND_DISCOVERY: "Blind Discovery",
    PRODUCT_COMPARISON: "Product Comparison",
    CROSS_PLATFORM_CONSISTENCY: "Cross-Platform Consistency",
    TOKEN_EFFICIENCY: "Token Efficiency",
    ADVERSARIAL: "Adversarial",
    ATTRIBUTE_COMPLETENESS: "Attribute Completeness",
    TEMPORAL_FRESHNESS: "Temporal Freshness",
    BRAND_KNOWLEDGE: "Brand Knowledge",
}

ATTRIBUTE_DISPLAY_NAMES = {
    "price": "Price",
    "material": "Material / Fabric",
    "sizes": "Sizes",
    "colors": "Colors",
    "availability": "Availability",
    "care": "Care Instructions",
}

MODEL_DISPLAY_NAMES = {
    "claude": "Claude",
    "gpt": "GPT-5.2",
    "gemini": "Gemini",
    "perplexity": "Perplexity",
}

MODEL_COLORS = {
    "claude": {"bg": "rgba(217, 119, 6, 0.4)", "border": "rgba(217, 119, 6, 1)"},
    "gpt": {"bg": "rgba(16, 185, 129, 0.4)", "border": "rgba(16, 185, 129, 1)"},
    "gemini": {"bg": "rgba(59, 130, 246, 0.4)", "border": "rgba(59, 130, 246, 1)"},
    "perplexity": {"bg": "rgba(168, 85, 247, 0.4)", "border": "rgba(168, 85, 247, 1)"},
}

HTML_TEMPLATE = Template('''\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ brand_name }} — Agentic Commerce Audit</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0f1117; color: #f8fafc;
    font-family: system-ui, -apple-system, sans-serif;
    line-height: 1.6; padding: 2rem;
  }
  .container { max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 2rem; margin-bottom: 0.25rem; }
  h2 { font-size: 1.5rem; margin: 2.5rem 0 1rem; border-bottom: 1px solid #333; padding-bottom: 0.5rem; }
  h3 { font-size: 1.1rem; margin-bottom: 0.75rem; color: #94a3b8; }
  .subtitle { color: #94a3b8; font-size: 0.9rem; }
  .timestamp { color: #64748b; font-size: 0.8rem; margin-top: 0.25rem; }
  .powered { color: #6366f1; font-size: 0.85rem; margin-top: 0.5rem; }

  .card {
    background: #1e2130; border-radius: 12px; padding: 1.5rem;
    margin-bottom: 1rem;
  }
  .grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; }
  .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }
  .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1rem; }

  .score-giant {
    font-size: 4rem; font-weight: 800; text-align: center;
    margin: 1rem 0;
  }
  .score-critical { color: #ef4444; }
  .score-at-risk { color: #f59e0b; }
  .score-competitive { color: #22c55e; }

  .badge {
    display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600;
  }
  .badge-red { background: #ef444433; color: #ef4444; }
  .badge-orange { background: #f59e0b33; color: #f59e0b; }
  .badge-amber { background: #f59e0b22; color: #fbbf24; }
  .badge-yellow { background: #eab30822; color: #eab308; }
  .badge-purple { background: #a855f733; color: #a855f7; }
  .badge-gray { background: #64748b33; color: #94a3b8; }
  .badge-green { background: #22c55e33; color: #22c55e; }
  .badge-indigo { background: #6366f133; color: #6366f1; }

  .delta-positive { color: #22c55e; }
  .delta-negative { color: #ef4444; }
  .delta-neutral { color: #94a3b8; }

  .metric-label { color: #94a3b8; font-size: 0.85rem; }
  .metric-value { font-size: 1.8rem; font-weight: 700; }
  .metric-detail { color: #64748b; font-size: 0.8rem; margin-top: 0.25rem; }

  .failure-card {
    background: #1e2130; border-left: 3px solid #ef4444;
    border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem;
  }
  .failure-card .question { color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.5rem; }
  .failure-card .claim { color: #f8fafc; font-style: italic; }
  .failure-card .impact { color: #64748b; font-size: 0.8rem; margin-top: 0.5rem; }

  table {
    width: 100%; border-collapse: collapse; margin: 1rem 0;
  }
  th, td {
    text-align: left; padding: 0.75rem 1rem;
    border-bottom: 1px solid #1e2130;
  }
  th { color: #94a3b8; font-weight: 600; font-size: 0.85rem; }
  td { font-size: 0.9rem; }

  .chart-container { position: relative; height: 350px; margin: 1.5rem 0; }

  .consistency-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 4px; max-width: 300px;
  }
  .consistency-cell {
    aspect-ratio: 1; border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 600;
  }
  .cell-agree { background: #22c55e33; color: #22c55e; }
  .cell-disagree { background: #ef444433; color: #ef4444; }
  .cell-self { background: #6366f133; color: #6366f1; }

  .methodology {
    background: #1e2130; border-radius: 12px; padding: 1.5rem;
    margin-top: 2rem; color: #94a3b8; font-size: 0.85rem;
  }
  .baseline-notice {
    background: #6366f122; border: 1px solid #6366f1;
    border-radius: 8px; padding: 1rem; margin: 1rem 0;
    color: #a5b4fc; font-size: 0.9rem;
  }
  .response-text {
    max-height: 200px; overflow-y: auto; white-space: pre-wrap;
    word-wrap: break-word; font-size: 0.8rem; line-height: 1.4;
    color: #94a3b8; scrollbar-width: thin;
  }
  .response-text::-webkit-scrollbar { width: 4px; }
  .response-text::-webkit-scrollbar-thumb { background: #334155; border-radius: 2px; }

  .bullet-summary ul { padding-left: 1rem; margin: 0; }
  .bullet-summary li { margin-bottom: 0.25rem; }

  .badge-api-error {
    background: #33415533; color: #94a3b8;
    border: 1px dashed #475569;
  }
</style>
</head>
<body>
<div class="container">

  <!-- SECTION 1: HEADER -->
  <h1>{{ brand_name }} — Agentic Commerce Audit</h1>
  <div class="timestamp">Generated {{ timestamp }}</div>
  <div class="powered">Powered by Unstructrd</div>

  <!-- SECTION 2: EXECUTIVE SUMMARY -->
  <h2>Executive Summary</h2>

  <div class="card" style="text-align: center;">
    <h3>Agentic Readiness Score</h3>
    <div class="score-giant score-{{ readiness_level }}">
      {{ readiness_score }}%
    </div>
    <p style="color: #94a3b8; max-width: 600px; margin: 0 auto;">
      Here's how each AI model performed at recommending {{ brand_name }}'s products,
      and specifically what went wrong.
    </p>
  </div>

  {% if discovery_mode %}
  <div style="border: 2px solid #f59e0b; background: #f59e0b15; border-radius: 12px;
              padding: 1.25rem 1.5rem; margin: 1rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
      <span style="font-size: 1.5rem;">&#9888;</span>
      <strong style="color: #fcd34d; font-size: 1.1rem;">Discovery Mode — No Ground Truth</strong>
    </div>
    <p style="color: #fcd34d; margin: 0 0 0.5rem; font-size: 0.95rem;">
      No ground truth catalog was provided. Scores currently measure <strong>response specificity
      and internal consistency</strong>. To measure verified factual accuracy, upload a product
      catalog (CSV/XML) to test against a true baseline.
    </p>
    {% if brand_intelligence_summary %}
    <p style="color: #d4a053; margin: 0; font-size: 0.85rem;">
      Brand intelligence was pre-researched using Gemini + Google Search.
      Queries are organized into <strong>Layer 1 (Discoverability)</strong> and <strong>Layer 2 (Brand DNA)</strong>.
    </p>
    {% endif %}
  </div>
  {% endif %}

  {% if low_sample_warning %}
  <div style="border: 1px solid #f59e0b; background: #f59e0b11; border-radius: 8px;
              padding: 1rem 1.25rem; margin: 0.75rem 0;">
    <strong style="color: #fbbf24; font-size: 0.9rem;">&#9888; Low Sample Size</strong>
    <p style="color: #d4a053; margin: 0.25rem 0 0; font-size: 0.85rem;">
      Some categories have very few queries per model (n &le; {{ low_sample_threshold }}).
      For a statistically significant enterprise audit, require a minimum of 5–10 queries
      per category. Increase <code>--queries-per-category</code> accordingly.
    </p>
    <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #94a3b8;">
      {% for cat_name, model_name, count in low_sample_categories %}
      <span class="badge badge-amber" style="margin: 2px;">{{ cat_name }}: {{ model_name }} (n={{ count }})</span>
      {% endfor %}
    </div>
  </div>
  {% endif %}

  {% if readiness_breakdown %}
  <div class="card" style="margin-top: 1rem;">
    <details>
      <summary style="cursor: pointer; color: #94a3b8; font-size: 0.9rem;">
        <strong>How this score is calculated</strong> — click to expand
      </summary>
      <div style="margin-top: 0.75rem;">
        <p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.75rem;">
          The Agentic Readiness Score is a weighted average of category scores.
          Each category contributes proportionally based on its impact on real customer experience.
        </p>
        <table style="font-size: 0.82rem;">
          <thead>
            <tr>
              <th>Category</th>
              {% if discovery_mode %}<th style="text-align: center;">Layer</th>{% endif %}
              <th style="text-align: center;">Weight</th>
              <th style="text-align: center;">Score</th>
              <th style="text-align: center;">Contribution</th>
            </tr>
          </thead>
          <tbody>
            {% for item in readiness_breakdown %}
            <tr>
              <td>{{ item.label }}</td>
              {% if discovery_mode %}<td style="text-align: center;"><span class="badge {{ 'badge-indigo' if item.layer == 'L1' else 'badge-purple' }}">{{ item.layer }}</span></td>{% endif %}
              <td style="text-align: center;">{{ (item.weight * 100) | round(0) | int }}%</td>
              <td style="text-align: center;">
                <span class="{{ 'score-competitive' if item.score >= 0.7 else ('score-at-risk' if item.score >= 0.4 else 'score-critical') }}" style="font-weight: 600;">
                  {{ (item.score * 100) | round(1) }}%
                </span>
              </td>
              <td style="text-align: center; color: #64748b;">{{ (item.weighted * 100) | round(1) }}pts</td>
            </tr>
            {% endfor %}
            <tr style="border-top: 2px solid #333; font-weight: 700;">
              <td>Total</td>
              {% if discovery_mode %}<td></td>{% endif %}
              <td style="text-align: center;">100%</td>
              <td></td>
              <td style="text-align: center; color: {{ 'var(--green, #22c55e)' if readiness_score >= 70 else ('var(--amber, #f59e0b)' if readiness_score >= 40 else 'var(--red, #ef4444)') }};">
                {{ readiness_score }}%
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </details>
  </div>
  {% endif %}

  {% if brand_intelligence_summary %}
  <div class="card" style="border-left: 3px solid #6366f1;">
    <h3 style="margin: 0 0 0.5rem;">Brand Intelligence (Pre-Research)</h3>
    <p class="subtitle" style="margin-bottom: 0.75rem;">
      This intelligence was gathered via web search before generating test queries.
      All Layer 2 (Brand DNA) queries are informed by this research.
    </p>
    <pre style="background: #0f1117; padding: 1rem; border-radius: 8px; color: #94a3b8;
                font-size: 0.8rem; white-space: pre-wrap; line-height: 1.5;">{{ brand_intelligence_summary }}</pre>
  </div>
  {% endif %}

  {% if layer_scores %}
  <div class="grid-2" style="margin-top: 1rem;">
    <div class="card" style="text-align: center; border-left: 3px solid #6366f1;">
      <h3 style="margin: 0 0 0.5rem;">Layer 1: Discoverability</h3>
      <div class="metric-detail" style="margin-bottom: 0.5rem;">Can the LLM help users find products?</div>
      {% set l1 = layer_scores.get('discoverability', 0) %}
      <div class="metric-value {{ 'score-competitive' if l1 >= 0.7 else ('score-at-risk' if l1 >= 0.4 else 'score-critical') }}">
        {{ (l1 * 100) | round(1) }}%
      </div>
      <div class="metric-detail" style="margin-top: 0.5rem;">
        {% if layer_failure_summaries.get('discoverability') %}{{ layer_failure_summaries['discoverability'] }}
        {% elif l1 >= 0.7 %}AI models can generally find and recommend {{ brand_name }} products with specific details.
        {% elif l1 >= 0.4 %}AI models find some products but often miss details like prices, sizes, or give vague answers.
        {% else %}AI models struggle to surface {{ brand_name }} products — responses are mostly generic or missing key details.{% endif %}
      </div>
    </div>
    <div class="card" style="text-align: center; border-left: 3px solid #a855f7;">
      <h3 style="margin: 0 0 0.5rem;">Layer 2: Brand DNA</h3>
      <div class="metric-detail" style="margin-bottom: 0.5rem;">Does the LLM deeply understand this brand?</div>
      {% set l2 = layer_scores.get('brand_dna', 0) %}
      <div class="metric-value {{ 'score-competitive' if l2 >= 0.7 else ('score-at-risk' if l2 >= 0.4 else 'score-critical') }}">
        {{ (l2 * 100) | round(1) }}%
      </div>
      <div class="metric-detail" style="margin-top: 0.5rem;">
        {% if layer_failure_summaries.get('brand_dna') %}{{ layer_failure_summaries['brand_dna'] }}
        {% elif l2 >= 0.7 %}AI models know {{ brand_name }}'s collections, materials, and positioning well.
        {% elif l2 >= 0.4 %}AI models have surface-level brand knowledge but miss technologies, collections, or give inaccurate details.
        {% else %}AI models don't know {{ brand_name }} well — they confuse products, invent features, or give generic answers.{% endif %}
      </div>
    </div>
  </div>
  {% endif %}

  {% if is_baseline %}
  <div class="baseline-notice">
    This is a <strong>baseline audit (Agent A only)</strong>. No Unstructrd endpoint was
    provided. Agent B columns show "N/A". To see the full A/B comparison, re-run with
    <code>--unstructrd-url</code>.
  </div>
  {% endif %}

  {% if top_failures %}
  <h3>Critical Failures</h3>
  {% for f in top_failures[:3] %}
  <div class="failure-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
      <span class="badge badge-{{ failure_type_color(f.failure_type) }}">{{ f.failure_type }}</span>
      <span style="color: #64748b; font-size: 0.8rem;">
        {{ category_labels.get(f.category, f.category) }}
        {% if f.failed_models %} · Failed by: {% for mk in f.failed_models %}{{ model_display_names.get(mk, mk) }}{% if not loop.last %}, {% endif %}{% endfor %}{% endif %}
      </span>
    </div>
    <div class="question" style="margin-bottom: 0.4rem;">Q: {{ f.prompt }}</div>
    {% if f.root_cause %}
    <div style="color: #cbd5e1; font-size: 0.85rem; margin-bottom: 0.4rem;">
      {{ f.root_cause }}
    </div>
    {% endif %}
    {% if f.failure and f.failure != f.root_cause %}
    <details style="margin-top: 0.3rem;">
      <summary style="color: #64748b; font-size: 0.75rem; cursor: pointer;">Show judge's finding</summary>
      <div class="claim" style="margin-top: 0.3rem; font-size: 0.8rem;">"{{ f.failure }}"</div>
    </details>
    {% endif %}
  </div>
  {% endfor %}
  {% endif %}

  {% if not is_baseline and token_stats.estimated_monthly_savings > 0 %}
  <div class="card">
    <h3>Cost Savings Potential</h3>
    <p>Adding the Unstructrd layer would reduce AI infrastructure costs by
       <strong class="delta-positive">{{ token_stats.token_reduction_pct | round(1) }}%</strong>,
       saving ~<strong>${{ token_stats.estimated_monthly_savings | round(0) | int }}/month</strong>
       at 1M queries.</p>
  </div>
  {% endif %}

  {% if recommendations %}
  <h3>Top Recommendations</h3>
  {% for rec in recommendations %}
  <div class="card" style="border-left: 3px solid {{ '#ef4444' if rec.current_score < 40 else ('#f59e0b' if rec.current_score < 70 else '#22c55e') }};">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
      <strong style="font-size: 1rem;">#{{ rec.priority }}: {{ rec.title }}</strong>
      <span class="badge {{ 'badge-red' if rec.current_score < 40 else ('badge-orange' if rec.current_score < 70 else 'badge-green') }}">
        {{ rec.current_score }}%
      </span>
    </div>
    <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">{{ rec.description }}</p>
  </div>
  {% endfor %}
  {% endif %}

  <!-- SUSPICIOUS CLAIMS (moved up for visibility) -->
  {% if hallucination_details %}
  <h2>Suspicious Claims</h2>
  <p class="subtitle" style="margin-bottom: 1rem;">
    The AI judge flagged these claims as potentially false or fabricated.
    When AI agents confidently state wrong facts, customers act on bad information —
    creating returns, support tickets, and brand trust erosion.
  </p>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Customer asked</th>
          <th>AI claimed (flagged as suspicious)</th>
          <th>Issue</th>
          <th>Why</th>
        </tr>
      </thead>
      <tbody>
        {% for h in hallucination_details[:5] %}
        <tr>
          <td style="font-size: 0.85rem; white-space: nowrap;">{{ model_display_names.get(h.model_key, h.model_key) }}</td>
          <td style="color: #94a3b8; font-size: 0.85rem;">{{ h.prompt }}</td>
          <td style="color: #ef4444; font-size: 0.85rem;">"{{ h.suspicious_claim }}"</td>
          <td>
            <span class="badge badge-{{ failure_type_color(h.failure_type) }}">{{ h.failure_type }}</span>
          </td>
          <td style="color: #cbd5e1; font-size: 0.8rem;">{{ h.root_cause if h.root_cause else '—' }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}

  {% if comparison_examples %}
  <h2>Before / After: Structured Data Impact</h2>
  <p class="subtitle" style="margin-bottom: 1rem;">
    Side-by-side comparisons showing where structured data (Agent B) significantly outperformed web-only (Agent A).
  </p>
  {% for ex in comparison_examples %}
  <div class="card" style="margin-bottom: 1.5rem;">
    <div class="question" style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.75rem;">
      <strong>Q:</strong> {{ ex.prompt }}
      <span class="badge badge-gray" style="margin-left: 0.5rem;">{{ model_display_names.get(ex.model_key, ex.model_key) }}</span>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
      <div style="background: #ef444411; border: 1px solid #ef444433; border-radius: 8px; padding: 1rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
          <strong style="color: #ef4444; font-size: 0.85rem;">Agent A (Web Only)</strong>
          <span class="score-critical" style="font-weight: 700;">{{ (ex.agent_a_score * 100) | round(0) | int }}%</span>
        </div>
        <div class="response-text">{{ ex.agent_a_snippet }}</div>
      </div>
      <div style="background: #22c55e11; border: 1px solid #22c55e33; border-radius: 8px; padding: 1rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
          <strong style="color: #22c55e; font-size: 0.85rem;">Agent B (+ Structured Data)</strong>
          <span class="score-competitive" style="font-weight: 700;">{{ (ex.agent_b_score * 100) | round(0) | int }}%</span>
        </div>
        <div class="response-text">{{ ex.agent_b_snippet }}</div>
      </div>
    </div>
    {% if ex.failure_type and ex.failure_type.lower() != 'none' %}
    <div style="margin-top: 0.5rem;">
      <span class="badge badge-{{ failure_type_color(ex.failure_type) }}">{{ ex.failure_type }}</span>
      <span style="color: #64748b; font-size: 0.8rem; margin-left: 0.5rem;">+{{ (ex.delta * 100) | round(0) | int }}% improvement with structured data</span>
    </div>
    {% endif %}
  </div>
  {% endfor %}
  {% endif %}

  {% if product_vulnerabilities %}
  <h2>Product Vulnerability Analysis</h2>
  <p class="subtitle" style="margin-bottom: 1rem;">
    Which products are most at risk of AI agents giving customers wrong information?
  </p>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Product</th>
          <th style="text-align: center;">AI Accuracy</th>
          <th>Weakest Area</th>
          <th style="text-align: center;">Failures</th>
          <th>Failure Types</th>
        </tr>
      </thead>
      <tbody>
        {% for pv in product_vulnerabilities %}
        <tr>
          <td><strong>{{ pv.product_name }}</strong><br><span style="color: #64748b; font-size: 0.75rem;">{{ pv.product_id }} ({{ pv.num_queries }} queries)</span></td>
          <td style="text-align: center;">
            <span class="{{ 'score-competitive' if pv.avg_score >= 0.7 else ('score-at-risk' if pv.avg_score >= 0.4 else 'score-critical') }}" style="font-weight: 700; font-size: 1.1rem;">
              {{ (pv.avg_score * 100) | round(0) | int }}%
            </span>
          </td>
          <td>
            {{ category_labels.get(pv.worst_category, pv.worst_category) }}
            <span style="color: #64748b; font-size: 0.8rem;">({{ (pv.worst_category_score * 100) | round(0) | int }}%)</span>
          </td>
          <td style="text-align: center; {{ 'color: #ef4444; font-weight: 700;' if pv.failure_count > 0 else 'color: #64748b;' }}">
            {{ pv.failure_count }}
          </td>
          <td>
            {% for ft in pv.top_failure_types %}
            <span class="badge badge-{{ failure_type_color(ft) }}" style="margin-right: 4px;">{{ ft }}</span>
            {% endfor %}
            {% if not pv.top_failure_types %}—{% endif %}
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}

  <!-- CROSS-PLATFORM DISAGREEMENTS -->
  {% if consistency_examples %}
  <h2>Cross-Platform Disagreements</h2>
  <p class="subtitle" style="margin-bottom: 1rem;">Same question, different answers. Here's what each AI told your customers.</p>
  {% for ex in consistency_examples %}
  <div class="card" style="margin-bottom: 1.5rem;">
    <div class="question" style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.75rem;">
      <strong>Q:</strong> {{ ex.prompt }}
    </div>

    <!-- Per-model summaries as scannable bullets -->
    {% if ex.model_summaries %}
    <table style="margin: 0.75rem 0; font-size: 0.82rem; table-layout: fixed; width: 100%;">
      <thead>
        <tr>
          {% for m in ex.model_summaries %}
          <th style="color: #94a3b8; font-weight: 600; width: {{ (100 / ex.model_summaries|length)|round(0)|int }}%;">
            {{ model_display_names.get(m, m) }}
          </th>
          {% endfor %}
        </tr>
      </thead>
      <tbody>
        <tr>
          {% for m, summary in ex.model_summaries.items() %}
          <td class="bullet-summary" style="vertical-align: top; color: #cbd5e1; font-size: 0.78rem; line-height: 1.4; word-wrap: break-word;">
            {% if summary is string %}
            <ul>
              {% for sentence in summary.split('. ') %}
              {% if sentence.strip() %}<li>{{ sentence.strip().rstrip('.') }}</li>{% endif %}
              {% endfor %}
            </ul>
            {% elif summary is iterable %}
            <ul>
              {% for point in summary %}<li>{{ point }}</li>{% endfor %}
            </ul>
            {% endif %}
          </td>
          {% endfor %}
        </tr>
      </tbody>
    </table>
    {% endif %}

    <!-- Key Differences summary -->
    {% if ex.key_differences %}
    <div style="background: #0f1117; border-radius: 8px; padding: 0.75rem 1rem; margin-top: 0.5rem;">
      <strong style="color: #f59e0b; font-size: 0.82rem;">Key Differences</strong>
      <ul style="margin: 0.4rem 0 0 1.2rem; color: #94a3b8; font-size: 0.8rem; line-height: 1.5;">
        {% for diff in ex.key_differences %}
        <li>{{ diff }}</li>
        {% endfor %}
      </ul>
    </div>
    {% endif %}
  </div>
  {% endfor %}
  {% endif %}

  <!-- SECTION 3: KEY METRICS -->
  <h2>Key Metrics</h2>
  <div class="grid-3">
    <div class="card">
      <div class="metric-label">{{ "Response Specificity" if discovery_mode else "Attribute Completeness" }}</div>
      {% set ts_score = cat_scores.table_stakes.agent_a_mean %}
      <div class="metric-value {{ 'score-competitive' if ts_score >= 0.7 else ('score-at-risk' if ts_score >= 0.4 else 'score-critical') }}">
        {{ (ts_score * 100) | round(1) }}%
      </div>
      <div class="metric-detail">
        {% if discovery_mode %}
        {% if category_failure_summaries.get('table_stakes') and ts_score < 0.7 %}{{ category_failure_summaries['table_stakes'] }}
        {% elif ts_score >= 0.7 %}Models name real products with prices and details.
        {% elif ts_score >= 0.4 %}Models mention some products but often lack prices or specifics.
        {% else %}Models give mostly generic responses without naming products or prices.{% endif %}
        {% else %}
        {{ (ts_score * 100) | round(1) }}%
        ± {{ ((cat_scores.table_stakes.agent_a_ci[1] - cat_scores.table_stakes.agent_a_ci[0]) / 2 * 100) | round(1) }}% (95% CI)
        {% endif %}
      </div>
      {% if not is_baseline %}
      <div class="metric-detail">
        Agent B: {{ (cat_scores.table_stakes.agent_b_mean * 100) | round(1) }}%
        <span class="delta-positive">
          +{{ (cat_scores.table_stakes.delta * 100) | round(1) }}%
        </span>
      </div>
      {% endif %}
    </div>
    {% if not discovery_mode %}
    <div class="card">
      <div class="metric-label">Inventory OOS Accuracy</div>
      <div class="metric-value">
        {{ (cat_scores.inventory_oos.agent_a_mean * 100) | round(1) }}%
      </div>
      <div class="metric-detail">
        {{ (cat_scores.inventory_oos.agent_a_mean * 100) | round(1) }}%
        — {{ "PASSED" if cat_scores.inventory_oos.agent_a_mean >= 0.8 else "FAILED" }}
      </div>
      {% if not is_baseline %}
      <div class="metric-detail">
        Agent B: {{ (cat_scores.inventory_oos.agent_b_mean * 100) | round(1) }}%
        — {{ "PASSED" if cat_scores.inventory_oos.agent_b_mean >= 0.8 else "FAILED" }}
      </div>
      {% endif %}
    </div>
    {% endif %}
    <div class="card">
      <div class="metric-label">Cross-Platform Consistency</div>
      <div class="metric-value {{ 'score-competitive' if platform_consistency_a >= 0.7 else ('score-at-risk' if platform_consistency_a >= 0.4 else 'score-critical') }}">
        {{ (platform_consistency_a * 100) | round(1) }}%
      </div>
      <div class="metric-detail">
        {% if consistency_summary %}{{ consistency_summary }}
        {% elif platform_consistency_a >= 0.7 %}Models mostly agree on products and prices — customers get consistent info.
        {% elif platform_consistency_a >= 0.4 %}Models sometimes disagree — customers may get conflicting prices depending on which AI they ask.
        {% else %}Models frequently contradict each other — a customer asking Claude vs GPT gets very different answers.{% endif %}
      </div>
      {% if not is_baseline %}
      <div class="metric-detail">
        Agent B: {{ (platform_consistency_b * 100) | round(1) }}%
      </div>
      {% endif %}
    </div>
  </div>

  <!-- SECTION 4: CATEGORY SCORES -->
  <h2>Score Breakdown by Category</h2>
  <p class="subtitle" style="margin-bottom: 1rem;">Each category tests a different aspect of how AI models handle {{ brand_name }}'s products. Higher is better.</p>
  <div class="chart-container">
    <canvas id="reasoningChart"></canvas>
  </div>
  {% if discovery_mode %}
  <div class="card" style="margin-top: 0.5rem;">
    <table>
      <thead><tr><th>Category</th><th>What it measures</th><th style="text-align:center;">Score</th><th>Why this score</th></tr></thead>
      <tbody>
        {% set cat_explain = {
          "table_stakes": {"what": "Can the AI name specific products with exact prices?", "good": "Names real products with prices and details", "bad": "Gives generic responses without naming products or prices"},
          "occasion_reasoning": {"what": "Can the AI recommend products for a specific occasion or constraint?", "good": "Picks relevant products within budget/occasion", "bad": "Ignores constraints or gives irrelevant suggestions"},
          "blind_discovery": {"what": "Can the AI surface this brand's products when the customer describes what they want without naming the brand?", "good": "Recommends specific brand products with prices from a generic category description", "bad": "Recommends competitors' products or gives generic advice without mentioning the target brand"},
          "product_comparison": {"what": "Can the AI compare products within the brand side-by-side?", "good": "Detailed comparison with prices, materials, and trade-offs", "bad": "Vague comparisons without specific product details"},
          "token_efficiency": {"what": "How many tokens does the AI consume to answer product questions?", "good": "Uses fewer tokens than the median across models", "bad": "Uses significantly more tokens than other models for the same query"},
          "cross_platform_consistency": {"what": "Do different AI models agree on the same products and prices?", "good": "Models cite the same prices and products", "bad": "Different models give conflicting prices or products"},
          "brand_knowledge": {"what": "Does the AI understand the brand identity, collections, and positioning?", "good": "Knows collections, technologies, and brand values", "bad": "Confuses with competitors or invents product lines"},
          "adversarial": {"what": "Does the AI resist making up information when pressured?", "good": "Pushes back on false premises, expresses uncertainty", "bad": "Confirms false claims or invents details to sound confident"},
          "attribute_completeness": {"what": "Does the AI provide all 6 key attributes (price, material, sizes, colors, availability, care)?", "good": "Provides accurate values for all product attributes", "bad": "Misses attributes or gives wrong values (e.g. clothing sizes for bags)"},
          "temporal_freshness": {"what": "Does the AI acknowledge when its information might be outdated?", "good": "Hedges appropriately, notes data freshness limitations", "bad": "Makes definitive claims about current stock or prices"}
        } %}
        {% for cat_key, cat_label in category_labels.items() %}
        {% if cat_key in cat_scores_all and cat_key != "cross_platform_consistency" %}
        {% set cs = cat_scores_all[cat_key] %}
        {% set ex = cat_explain.get(cat_key, {}) %}
        <tr>
          <td><strong>{{ cat_label }}</strong></td>
          <td style="color: #94a3b8; font-size: 0.85rem;">{{ ex.get('what', '') }}</td>
          <td style="text-align:center;">
            <span class="{{ 'score-competitive' if cs.agent_a_mean >= 0.7 else ('score-at-risk' if cs.agent_a_mean >= 0.4 else 'score-critical') }}" style="font-weight: 700;">
              {{ (cs.agent_a_mean * 100) | round(1) }}%
            </span>
          </td>
          <td style="font-size: 0.85rem;">
            {% if cs.agent_a_mean >= 0.7 %}<span style="color: #4ade80;">{{ ex.get('good', '') }}</span>
            {% else %}<span style="color: #f87171;">{{ category_failure_summaries.get(cat_key, ex.get('bad', '')) }}</span>{% endif %}
          </td>
        </tr>
        {% endif %}
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}

  <!-- SECTION 5: PER-MODEL BREAKDOWN — INDIVIDUAL CARDS -->
  {% if per_model_scores %}
  <h2>How Each AI Model Performed</h2>
  <p class="subtitle" style="margin-bottom: 1rem;">
    What each AI platform gets right and wrong about {{ brand_name }}'s products, and why.
  </p>

  {% for mk in per_model_keys %}
  <div class="card" style="border-left: 3px solid {{ model_colors.get(mk, {}).get('border', '#64748b') }}; margin-bottom: 1.5rem;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
      <h3 style="margin: 0;">{{ model_display_names.get(mk, mk) }}</h3>
      {% if mk in error_models %}
      <span class="badge badge-api-error">API Error</span>
      {% else %}
      <div class="metric-value {{ 'score-competitive' if per_model_averages[mk] >= 0.7 else ('score-at-risk' if per_model_averages[mk] >= 0.4 else 'score-critical') }}">
        {{ (per_model_averages[mk] * 100) | round(1) }}%
      </div>
      {% endif %}
    </div>
    {% if mk not in error_models %}
    {% set model_cats = per_model_scores[mk] %}
    {% set worst_cat = None %}
    {% set worst_score = 1.0 %}
    {% for cat, sc in model_cats.items() %}
      {% if sc < worst_score and sc >= 0 %}
        {% set worst_score = sc %}
        {% set worst_cat = cat %}
      {% endif %}
    {% endfor %}
    {% set best_cat = None %}
    {% set best_score = 0.0 %}
    {% for cat, sc in model_cats.items() %}
      {% if sc > best_score %}
        {% set best_score = sc %}
        {% set best_cat = cat %}
      {% endif %}
    {% endfor %}
    <div style="margin: 0.5rem 0 0.75rem; font-size: 0.85rem; color: #94a3b8;">
      {% if best_cat %}<span style="color: #4ade80;">Best at:</span> {{ category_labels.get(best_cat, best_cat) }} ({{ (best_score * 100) | round(0) | int }}%){% endif %}
      {% if worst_cat and worst_score < 0.7 %} · <span style="color: #f87171;">Weakest:</span> {{ category_labels.get(worst_cat, worst_cat) }} ({{ (worst_score * 100) | round(0) | int }}%){% endif %}
    </div>
    {% endif %}

    {% if mk in error_models %}
    <div style="margin: 0.5rem 0;">
      <span class="badge badge-api-error">Infrastructure Failure</span>
      <p style="color: #94a3b8; font-size: 0.85rem; margin: 0.5rem 0 0;">All queries failed due to API errors (timeouts or provider overload). No valid responses were received.</p>
      <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">Infrastructure Timeout: Excluded from intelligence scoring.</div>
    </div>
    {% else %}
    <!-- Category scores for this model -->
    <table style="margin: 0.75rem 0;">
      {% for cat_key, cat_label in category_labels.items() %}
      {% if cat_key in per_model_categories and cat_key != "cross_platform_consistency" %}
      {% set score = per_model_scores[mk].get(cat_key, -1) %}
      {% if score >= 0 %}
      <tr>
        <td style="padding: 0.4rem 0.75rem;">{{ cat_label }}</td>
        <td style="text-align: right; padding: 0.4rem 0.75rem;">
          <span class="{{ 'score-competitive' if score >= 0.7 else ('score-at-risk' if score >= 0.4 else 'score-critical') }}" style="font-weight: 700;">
            {{ (score * 100) | round(1) }}%
          </span>
        </td>
      </tr>
      {% endif %}
      {% endif %}
      {% endfor %}
    </table>
    {% endif %}

    <!-- Top failures for THIS model -->
    {% set failures = per_model_failure_examples.get(mk, []) %}
    {% if failures %}
    <h4 style="color: #ef4444; font-size: 0.9rem; margin: 1rem 0 0.5rem;">What went wrong ({{ failures|length }} issue{{ 's' if failures|length > 1 }})</h4>
    {% for f in failures %}
    <div style="background: #0f1117; border-left: 2px solid #ef4444; border-radius: 6px; padding: 0.75rem; margin-bottom: 0.5rem;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
        <span class="badge badge-{{ failure_type_color(f.failure_type) }}">{{ f.failure_type }}</span>
        <span style="color: #64748b; font-size: 0.75rem;">{{ category_labels.get(f.category, f.category) }} · {{ (f.factual_accuracy * 100) | round(0) | int }}% accuracy</span>
      </div>
      <div style="color: #94a3b8; font-size: 0.8rem;">Q: {{ f.prompt }}</div>
      <div style="color: #f87171; font-size: 0.85rem; margin: 0.4rem 0;">{{ f.failure_quote }}</div>
      {% if f.response_snippet %}
      <details style="margin-top: 0.3rem;">
        <summary style="color: #64748b; font-size: 0.75rem; cursor: pointer;">Show model's response</summary>
        <div class="response-text" style="margin-top: 0.3rem; padding: 0.5rem; background: #161822; border-radius: 4px;">
          {{ f.response_snippet }}
        </div>
      </details>
      {% endif %}
    </div>
    {% endfor %}
    {% endif %}

    <!-- Token stats for THIS model -->
    {% set stats = per_model_token_stats.get(mk, {}) %}
    {% if stats %}
    <div style="display: flex; gap: 2rem; margin-top: 0.75rem; font-size: 0.8rem; color: #64748b; flex-wrap: wrap;">
      <span>Latency: {{ stats.get('avg_latency', 0) | round(1) }}s</span>
      <span>Tokens: ~{{ stats.get('avg_total_tokens', 0) | round(0) | int }}</span>
      <span>{{ stats.get('num_queries', 0) }} queries</span>
      {% set cite_rate = per_model_source_citation_rate.get(mk, 0) %}
      {% if cite_rate > 0 %}
      <span style="color: {{ '#22c55e' if cite_rate >= 0.5 else '#f59e0b' }};">Sources cited: {{ (cite_rate * 100) | round(0) | int }}%</span>
      {% endif %}
    </div>
    {% endif %}
  </div>
  {% endfor %}

  <!-- Combined chart below the cards -->
  <div class="chart-container">
    <canvas id="perModelChart"></canvas>
  </div>

  {% if per_model_failures or per_model_missing %}
  <div class="grid-2">
    {% if per_model_failures %}
    <div class="card">
      <h3>Failure Types by Model</h3>
      <table>
        <thead>
          <tr>
            <th>Failure Type</th>
            {% for mk in per_model_keys %}
            <th style="text-align: center;">{{ model_display_names.get(mk, mk) }}</th>
            {% endfor %}
          </tr>
        </thead>
        <tbody>
          {% for ft in all_failure_types %}
          <tr>
            <td><span class="badge badge-{{ failure_type_color(ft) }}">{{ ft }}</span></td>
            {% for mk in per_model_keys %}
            {% set count = per_model_failures.get(mk, {}).get(ft, 0) %}
            {% set total = per_model_query_count.get(mk, 0) %}
            <td style="text-align: center; {{ 'color: #ef4444; font-weight: 700;' if count > 0 else 'color: #64748b;' }}">
              {% if count > 0 and total > 0 %}{{ count }}/{{ total }} ({{ (count / total * 100) | round(0) | int }}%){% elif count > 0 %}{{ count }}{% else %}—{% endif %}
            </td>
            {% endfor %}
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% endif %}
    {% if per_model_missing %}
    <div class="card">
      <h3>What Each Model Didn't Mention</h3>
      <table>
        <thead>
          <tr>
            <th>Missing Detail</th>
            {% for mk in per_model_keys %}
            <th style="text-align: center;">{{ model_display_names.get(mk, mk) }}</th>
            {% endfor %}
          </tr>
        </thead>
        <tbody>
          {% for info_type in all_missing_types %}
          <tr>
            <td>{{ info_type | capitalize }}</td>
            {% for mk in per_model_keys %}
            {% set count = per_model_missing.get(mk, {}).get(info_type, 0) %}
            {% set total = per_model_query_count.get(mk, 0) %}
            <td style="text-align: center; {{ 'color: #f59e0b; font-weight: 700;' if count > 0 else 'color: #64748b;' }}">
              {% if count > 0 and total > 0 %}{{ count }}/{{ total }} ({{ (count / total * 100) | round(0) | int }}%){% elif count > 0 %}{{ count }}{% else %}—{% endif %}
            </td>
            {% endfor %}
          </tr>
          {% endfor %}
        </tbody>
      </table>
      <div class="metric-detail" style="margin-top: 0.5rem;">
        How often each model left out this detail when a customer asked about a product.
      </div>
    </div>
    {% endif %}
  </div>
  {% endif %}

  <!-- CATALOG COVERAGE / PRODUCT VISIBILITY -->
  {% if per_model_catalog_coverage %}
  {% set vis = per_model_catalog_coverage.get('_visibility', {}) %}
  <h2>Product Visibility Across AI Platforms</h2>
  <p class="subtitle" style="margin-bottom: 1rem;">
    When customers ask AI agents about {{ brand_name }}, which products get surfaced — and which are invisible?
  </p>

  <!-- Summary stats -->
  <div class="grid-4" style="margin-bottom: 1rem;">
    <div class="card" style="text-align: center;">
      <div class="metric-value">{{ vis.get('total_unique', 0) }}</div>
      <div class="metric-label">Total products surfaced</div>
    </div>
    <div class="card" style="text-align: center;">
      <div class="metric-value score-competitive">{{ vis.get('consensus_products', [])|length }}</div>
      <div class="metric-label">Known by all models</div>
      <div class="metric-detail">Surfaced by every model tested</div>
    </div>
    <div class="card" style="text-align: center;">
      <div class="metric-value score-at-risk">{{ vis.get('total_unique', 0) - vis.get('consensus_products', [])|length - vis.get('single_model_products', [])|length }}</div>
      <div class="metric-label">Partial visibility</div>
      <div class="metric-detail">Known by some models but not all</div>
    </div>
    <div class="card" style="text-align: center;">
      <div class="metric-value score-critical">{{ vis.get('single_model_products', [])|length }}</div>
      <div class="metric-label">Single-model only</div>
      <div class="metric-detail">Possible hallucination or blind spot</div>
    </div>
  </div>

  <!-- Product visibility matrix -->
  {% if vis.get('product_visibility') %}
  <div class="card">
    <h3>Product Visibility Matrix</h3>
    <div class="metric-detail" style="margin-bottom: 0.75rem;">
      Which AI models surface each product. Gaps indicate products that are invisible to specific platforms.
    </div>
    <table>
      <thead>
        <tr>
          <th>Product</th>
          {% for mk in per_model_keys %}
          <th style="text-align: center;">{{ model_display_names.get(mk, mk) }}</th>
          {% endfor %}
          <th style="text-align: center;">Coverage</th>
        </tr>
      </thead>
      <tbody>
        {% for row in vis.get('product_visibility', []) %}
        <tr>
          <td style="font-size: 0.85rem;">{{ row.name }}</td>
          {% for mk in per_model_keys %}
          <td style="text-align: center;">
            {% if mk in row.models %}
            <span style="color: #22c55e;">&#10003;</span>
            {% else %}
            <span style="color: #ef4444;">&#10007;</span>
            {% endif %}
          </td>
          {% endfor %}
          <td style="text-align: center;">
            <span class="{{ 'score-competitive' if row.coverage == per_model_keys|length else ('score-at-risk' if row.coverage > 1 else 'score-critical') }}">
              {{ row.coverage }}/{{ per_model_keys|length }}
            </span>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}

  <!-- Single-model products warning -->
  {% if vis.get('single_model_products') %}
  <div class="card" style="border-left: 3px solid #ef4444;">
    <h3>Products Only One Model Mentions</h3>
    <div class="metric-detail" style="margin-bottom: 0.5rem;">
      These products were surfaced by a single AI model only. They may be hallucinated, or other models simply can't find them.
    </div>
    {% for name, mk in vis.get('single_model_products', []) %}
    <div style="display: flex; justify-content: space-between; padding: 0.25rem 0; border-bottom: 1px solid #1e2130;">
      <span style="color: #f8fafc; font-size: 0.85rem;">{{ name }}</span>
      <span class="badge" style="background: {{ model_colors.get(mk, {}).get('bg', '#64748b33') }}; color: {{ model_colors.get(mk, {}).get('border', '#94a3b8') }};">{{ model_display_names.get(mk, mk) }} only</span>
    </div>
    {% endfor %}
  </div>
  {% endif %}
  {% endif %}
  {% endif %}

  <!-- SECTION 6: COST & SPEED  -->
  <h2>Cost & Speed</h2>
  <p class="subtitle" style="margin-bottom: 1rem;">
    Why this matters: High latency and bloated token usage lead to AI agents abandoning your brand.
    Agents prioritize fast, structured data endpoints over slow, scraped web pages.
  </p>
  {% if not is_baseline %}
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th>Agent A</th>
          <th>Agent B</th><th>Delta</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Avg Response Latency</td>
          <td>{{ token_stats.avg_latency_a | round(2) }}s</td>
          <td>{{ token_stats.avg_latency_b | round(2) }}s</td>
          <td class="delta-positive">-{{ token_stats.latency_reduction_pct | round(1) }}%</td>
        </tr>
        <tr>
          <td>Avg Total Tokens</td>
          <td>~{{ token_stats.avg_total_tokens_a | round(0) | int }}</td>
          <td>~{{ token_stats.avg_total_tokens_b | round(0) | int }}</td>
          {% set total_reduction = ((token_stats.avg_total_tokens_a - token_stats.avg_total_tokens_b) / token_stats.avg_total_tokens_a * 100) if token_stats.avg_total_tokens_a > 0 else 0 %}
          <td class="delta-positive">-{{ total_reduction | round(1) }}%</td>
        </tr>
      </tbody>
    </table>
  </div>
  {% endif %}

  {% if per_model_token_stats %}
  <div class="card" style="margin-top: 1rem;">
    <h3>Per-Model Technical Performance</h3>
    <p class="subtitle" style="margin-bottom: 0.75rem;">
      Total token benchmark: ~{{ token_stats.benchmark_median_total | round(0) | int }} median tokens per query (input + output).
      Cost uses actual API pricing per provider.
    </p>
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          {% for mk in per_model_keys %}
          <th style="text-align: center;">{{ model_display_names[mk] }}</th>
          {% endfor %}
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Avg Latency</td>
          {% for mk in per_model_keys %}
          {% set stats = per_model_token_stats.get(mk, {}) %}
          <td style="text-align: center;">{{ stats.get('avg_latency', 0) | round(2) }}s</td>
          {% endfor %}
        </tr>
        <tr>
          <td>Avg Total Tokens</td>
          {% for mk in per_model_keys %}
          {% set stats = per_model_token_stats.get(mk, {}) %}
          <td style="text-align: center;">~{{ stats.get('avg_total_tokens', 0) | round(0) | int }}</td>
          {% endfor %}
        </tr>
        <tr>
          <td>Accuracy / Second</td>
          {% for mk in per_model_keys %}
          {% set stats = per_model_token_stats.get(mk, {}) %}
          {% set aps = stats.get('accuracy_per_second', 0) %}
          <td style="text-align: center;">
            <span class="{{ 'score-competitive' if aps >= 0.04 else ('score-at-risk' if aps >= 0.02 else 'score-critical') }}">
              {{ aps }}
            </span>
          </td>
          {% endfor %}
        </tr>
        <tr>
          <td>vs Benchmark (total)</td>
          {% for mk in per_model_keys %}
          {% set stats = per_model_token_stats.get(mk, {}) %}
          {% set vs = stats.get('vs_benchmark', 0) %}
          <td style="text-align: center;">
            <span class="{{ 'score-competitive' if vs <= 1.0 else ('score-at-risk' if vs <= 2.0 else 'score-critical') }}">
              {{ vs }}x
            </span>
          </td>
          {% endfor %}
        </tr>
        <tr>
          <td>Est. Cost per Query</td>
          {% for mk in per_model_keys %}
          {% set stats = per_model_token_stats.get(mk, {}) %}
          <td style="text-align: center;">${{ stats.get('est_cost_per_query', 0) | round(4) }}</td>
          {% endfor %}
        </tr>
        <tr>
          <td>Query Count</td>
          {% for mk in per_model_keys %}
          {% set stats = per_model_token_stats.get(mk, {}) %}
          <td style="text-align: center;">{{ stats.get('num_queries', 0) | int }}</td>
          {% endfor %}
        </tr>
      </tbody>
    </table>
    <div class="baseline-notice" style="border-color: #6366f1; background: #6366f122; color: #a5b4fc; margin-top: 0.75rem; font-size: 0.85rem;">
      <strong>Note:</strong> Total tokens = input + output. Web search tools (Claude, GPT) add search results as input tokens,
      which is reflected in both the token count and cost. Cost uses actual per-provider API pricing.
    </div>
  </div>
  {% endif %}

  <!-- (Suspicious Claims section moved up, above Cross-Platform Disagreements) -->

  <!-- SECTION 7: CROSS-PLATFORM CONSISTENCY (matrix + interpretation) -->
  {% if consistency_matrix %}
  <h2>Model Agreement Matrix</h2>
  <p class="subtitle" style="margin-bottom: 1rem;">
    When two AI models answer the same question about {{ brand_name }}, do they cite the same products and prices?
    Agreement is measured by shared product names AND prices across model pairs.
  </p>
  {% set disagreement_pct = ((1 - platform_consistency_a) * 100) | round(0) | int %}
  {% if disagreement_pct > 30 %}
  <div class="baseline-notice" style="border-color: #ef4444; background: #ef444422; color: #fca5a5; margin-bottom: 1rem;">
    <strong>Brand Trust Risk:</strong> Models disagree {{ disagreement_pct }}% of the time — customers get different
    answers depending on which AI assistant they use.
  </div>
  {% endif %}

  {% if consistency_matrix.pairs_a %}
  <div class="card">
    <h3>Pairwise Agreement — {{ (platform_consistency_a * 100) | round(1) }}% overall</h3>
    <p class="metric-detail" style="margin-bottom: 0.75rem;">
      {% if consistency_summary %}{{ consistency_summary }}
      {% elif platform_consistency_a >= 0.7 %}Most model pairs agree on products and prices.
      {% elif platform_consistency_a >= 0.4 %}Some model pairs agree but there are notable conflicts.
      {% else %}Models frequently contradict each other — customers get inconsistent information.{% endif %}
    </p>
    <table>
      <thead>
        <tr>
          <th></th>
          {% for name in consistency_matrix.display_names %}
          <th style="text-align: center;">{{ name }}</th>
          {% endfor %}
        </tr>
      </thead>
      <tbody>
        {% for i in range(consistency_matrix.models | length) %}
        <tr>
          <td style="font-weight: 600;">{{ consistency_matrix.display_names[i] }}</td>
          {% for cell in consistency_matrix.agent_a[i] %}
          <td style="text-align: center;">
            <span class="{{ cell.css }}">{{ cell.label }}</span>
          </td>
          {% endfor %}
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}

  {% if not is_baseline and consistency_matrix.pairs_b %}
  <div class="card" style="margin-top: 1rem;">
    <h3>Agent B — Model Agreement — {{ (platform_consistency_b * 100) | round(1) }}% overall</h3>
    <p class="metric-detail" style="margin-bottom: 0.75rem;">With structured data</p>
    <table>
      <thead>
        <tr>
          <th></th>
          {% for name in consistency_matrix.display_names %}
          <th style="text-align: center;">{{ name }}</th>
          {% endfor %}
        </tr>
      </thead>
      <tbody>
        {% for i in range(consistency_matrix.models | length) %}
        <tr>
          <td style="font-weight: 600;">{{ consistency_matrix.display_names[i] }}</td>
          {% for cell in consistency_matrix.agent_b[i] %}
          <td style="text-align: center;">
            <span class="{{ cell.css }}">{{ cell.label }}</span>
          </td>
          {% endfor %}
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}
  {% endif %}

  <!-- SECTION 8: ATTRIBUTE ACCURACY PER MODEL -->
  {% if per_model_attribute_coverage %}
  <h2>Product Attribute Accuracy</h2>
  <p class="subtitle" style="margin-bottom: 1rem;">
    When a customer asks about a product, does the AI provide accurate details?
    This table checks 6 key attributes customers need before purchasing.
    Accuracy is verified three ways — look for the badge on each attribute row.
  </p>
  <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; font-size: 0.8rem;">
    <span style="display: inline-flex; align-items: center; gap: 0.3rem;">
      <span style="background: rgba(34,197,94,0.2); color: #22c55e; padding: 1px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: 600;">CODE</span>
      Verified against known data (e.g. price within 15% of research price)
    </span>
    <span style="display: inline-flex; align-items: center; gap: 0.3rem;">
      <span style="background: rgba(59,130,246,0.2); color: #3b82f6; padding: 1px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: 600;">CONSENSUS</span>
      Cross-model agreement (3+ models gave similar answers)
    </span>
    <span style="display: inline-flex; align-items: center; gap: 0.3rem;">
      <span style="background: rgba(245,158,11,0.2); color: #f59e0b; padding: 1px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: 600;">JUDGE</span>
      LLM judge assessment only — lower confidence
    </span>
  </div>
  {% set ns = namespace(any_low_n=false) %}
  {% for mk in attr_coverage_models %}
    {% for attr in required_attributes %}
      {% set n = per_model_attribute_coverage.get(mk, {}).get(attr, {}).get('total', 0) %}
      {% if n > 0 and n < 5 %}{% set ns.any_low_n = true %}{% endif %}
    {% endfor %}
  {% endfor %}
  {% if ns.any_low_n %}
  <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.85rem; color: #fbbf24;">
    <strong>Low sample size</strong> — Some cells are based on fewer than 5 observations (shown as <em>n=X</em>).
    Results may not be statistically meaningful. Run with <code>--queries-per-category 5</code> or higher for reliable accuracy metrics.
  </div>
  {% endif %}
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Attribute</th>
          <th style="text-align: center; font-size: 0.75rem; color: #94a3b8;">Verified by</th>
          {% for mk in attr_coverage_models %}
          <th style="text-align: center;">{{ model_display_names[mk] }}</th>
          {% endfor %}
        </tr>
      </thead>
      <tbody>
        {% for attr in required_attributes %}
        {% set vs = namespace(methods=[]) %}
        {% for mk in attr_coverage_models %}
          {% set v = per_model_attribute_coverage.get(mk, {}).get(attr, {}).get('verification', 'none') %}
          {% if v not in vs.methods %}{% set vs.methods = vs.methods + [v] %}{% endif %}
        {% endfor %}
        {% set best_method = 'code' if 'code' in vs.methods else ('consensus' if 'consensus' in vs.methods else ('judge' if 'judge' in vs.methods else 'none')) %}
        <tr>
          <td>{{ attribute_display_names.get(attr, attr) }}</td>
          <td style="text-align: center;">
            {% if best_method == 'code' %}
            <span style="background: rgba(34,197,94,0.2); color: #22c55e; padding: 1px 6px; border-radius: 4px; font-size: 0.65rem; font-weight: 600;">CODE</span>
            {% elif best_method == 'consensus' %}
            <span style="background: rgba(59,130,246,0.2); color: #3b82f6; padding: 1px 6px; border-radius: 4px; font-size: 0.65rem; font-weight: 600;">CONSENSUS</span>
            {% elif best_method == 'judge' %}
            <span style="background: rgba(245,158,11,0.2); color: #f59e0b; padding: 1px 6px; border-radius: 4px; font-size: 0.65rem; font-weight: 600;">JUDGE</span>
            {% else %}
            <span style="font-size: 0.65rem; color: #64748b;">—</span>
            {% endif %}
          </td>
          {% for mk in attr_coverage_models %}
          {% set data = per_model_attribute_coverage.get(mk, {}).get(attr, {}) %}
          {% set accurate = data.get('accurate_rate', 0) %}
          {% set specific = data.get('specific_rate', 0) %}
          {% set n = data.get('total', 0) %}
          {% set v = data.get('verification', 'none') %}
          <td style="text-align: center;">
            <span class="{{ 'score-competitive' if accurate >= 0.7 else ('score-at-risk' if accurate >= 0.4 else 'score-critical') }}" style="font-weight: 700;">
              {{ (accurate * 100) | round(0) | int }}%
            </span>
            <span style="font-size: 0.65rem; color: {{ '#f59e0b' if n < 5 else '#64748b' }};">n={{ n }}</span>
            {% if specific > accurate %}
            <div style="font-size: 0.7rem; color: #f59e0b;">{{ ((specific - accurate) * 100) | round(0) | int }}% gave wrong value</div>
            {% endif %}
            {% if data.get('inaccurate_values') %}
            <div style="font-size: 0.65rem; color: #f87171; margin-top: 2px;">
              e.g. "{{ data.inaccurate_values[0] }}"
            </div>
            {% endif %}
          </td>
          {% endfor %}
        </tr>
        {% endfor %}
        <tr style="border-top: 2px solid #333; font-weight: 700;">
          <td>Overall Accuracy</td>
          <td></td>
          {% for mk in attr_coverage_models %}
          {% set attrs = per_model_attribute_coverage.get(mk, {}) %}
          {% set avg = [] %}
          {% for attr in required_attributes %}
            {% set _ = avg.append(attrs.get(attr, {}).get('accurate_rate', 0)) %}
          {% endfor %}
          {% set overall = (avg | sum) / (avg | length) if avg else 0 %}
          {% set total_n = attrs.get(required_attributes[0], {}).get('total', 0) if required_attributes else 0 %}
          <td style="text-align: center;">
            <span class="{{ 'score-competitive' if overall >= 0.7 else ('score-at-risk' if overall >= 0.4 else 'score-critical') }}" style="font-size: 1.1rem;">
              {{ (overall * 100) | round(1) }}%
            </span>
            <span style="font-size: 0.65rem; color: {{ '#f59e0b' if total_n < 5 else '#64748b' }};">n={{ total_n }}</span>
          </td>
          {% endfor %}
        </tr>
      </tbody>
    </table>
  </div>

  <div class="grid-{{ attr_coverage_models | length }}">
  {% for mk in attr_coverage_models %}
  <div class="card" style="border-left: 3px solid {{ model_colors.get(mk, {}).get('border', '#64748b') }};">
    <h3>{{ model_display_names[mk] }}</h3>
    {% set attrs = per_model_attribute_coverage.get(mk, {}) %}
    {% for attr in required_attributes %}
    {% set data = attrs.get(attr, {}) %}
    {% set accurate = data.get('accurate_rate', 0) %}
    {% set specific = data.get('specific_rate', 0) %}
    <div style="margin-bottom: 0.5rem;">
      {% set v = data.get('verification', 'none') %}
      <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
        <span>{{ attribute_display_names.get(attr, attr) }}
          {% if v == 'code' %}<span style="background: rgba(34,197,94,0.2); color: #22c55e; padding: 0 4px; border-radius: 3px; font-size: 0.6rem; font-weight: 600; margin-left: 4px;">CODE</span>
          {% elif v == 'consensus' %}<span style="background: rgba(59,130,246,0.2); color: #3b82f6; padding: 0 4px; border-radius: 3px; font-size: 0.6rem; font-weight: 600; margin-left: 4px;">CONSENSUS</span>
          {% elif v == 'judge' %}<span style="background: rgba(245,158,11,0.2); color: #f59e0b; padding: 0 4px; border-radius: 3px; font-size: 0.6rem; font-weight: 600; margin-left: 4px;">JUDGE</span>
          {% endif %}
        </span>
        <span>
          <span class="{{ 'score-competitive' if accurate >= 0.7 else ('score-at-risk' if accurate >= 0.4 else 'score-critical') }}">
            {{ (accurate * 100) | round(0) | int }}%
          </span>
          <span style="font-size: 0.65rem; color: #64748b;">n={{ data.get('total', 0) }}</span>
        </span>
      </div>
      <div style="background: #333; border-radius: 4px; height: 6px; margin-top: 2px;">
        <div style="background: {{ 'rgba(34,197,94,0.8)' if accurate >= 0.7 else ('rgba(245,158,11,0.8)' if accurate >= 0.4 else 'rgba(239,68,68,0.8)') }}; width: {{ (specific * 100) | round(0) }}%; height: 100%; border-radius: 4px;">
          <div style="background: {{ 'rgba(34,197,94,1)' if accurate >= 0.7 else ('rgba(245,158,11,1)' if accurate >= 0.4 else 'rgba(239,68,68,1)') }}; width: {{ ((accurate / specific * 100) if specific > 0 else 0) | round(0) }}%; height: 100%; border-radius: 4px;"></div>
        </div>
      </div>
    </div>
    {% endfor %}
  </div>
  {% endfor %}
  </div>
  {% endif %}

  <!-- SECTION: RESPONSE RELIABILITY (CONFIDENCE) — discovery mode only -->
  {% if reliability_scores %}
  <h2>Response Confidence</h2>
  <p class="subtitle" style="margin-bottom: 1rem;">
    How confident can an autonomous agent be in these responses?
    Measured across {{ '4' if reliability_scores.probes_enabled else '3' }} dimensions
    without requiring ground truth data.
  </p>

  <!-- Dimension legend -->
  <div style="display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 1rem; font-size: 0.8rem; color: #94a3b8;">
    {% if reliability_scores.probes_enabled %}
    <span><strong style="color: #a78bfa;">Self-Consistency</strong> — Same question rephrased → same answer?</span>
    {% endif %}
    <span><strong style="color: #3b82f6;">Convergence</strong> — Do multiple models agree?</span>
    <span><strong style="color: #22c55e;">Source Grounding</strong> — Are verifiable sources cited?</span>
    <span><strong style="color: #f59e0b;">Specificity</strong> — Concrete details vs vague hedging?</span>
  </div>

  <!-- Overall reliability gauge -->
  <div class="card" style="text-align: center; padding: 1.5rem;">
    <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.5rem;">Overall Confidence Score</div>
    <div style="font-size: 2.5rem; font-weight: 800;" class="{{ 'score-competitive' if reliability_scores.overall_score >= 0.7 else ('score-at-risk' if reliability_scores.overall_score >= 0.4 else 'score-critical') }}">
      {{ (reliability_scores.overall_score * 100) | round(1) }}%
    </div>
    <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">
      Weights:
      {% for dim, w in reliability_scores.weights.items() %}
        {{ dim | replace('_', ' ') | title }}: {{ (w * 100) | round(0) | int }}%{{ ', ' if not loop.last }}
      {% endfor %}
    </div>
  </div>

  <!-- Per-model reliability breakdown -->
  <div class="card">
    <h3 style="margin-top: 0;">Per-Model Confidence</h3>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          {% if reliability_scores.probes_enabled %}<th style="text-align: center;">Self-Consistency</th>{% endif %}
          <th style="text-align: center;">Convergence</th>
          <th style="text-align: center;">Source Grounding</th>
          <th style="text-align: center;">Specificity</th>
          <th style="text-align: center;">Composite</th>
        </tr>
      </thead>
      <tbody>
        {% for mk in reliability_models %}
        {% set mr = reliability_scores.per_model[mk] %}
        <tr>
          <td>{{ model_display_names.get(mk, mk) }}</td>
          {% if reliability_scores.probes_enabled %}
          <td style="text-align: center;">
            <span class="{{ 'score-competitive' if mr.self_consistency.score >= 0.7 else ('score-at-risk' if mr.self_consistency.score >= 0.4 else 'score-critical') }}" style="font-weight: 700;">
              {{ (mr.self_consistency.score * 100) | round(1) }}%
            </span>
            <span style="font-size: 0.65rem; color: #64748b;">n={{ mr.self_consistency.n_samples }}</span>
          </td>
          {% endif %}
          <td style="text-align: center;">
            <span class="{{ 'score-competitive' if mr.cross_model_convergence.score >= 0.7 else ('score-at-risk' if mr.cross_model_convergence.score >= 0.4 else 'score-critical') }}" style="font-weight: 700;">
              {{ (mr.cross_model_convergence.score * 100) | round(1) }}%
            </span>
            <span style="font-size: 0.65rem; color: #64748b;">n={{ mr.cross_model_convergence.n_samples }}</span>
          </td>
          <td style="text-align: center;">
            <span class="{{ 'score-competitive' if mr.source_grounding.score >= 0.7 else ('score-at-risk' if mr.source_grounding.score >= 0.4 else 'score-critical') }}" style="font-weight: 700;">
              {{ (mr.source_grounding.score * 100) | round(1) }}%
            </span>
            <span style="font-size: 0.65rem; color: #64748b;">n={{ mr.source_grounding.n_samples }}</span>
          </td>
          <td style="text-align: center;">
            <span class="{{ 'score-competitive' if mr.specificity.score >= 0.7 else ('score-at-risk' if mr.specificity.score >= 0.4 else 'score-critical') }}" style="font-weight: 700;">
              {{ (mr.specificity.score * 100) | round(1) }}%
            </span>
            <span style="font-size: 0.65rem; color: #64748b;">n={{ mr.specificity.n_samples }}</span>
          </td>
          <td style="text-align: center;">
            <span class="{{ 'score-competitive' if mr.composite_score >= 0.7 else ('score-at-risk' if mr.composite_score >= 0.4 else 'score-critical') }}" style="font-weight: 700; font-size: 1.1rem;">
              {{ (mr.composite_score * 100) | round(1) }}%
            </span>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  {% endif %}

  <!-- SECTION: METHODOLOGY -->
  <div class="methodology">
    <h2 style="border: none; margin-top: 0; font-size: 1.1rem;">Methodology</h2>
    <p>This audit ran {{ total_queries }} queries across
       {% if discovery_mode %}9 discovery categories (5 Discoverability + 4 Brand DNA){% else %}10 test categories{% endif %}.
       All queries were sent to every selected model for full per-model comparison.
       Test models: {{ models_used }}.
       Judge: {{ judge_model }}.
       {% if discovery_mode %}
       <strong>Discovery Mode:</strong> Scores reflect response quality and specificity
       (not factual accuracy against ground truth).
       Categories tested: Table Stakes, Occasion Reasoning, Blind Discovery,
       Cross-Platform Consistency, Token Efficiency, Adversarial (Hallucination Traps),
       Attribute Completeness, Temporal Freshness, Brand Knowledge.
       {% else %}
       nDCG@3 used for ranking and discovery.
       {% endif %}
       Confidence intervals via bootstrap resampling ({{ bootstrap_iters }} iterations).
       CoV measures response consistency across queries.
       Platform consistency measured via pairwise model agreement across all queries.
       {% if raw_json_path %}Raw results: {{ raw_json_path }}{% endif %}
    </p>
  </div>

</div>

<script>
// Agentic Reasoning Bar Chart
const ctx = document.getElementById('reasoningChart').getContext('2d');
new Chart(ctx, {
  type: 'bar',
  data: {
    labels: {{ chart_labels | tojson }},
    datasets: [
      {
        label: 'Agent A',
        data: {{ chart_a_data | tojson }},
        backgroundColor: 'rgba(99, 102, 241, 0.4)',
        borderColor: 'rgba(99, 102, 241, 1)',
        borderWidth: 1,
      },
      {% if not is_baseline %}
      {
        label: 'Agent B',
        data: {{ chart_b_data | tojson }},
        backgroundColor: 'rgba(99, 102, 241, 0.8)',
        borderColor: 'rgba(99, 102, 241, 1)',
        borderWidth: 1,
      },
      {% endif %}
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#94a3b8' } },
      datalabels: {
        color: function(ctx) {
          return ctx.dataset.data[ctx.dataIndex] === 0 ? '#ef4444' : '#94a3b8';
        },
        anchor: 'end',
        align: 'end',
        font: { size: 11, weight: 'bold' },
        formatter: function(v) { return v + '%'; },
        display: function(ctx) {
          return ctx.dataset.data[ctx.dataIndex] <= 5;
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true, max: 100,
        ticks: { color: '#64748b', callback: v => v + '%' },
        grid: { color: '#1e2130' },
      },
      x: {
        ticks: { color: '#94a3b8' },
        grid: { display: false },
      },
    },
  },
});

// Per-Model Comparison Chart
{% if per_model_scores %}
const pmCtx = document.getElementById('perModelChart').getContext('2d');
new Chart(pmCtx, {
  type: 'bar',
  data: {
    labels: {{ pm_chart_labels | tojson }},
    datasets: {{ pm_chart_datasets | tojson }},
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#94a3b8' } },
      datalabels: {
        color: function(ctx) {
          return ctx.dataset.data[ctx.dataIndex] === 0 ? '#ef4444' : '#94a3b8';
        },
        anchor: 'end',
        align: 'end',
        font: { size: 10, weight: 'bold' },
        formatter: function(v) { return v + '%'; },
        display: function(ctx) {
          return ctx.dataset.data[ctx.dataIndex] <= 5;
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true, max: 100,
        ticks: { color: '#64748b', callback: v => v + '%' },
        grid: { color: '#1e2130' },
      },
      x: {
        ticks: { color: '#94a3b8' },
        grid: { display: false },
      },
    },
  },
});
{% endif %}
</script>
</body>
</html>
''')


def _failure_type_color(ft: str) -> str:
    """Map failure_type to badge color."""
    mapping = {
        "Ghost Inventory": "red",
        "Wrong Price": "orange",
        "Wrong Sizing": "amber",
        "Wrong Care": "yellow",
        "Wrong Policy": "purple",
        "Non-existent Feature": "gray",
        "Vague Response": "yellow",
        "No Products Cited": "orange",
        "Wrong Brand": "red",
        "Fabricated Product": "red",
        "Irrelevant Recommendation": "amber",
        "Incomplete Answer": "yellow",
        "Stale Data": "orange",
        "Fabricated Date": "red",
        "No Freshness Caveat": "yellow",
        "Invented Promotion": "red",
        "Fake Restock": "red",
        "Wrong Product Details": "orange",
        "Missing Technology Knowledge": "yellow",
        "Wrong Brand Values": "amber",
        "Missing Collection Info": "yellow",
        "Generic Response": "gray",
        "Wrong Competitive Positioning": "orange",
        "API Error": "gray",
        "Timeout": "gray",
        "Missing Price": "orange",
        "Missing Material": "yellow",
        "Missing Sizes": "yellow",
        "Missing Colors": "yellow",
        "Missing Availability": "yellow",
        "Missing Care": "yellow",
        "Missing Details": "orange",
        "Constraint Violation": "red",
    }
    return mapping.get(ft, "gray")


def _safe_cat(scores: AuditScores, cat: str):
    """Get category scores with safe defaults."""
    from src.scorer import CategoryScores
    return scores.category_scores.get(cat, CategoryScores(category=cat))


def _build_consistency_matrix(
    scores: AuditScores,
    models_used: list[str],
    model_display_names: dict[str, str],
) -> dict | None:
    """Build a pairwise consistency matrix for the HTML template.

    Returns:
        {
            "models": ["claude", "gpt", ...],
            "display_names": ["Claude", "GPT-5.2", ...],
            "agent_a": [[{label, css}, ...], ...],   # NxN grid
            "agent_b": [[{label, css}, ...], ...],   # NxN grid or None
            "pairs_a": [{m1, m2, rate, pct, css}, ...],  # sorted pairs
            "pairs_b": [{m1, m2, rate, pct, css}, ...],  # sorted pairs or None
        }
    """
    details = scores.consistency_details
    if not details:
        return None

    models = models_used
    if len(models) < 2:
        return None

    display = [model_display_names.get(m, m) for m in models]

    def _get_pair_rate(per_pair: dict[str, float] | None, m1: str, m2: str, fallback: float) -> float:
        """Look up the agreement rate for a specific model pair."""
        if not per_pair:
            return fallback
        key_fwd = f"{m1}_{m2}"
        key_rev = f"{m2}_{m1}"
        if key_fwd in per_pair:
            return per_pair[key_fwd]
        if key_rev in per_pair:
            return per_pair[key_rev]
        return fallback

    def _build_grid(per_pair: dict[str, float] | None, fallback_rate: float) -> list[list[dict]]:
        """Build an NxN grid of cells using per-pair agreement rates."""
        n = len(models)
        grid = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append({"label": "—", "css": "cell-self"})
                else:
                    rate = _get_pair_rate(per_pair, models[i], models[j], fallback_rate)
                    pct = round(rate * 100)
                    css = "cell-agree" if rate >= 0.5 else "cell-disagree"
                    row.append({"label": f"{pct}%", "css": css})
            grid.append(row)
        return grid

    def _build_pairs(per_pair: dict[str, float] | None, fallback_rate: float) -> list[dict]:
        """Build a sorted list of unique model pairs with agreement rates."""
        pairs = []
        n = len(models)
        for i in range(n):
            for j in range(i + 1, n):
                rate = _get_pair_rate(per_pair, models[i], models[j], fallback_rate)
                pct = round(rate * 100)
                css = "score-competitive" if rate >= 0.5 else ("score-at-risk" if rate >= 0.3 else "score-critical")
                pairs.append({
                    "m1": display[i],
                    "m2": display[j],
                    "rate": rate,
                    "pct": pct,
                    "css": css,
                })
        pairs.sort(key=lambda p: p["rate"], reverse=True)
        return pairs

    result: dict = {"models": models, "display_names": display}
    a_rates = details.get("agent_a_per_query")
    a_per_pair = details.get("agent_a_per_pair")
    b_rates = details.get("agent_b_per_query")
    b_per_pair = details.get("agent_b_per_pair")
    fallback_a = float(sum(a_rates) / len(a_rates)) if a_rates else 0.0
    fallback_b = float(sum(b_rates) / len(b_rates)) if b_rates else 0.0
    result["agent_a"] = _build_grid(a_per_pair, fallback_a) if a_rates else None
    result["agent_b"] = _build_grid(b_per_pair, fallback_b) if b_rates else None
    result["pairs_a"] = _build_pairs(a_per_pair, fallback_a) if a_rates else None
    result["pairs_b"] = _build_pairs(b_per_pair, fallback_b) if b_rates else None
    return result


def render_report(
    scores: AuditScores,
    brand_name: str,
    models_used: list[str],
    total_queries: int,
    is_baseline: bool,
    output_dir: str = "output",
    raw_json_path: str = "",
    discovery_mode: bool = False,
    brand_intel_summary: str | None = None,
    reliability_scores=None,
) -> str:
    """Render the self-contained HTML dashboard and save to disk.

    Returns the path to the saved HTML file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare chart data for Phase 2
    if discovery_mode:
        chart_categories = [
            # Layer 1: Discoverability
            (TABLE_STAKES, "Table Stakes (L1)"),
            (OCCASION_REASONING, "Occasion Reasoning (L1)"),
            (BLIND_DISCOVERY, "Blind Discovery (L1)"),
            (PRODUCT_COMPARISON, "Product Comparison (L1)"),
            (TOKEN_EFFICIENCY, "Token Efficiency (L1)"),
            # Layer 2: Brand DNA
            (BRAND_KNOWLEDGE, "Brand Knowledge (L2)"),
            (ADVERSARIAL, "Adversarial Resistance (L2)"),
            (ATTRIBUTE_COMPLETENESS, "Attribute Completeness (L2)"),
            (TEMPORAL_FRESHNESS, "Temporal Freshness (L2)"),
        ]
    else:
        chart_categories = [
            (OCCASION_REASONING, "Occasion Constraint Match"),
            (MULTI_PRODUCT_COMPARISON, "Multi-Product Ranking (nDCG@3)"),
            (BLIND_DISCOVERY, "Blind Discovery Rank (nDCG@3)"),
            (ADVERSARIAL, "Adversarial Resistance"),
            (ATTRIBUTE_COMPLETENESS, "Attribute Completeness"),
            (TEMPORAL_FRESHNESS, "Temporal Freshness"),
        ]
    chart_labels = [label for _, label in chart_categories]
    chart_a_data = [
        round(_safe_cat(scores, cat).agent_a_mean * 100, 1)
        for cat, _ in chart_categories
    ]
    chart_b_data = [
        round(_safe_cat(scores, cat).agent_b_mean * 100, 1)
        for cat, _ in chart_categories
    ]

    # Prepare category scores dict for template
    cat_scores = {
        "table_stakes": _safe_cat(scores, TABLE_STAKES),
        "inventory_oos": _safe_cat(scores, INVENTORY_OOS),
        "cross_platform": _safe_cat(scores, CROSS_PLATFORM_CONSISTENCY),
    }

    # Build consistency matrix for template
    consistency_matrix = _build_consistency_matrix(scores, models_used, MODEL_DISPLAY_NAMES)

    # Build per-model comparison data
    per_model_keys = [m for m in models_used if scores.per_model_scores and m in scores.per_model_scores]
    per_model_categories = set()
    per_model_averages: dict[str, float] = {}
    pm_chart_labels: list[str] = []
    pm_chart_datasets: list[dict] = []

    if scores.per_model_scores and per_model_keys:
        # Collect all categories that have per-model scores
        for mk in per_model_keys:
            for cat in scores.per_model_scores.get(mk, {}):
                per_model_categories.add(cat)

        # Compute per-model overall averages (exclude cross_platform_consistency)
        for mk in per_model_keys:
            model_scores = scores.per_model_scores.get(mk, {})
            vals = [v for cat, v in model_scores.items() if v >= 0 and cat != CROSS_PLATFORM_CONSISTENCY]
            per_model_averages[mk] = float(sum(vals) / len(vals)) if vals else 0.0

    # Detect models where ALL queries were errors (0 successful responses)
    error_models: set[str] = set()
    if scores.per_model_token_stats:
        for mk in per_model_keys:
            stats = scores.per_model_token_stats.get(mk, {})
            if stats.get("num_queries", 0) == 0:
                error_models.add(mk)

    # Build chart data: categories as x-axis, one dataset per model
    # Exclude error-only models from the chart (their 0% bars are misleading)
    if scores.per_model_scores and per_model_keys:
        ordered_cats = [
            (k, v) for k, v in CATEGORY_LABELS.items()
            if k in per_model_categories and k != CROSS_PLATFORM_CONSISTENCY
        ]
        pm_chart_labels = [label for _, label in ordered_cats]
        for mk in per_model_keys:
            if mk in error_models:
                continue
            model_data = scores.per_model_scores.get(mk, {})
            pm_chart_datasets.append({
                "label": MODEL_DISPLAY_NAMES.get(mk, mk),
                "data": [round(model_data.get(cat, 0) * 100, 1) for cat, _ in ordered_cats],
                "backgroundColor": MODEL_COLORS.get(mk, {"bg": "rgba(100,100,100,0.4)"})["bg"],
                "borderColor": MODEL_COLORS.get(mk, {"border": "rgba(100,100,100,1)"})["border"],
                "borderWidth": 1,
            })

    # Compute low sample size warnings
    LOW_SAMPLE_THRESHOLD = 3
    low_sample_categories: list[tuple[str, str, int]] = []
    if scores.sample_sizes:
        for cat, model_counts in scores.sample_sizes.items():
            for mk, count in model_counts.items():
                if count <= LOW_SAMPLE_THRESHOLD:
                    cat_label = CATEGORY_LABELS.get(cat, cat)
                    model_label = MODEL_DISPLAY_NAMES.get(mk, mk)
                    low_sample_categories.append((cat_label, model_label, count))
    low_sample_warning = len(low_sample_categories) > 0

    html = HTML_TEMPLATE.render(
        brand_name=brand_name,
        timestamp=timestamp,
        readiness_score=round(scores.agentic_readiness_score, 1),
        readiness_level=scores.readiness_level,
        is_baseline=is_baseline,
        discovery_mode=discovery_mode,
        top_failures=scores.top_failures,
        token_stats=scores.token_stats,
        cat_scores=cat_scores,
        cat_scores_all=scores.category_scores,
        category_labels=CATEGORY_LABELS,
        platform_consistency_a=scores.platform_consistency_a,
        platform_consistency_b=scores.platform_consistency_b,
        consistency_matrix=consistency_matrix,
        hallucination_details=scores.hallucination_details,
        chart_labels=chart_labels,
        chart_a_data=chart_a_data,
        chart_b_data=chart_b_data,
        models_used=", ".join(models_used),
        judge_model=cfg.JUDGE_MODEL,
        total_queries=total_queries,
        bootstrap_iters=f"{cfg.BOOTSTRAP_ITERATIONS:,}",
        raw_json_path=raw_json_path,
        failure_type_color=_failure_type_color,
        per_model_scores=scores.per_model_scores,
        per_model_keys=per_model_keys,
        per_model_categories=per_model_categories,
        per_model_averages=per_model_averages,
        error_models=error_models,
        model_display_names=MODEL_DISPLAY_NAMES,
        pm_chart_labels=pm_chart_labels,
        pm_chart_datasets=pm_chart_datasets,
        per_model_failures=scores.per_model_failures or {},
        per_model_missing=scores.per_model_missing or {},
        all_failure_types=sorted({
            ft for model_fts in (scores.per_model_failures or {}).values()
            for ft in model_fts
        }),
        all_missing_types=sorted({
            mt for model_mts in (scores.per_model_missing or {}).values()
            for mt in model_mts
        }),
        per_model_token_stats=scores.per_model_token_stats or {},
        per_model_query_count=scores.per_model_query_count or {},
        per_model_attribute_coverage=scores.per_model_attribute_coverage or {},
        attr_coverage_models=sorted(scores.per_model_attribute_coverage.keys()) if scores.per_model_attribute_coverage else [],
        required_attributes=REQUIRED_ATTRIBUTES,
        attribute_display_names=ATTRIBUTE_DISPLAY_NAMES,
        model_colors=MODEL_COLORS,
        comparison_examples=scores.comparison_examples or [],
        product_vulnerabilities=scores.product_vulnerabilities or [],
        recommendations=scores.recommendations or [],
        per_model_failure_examples=scores.per_model_failure_examples or {},
        consistency_examples=scores.consistency_examples or [],
        per_model_catalog_coverage=scores.per_model_catalog_coverage or {},
        brand_intelligence_summary=brand_intel_summary or "",
        layer_scores=scores.layer_scores or {},
        layer_labels=LAYER_LABELS,
        readiness_breakdown=scores.readiness_breakdown or [],
        low_sample_warning=low_sample_warning,
        low_sample_threshold=LOW_SAMPLE_THRESHOLD,
        low_sample_categories=low_sample_categories,
        category_failure_summaries=scores.category_failure_summaries or {},
        layer_failure_summaries=scores.layer_failure_summaries or {},
        consistency_summary=scores.consistency_summary or "",
        per_model_source_citation_rate=scores.per_model_source_citation_rate or {},
        reliability_scores=reliability_scores,
        reliability_models=sorted(reliability_scores.per_model.keys()) if reliability_scores else [],
        reliability_categories=sorted(reliability_scores.per_category.keys()) if reliability_scores else [],
    )

    # Save to output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{brand_name.lower().replace(' ', '_')}_audit_{file_timestamp}.html"
    out_path = out_dir / filename
    out_path.write_text(html, encoding="utf-8")

    return str(out_path)


def save_raw_json(
    data: dict,
    brand_name: str,
    output_dir: str = "output",
) -> str:
    """Save raw results as JSON for reproducibility."""
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{brand_name.lower().replace(' ', '_')}_raw_{file_timestamp}.json"
    out_path = out_dir / filename
    out_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return str(out_path)


def save_transcript(
    brand_name: str,
    brand_url: str,
    queries: list,
    agent_a_responses: dict,
    agent_b_responses: dict,
    judge_results: list,
    scores: AuditScores,
    models_used: list[str],
    is_baseline: bool = False,
    discovery_mode: bool = False,
    output_dir: str = "output",
) -> str:
    """Save a human-readable transcript of every query, response, and judge verdict."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lines: list[str] = []

    # --- Header ---
    lines.append("=" * 80)
    lines.append(f"  AUDIT TRANSCRIPT — {brand_name}")
    lines.append(f"  {brand_url}")
    lines.append(f"  Generated: {timestamp}")
    lines.append(f"  Models: {', '.join(MODEL_DISPLAY_NAMES.get(m, m) for m in models_used)}")
    lines.append(f"  Mode: {'Discovery' if discovery_mode else 'Full (CSV)'}")
    lines.append(f"  Readiness Score: {scores.agentic_readiness_score:.1f}%  ({scores.readiness_level})")
    lines.append("=" * 80)
    lines.append("")

    # --- Detect error-only models (all queries failed) ---
    transcript_error_models: set[str] = set()
    if scores.per_model_token_stats:
        for mk in models_used:
            stats = scores.per_model_token_stats.get(mk, {})
            if stats and stats.get("num_queries", 0) == 0:
                transcript_error_models.add(mk)

    # --- Per-model summary ---
    if scores.per_model_scores:
        lines.append("-" * 80)
        lines.append("  PER-MODEL SCORE SUMMARY")
        lines.append("-" * 80)
        for mk in models_used:
            model_cats = scores.per_model_scores.get(mk, {})
            if model_cats:
                if mk in transcript_error_models:
                    lines.append(f"  {MODEL_DISPLAY_NAMES.get(mk, mk):>12s}:  Error (all queries failed)")
                    for cat in model_cats:
                        lines.append(f"                 {CATEGORY_LABELS.get(cat, cat)}: Error")
                else:
                    avg = sum(model_cats.values()) / len(model_cats) if model_cats else 0
                    lines.append(f"  {MODEL_DISPLAY_NAMES.get(mk, mk):>12s}:  {avg * 100:.1f}% overall")
                    for cat, score in model_cats.items():
                        lines.append(f"                 {CATEGORY_LABELS.get(cat, cat)}: {score * 100:.1f}%")
        lines.append("")

    # --- Per-model tech stats ---
    if scores.per_model_token_stats:
        lines.append("-" * 80)
        lines.append("  PER-MODEL TECHNICAL PERFORMANCE")
        lines.append("-" * 80)
        for mk in models_used:
            stats = scores.per_model_token_stats.get(mk, {})
            if stats:
                lines.append(f"  {MODEL_DISPLAY_NAMES.get(mk, mk):>12s}:  "
                             f"latency {stats.get('avg_latency', 0):.1f}s  |  "
                             f"tokens {stats.get('avg_total_tokens', 0):.0f}  |  "
                             f"{stats.get('num_queries', 0)} queries")
        lines.append("")

    # --- Build lookup dicts ---
    jr_by_key: dict[str, dict] = {}
    for jr in judge_results:
        key = f"{jr.query_id}_{jr.model_key}" if hasattr(jr, "model_key") and jr.model_key else jr.query_id
        jr_by_key[key] = jr

    # --- Query-by-query transcript ---
    lines.append("=" * 80)
    lines.append("  FULL QUERY TRANSCRIPT")
    lines.append("=" * 80)

    for qi, q in enumerate(queries, 1):
        cat_label = CATEGORY_LABELS.get(q.category, q.category)
        lines.append("")
        lines.append(f"{'─' * 80}")
        lines.append(f"  QUERY {qi}: [{cat_label}]  {q.query_id}")
        lines.append(f"{'─' * 80}")
        lines.append(f"  PROMPT: {q.prompt}")
        lines.append("")

        # Determine which models ran this query (all queries go to all models now)
        query_models = [m for m in models_used
                       if f"{q.query_id}_{m}" in agent_a_responses]

        for mk in query_models:
            resp_key = f"{q.query_id}_{mk}"
            model_name = MODEL_DISPLAY_NAMES.get(mk, mk)

            # Agent A response
            a_resp = agent_a_responses.get(resp_key)
            lines.append(f"  ┌─ {model_name} — Agent A")
            if a_resp:
                lines.append(f"  │  Latency: {a_resp.latency_seconds:.1f}s  |  Tokens: {a_resp.total_tokens}")
                for resp_line in a_resp.text.strip().split("\n"):
                    lines.append(f"  │  {resp_line}")
            else:
                lines.append(f"  │  [NO RESPONSE]")

            # Agent B response
            if not is_baseline:
                b_resp = agent_b_responses.get(resp_key)
                lines.append(f"  ├─ {model_name} — Agent B")
                if b_resp:
                    lines.append(f"  │  Latency: {b_resp.latency_seconds:.1f}s  |  Tokens: {b_resp.total_tokens}")
                    for resp_line in b_resp.text.strip().split("\n"):
                        lines.append(f"  │  {resp_line}")
                else:
                    lines.append(f"  │  [NO RESPONSE]")

            # Judge verdict
            jr = jr_by_key.get(resp_key)
            if jr:
                fa = jr.agent_a_scores.get("factual_accuracy", "—")
                ca = jr.agent_a_scores.get("constraint_adherence", "—")
                hallu = jr.agent_a_scores.get("hallucinations", [])
                missing = jr.agent_a_scores.get("missing_info", [])
                lines.append(f"  ├─ JUDGE VERDICT")
                lines.append(f"  │  Factual Accuracy: {fa}  |  Constraint Adherence: {ca}")
                if jr.failure_type and jr.failure_type.lower() != "none":
                    lines.append(f"  │  Failure Type: {jr.failure_type}")
                if jr.most_significant_failure:
                    lines.append(f"  │  Failure: {jr.most_significant_failure}")
                if hallu:
                    lines.append(f"  │  Suspicious Claims: {hallu}")
                if missing:
                    lines.append(f"  │  Missing Info: {', '.join(missing) if isinstance(missing, list) else missing}")
            lines.append(f"  └{'─' * 40}")
            lines.append("")

    # --- Footer ---
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"  END OF TRANSCRIPT — {len(queries)} queries across {len(set(q.category for q in queries))} categories")
    lines.append("=" * 80)

    # Save
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{brand_name.lower().replace(' ', '_')}_transcript_{file_timestamp}.txt"
    out_path = out_dir / filename
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)
