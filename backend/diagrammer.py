"""
diagrammer.py — Diagram generation from LLM-structured data for Lexora.

When the RAG answer contains structured data (tables, comparisons, timelines,
statistics), this module renders it as a diagram/chart image.

Flow:
  1. rag.py calls answer_question() with diagram=True.
  2. rag.py first asks Groq to extract structured data as JSON from the context.
  3. diagrammer.detect_and_render(structured_data) picks the right chart type
     and saves a PNG to DIAGRAM_OUTPUT_DIR.
  4. The path is returned alongside the text answer.

Supported diagram types:
  - bar_chart:   categorical comparisons
  - line_chart:  time-series / sequential data
  - pie_chart:   proportions / percentages
  - table:       structured multi-column data
  - timeline:    dated events
"""

import os
import json
import logging
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("[Diagrammer] matplotlib not available — diagrams disabled.")


from config import DIAGRAM_OUTPUT_DIR


# ── Helpers ──────────────────────────────────────────────────────────────────

def _output_path(prefix: str, query: str) -> str:
    """Generate a deterministic output filename."""
    slug = hashlib.md5(query.encode()).hexdigest()[:8]
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{prefix}_{ts}_{slug}.png"
    return os.path.join(DIAGRAM_OUTPUT_DIR, fname)


def _apply_lexora_style(fig, ax=None):
    """Apply consistent dark Lexora theme."""
    fig.patch.set_facecolor("#0f1117")
    if ax:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#ffffff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")


# ── Chart renderers ──────────────────────────────────────────────────────────

def render_bar_chart(data: dict, query: str) -> str:
    """
    data format:
      {"title": "...", "labels": [...], "values": [...], "unit": "..."}
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    labels = data.get("labels", [])
    values = data.get("values", [])
    title  = data.get("title", "Comparison")
    unit   = data.get("unit", "")

    if not labels or not values:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_lexora_style(fig, ax)

    colors = ["#58a6ff" if i % 2 == 0 else "#3fb950" for i in range(len(labels))]
    bars = ax.bar(labels, values, color=colors, edgecolor="#30363d", linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val}{' ' + unit if unit else ''}",
            ha="center", va="bottom", color="#c9d1d9", fontsize=9
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel(unit)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = _output_path("bar", query)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"[Diagrammer] Bar chart saved: {path}")
    return path


def render_line_chart(data: dict, query: str) -> str:
    """
    data format:
      {"title": "...", "x_label": "...", "y_label": "...",
       "series": [{"name": "...", "x": [...], "y": [...]}]}
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    series = data.get("series", [])
    title  = data.get("title", "Trend")

    if not series:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_lexora_style(fig, ax)

    line_colors = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657"]
    for i, s in enumerate(series):
        color = line_colors[i % len(line_colors)]
        ax.plot(s["x"], s["y"], marker="o", label=s.get("name", f"Series {i+1}"),
                color=color, linewidth=2, markersize=5)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(data.get("x_label", ""))
    ax.set_ylabel(data.get("y_label", ""))
    ax.legend(facecolor="#1a1d27", edgecolor="#30363d", labelcolor="#c9d1d9")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = _output_path("line", query)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"[Diagrammer] Line chart saved: {path}")
    return path


def render_pie_chart(data: dict, query: str) -> str:
    """
    data format:
      {"title": "...", "labels": [...], "values": [...]}
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    labels = data.get("labels", [])
    values = data.get("values", [])
    title  = data.get("title", "Distribution")

    if not labels or not values:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    _apply_lexora_style(fig, ax)

    palette = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657",
               "#79c0ff", "#56d364", "#ff7b72", "#bc8cff", "#ffb74d"]
    colors = [palette[i % len(palette)] for i in range(len(labels))]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        pctdistance=0.82,
        startangle=140,
        wedgeprops={"edgecolor": "#0f1117", "linewidth": 2},
    )
    for t in texts:
        t.set_color("#c9d1d9")
    for at in autotexts:
        at.set_color("#ffffff")
        at.set_fontsize(9)

    ax.set_title(title, fontsize=14, fontweight="bold", color="#ffffff", pad=15)
    plt.tight_layout()

    path = _output_path("pie", query)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"[Diagrammer] Pie chart saved: {path}")
    return path


def render_table(data: dict, query: str) -> str:
    """
    data format:
      {"title": "...", "headers": [...], "rows": [[...], [...]]}
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    headers = data.get("headers", [])
    rows    = data.get("rows", [])
    title   = data.get("title", "Data Table")

    if not headers or not rows:
        return None

    n_cols = len(headers)
    n_rows = len(rows)
    fig_h  = max(3, 0.5 * n_rows + 1.5)

    fig, ax = plt.subplots(figsize=(min(2 * n_cols + 2, 14), fig_h))
    _apply_lexora_style(fig, ax)
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)

    # Style header row
    for col in range(n_cols):
        cell = tbl[0, col]
        cell.set_facecolor("#1f6feb")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for row in range(1, n_rows + 1):
        bg = "#1a1d27" if row % 2 == 0 else "#21262d"
        for col in range(n_cols):
            tbl[row, col].set_facecolor(bg)
            tbl[row, col].set_text_props(color="#c9d1d9")
            tbl[row, col].set_edgecolor("#30363d")

    ax.set_title(title, fontsize=13, fontweight="bold", color="#ffffff",
                 pad=12, loc="left")
    plt.tight_layout()

    path = _output_path("table", query)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"[Diagrammer] Table saved: {path}")
    return path


def render_timeline(data: dict, query: str) -> str:
    """
    data format:
      {"title": "...", "events": [{"date": "...", "label": "..."}]}
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    events = data.get("events", [])
    title  = data.get("title", "Timeline")

    if not events:
        return None

    labels = [e.get("label", "") for e in events]
    dates  = [e.get("date", "") for e in events]
    n      = len(events)

    fig, ax = plt.subplots(figsize=(max(10, n * 1.5), 4))
    _apply_lexora_style(fig, ax)
    ax.axis("off")

    # Draw horizontal spine
    ax.axhline(0.5, xmin=0.05, xmax=0.95, color="#58a6ff", linewidth=2)

    positions = [0.1 + 0.8 * i / max(n - 1, 1) for i in range(n)]
    for i, (pos, label, date) in enumerate(zip(positions, labels, dates)):
        y_dot   = 0.5
        y_text  = 0.75 if i % 2 == 0 else 0.25
        y_conn  = 0.58 if i % 2 == 0 else 0.42

        ax.plot(pos, y_dot, "o", color="#f78166", markersize=10, zorder=5)
        ax.plot([pos, pos], [y_dot, y_conn], color="#58a6ff", linewidth=1, linestyle="--")
        ax.text(pos, y_text, label, ha="center", va="center",
                fontsize=8, color="#c9d1d9", wrap=True,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262d",
                          edgecolor="#30363d", alpha=0.9))
        ax.text(pos, 0.5 - (0.1 if i % 2 == 0 else -0.1),
                date, ha="center", va="center", fontsize=7,
                color="#8b949e")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#ffffff", pad=10)
    plt.tight_layout()

    path = _output_path("timeline", query)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"[Diagrammer] Timeline saved: {path}")
    return path


# ── Dispatcher ───────────────────────────────────────────────────────────────

RENDERERS = {
    "bar_chart":  render_bar_chart,
    "line_chart": render_line_chart,
    "pie_chart":  render_pie_chart,
    "table":      render_table,
    "timeline":   render_timeline,
}


def detect_and_render(structured: dict, query: str) -> str | None:
    """
    Given a structured dict from Groq (with a "type" key), render the right diagram.

    Returns the output file path, or None if rendering fails or type is unknown.
    """
    dtype = structured.get("type", "none").lower()
    if dtype == "none" or dtype not in RENDERERS:
        logger.info(f"[Diagrammer] No diagram for type: '{dtype}'")
        return None

    try:
        return RENDERERS[dtype](structured, query)
    except Exception as e:
        logger.error(f"[Diagrammer] Render failed for type '{dtype}': {e}")
        return None
