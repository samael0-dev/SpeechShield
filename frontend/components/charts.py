"""Altair charts and UI cards for the Streamlit moderation dashboard."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st


def render_action_badge(action: str, score: float) -> None:
    """Render a colored moderation status pill."""
    label_map = {
        "allow": ("SAFE / ALLOW", "badge-allow"),
        "review": ("BORDERLINE / HUMAN REVIEW", "badge-review"),
        "block": ("BLOCK / HIGH TOXICITY", "badge-block"),
    }
    label, css_class = label_map.get(action, (action.upper(), "badge-review"))
    st.markdown(
        f'<span class="status-badge {css_class}">{label} &bull; {score:.1%}</span>',
        unsafe_allow_html=True,
    )


def _score_tier(score: float) -> str:
    if score >= 0.60:
        return "High"
    if score >= 0.40:
        return "Borderline"
    return "Low"


def render_category_breakdown(category_scores: dict[str, float]) -> None:
    """Display horizontal bar chart of category toxicity probabilities."""
    df = pd.DataFrame(
        [
            {
                "Category": cat.replace("_", " ").title(),
                "Probability": float(score),
                "Severity": _score_tier(float(score)),
            }
            for cat, score in category_scores.items()
        ]
    )
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("Category:N", sort="-x", title=None),
            color=alt.Color(
                "Severity:N",
                scale=alt.Scale(
                    domain=["High", "Borderline", "Low"],
                    range=["#ef4444", "#f59e0b", "#3b82f6"],
                ),
                legend=None,
            ),
            tooltip=["Category", "Severity", alt.Tooltip("Probability:Q", format=".1%")],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, width="stretch")


def render_spike_timeline(logs: list[dict[str, object]]) -> None:
    """Plot recent comment toxicity scores over sequence/time to surface hate raids."""
    if not logs:
        st.info("No moderation logs recorded yet. Analyze comments or simulate a stream to populate metrics.")
        return

    ordered = list(reversed(logs))
    df = pd.DataFrame(
        [
            {
                "Sequence": idx + 1,
                "Toxicity": float(item.get("overall_score", 0.0)),
                "Post": str(item.get("post_id", "global_feed")),
                "Platform": str(item.get("platform", "social")),
                "Action": str(item.get("action", "allow")),
            }
            for idx, item in enumerate(ordered)
        ]
    )

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Sequence:Q", title="Processed Comment Order"),
            y=alt.Y("Toxicity:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color("Post:N", title="Post / Thread"),
            tooltip=["Sequence", "Post", "Platform", "Action", alt.Tooltip("Toxicity:Q", format=".1%")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")
