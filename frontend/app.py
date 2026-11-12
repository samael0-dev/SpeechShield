"""Streamlit moderation dashboard for the Social Media Toxicity Detector."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components.charts import (
    render_action_badge,
    render_category_breakdown,
    render_spike_timeline,
)
from src.data_pipeline.stream_ingest import MockSocialStream
from src.model.classifier import ToxicityClassifier

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
STYLE_PATH = Path(__file__).parent / "assets" / "style.css"

st.set_page_config(
    page_title="Social Media Toxicity Detector",
    page_icon="🛡️",
    layout="wide",
)

if STYLE_PATH.exists():
    st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_local_classifier() -> ToxicityClassifier:
    """Fallback local classifier used only when FastAPI backend is offline."""
    return ToxicityClassifier()


def api_request(method: str, path: str, **kwargs: object) -> tuple[bool, object]:
    """Call FastAPI backend and return `(success, payload_or_error)`."""
    url = f"{API_BASE_URL}{path}"
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.request(method, url, **kwargs)
            resp.raise_for_status()
            return True, resp.json()
    except httpx.HTTPError as exc:
        return False, str(exc)


def main() -> None:
    st.title("🛡️ Social Media Toxicity Detector")
    st.caption(
        "Real-time social media moderation with emoji/slang cleaning, fast-pass filtering, "
        "BERTweet/RoBERTa scoring, and 60%–80% borderline human review."
    )

    health_ok, health_data = api_request("GET", "/health")
    with st.sidebar:
        st.header("System Status")
        if health_ok and isinstance(health_data, dict):
            st.success(f"FastAPI Connected ({API_BASE_URL})")
            backends = health_data.get("backends", {})
            st.json(backends)
        else:
            st.warning(f"FastAPI offline ({API_BASE_URL}). Using local classifier fallback for single/batch analysis.")
            st.code("uvicorn src.api.main:app --reload", language="bash")

        st.divider()
        st.subheader("Moderation Policy")
        st.markdown(
            "- **< 60%**: Auto-Allow (`allow`)\n"
            "- **60% – 80%**: Borderline Queue (`review`)\n"
            "- **> 80% / Severe Threat**: Auto-Block (`block`)"
        )

    tab_single, tab_batch, tab_review, tab_spikes = st.tabs(
        [
            "🔍 Live Text Analyzer",
            "📦 Batch / CSV & Stream",
            "🧑‍⚖️ Borderline Review Queue",
            "📈 Hate-Raid & Spike Monitor",
        ]
    )

    with tab_single:
        col_in, col_meta = st.columns([3, 1])
        with col_in:
            user_text = st.text_area(
                "Paste a social media post, tweet, or comment",
                placeholder="e.g., Shut up you @ss clown 🤡 #IdentityHate",
                height=120,
            )
        with col_meta:
            platform = st.selectbox("Platform", ["x", "reddit", "youtube", "discord", "other"])
            post_id = st.text_input("Post / Thread ID", value="post_101")
            author = st.text_input("Author Handle", value="social_user")

        if st.button("Analyze Comment", type="primary", width="stretch"):
            if not user_text.strip():
                st.warning("Please enter text to analyze.")
            else:
                payload = {
                    "text": user_text,
                    "platform": platform,
                    "post_id": post_id,
                    "author_handle": author,
                }
                ok, data = api_request("POST", "/predict", json=payload)
                if not ok:
                    res = get_local_classifier().predict(user_text)
                    data = {
                        "raw_text": res.raw_text,
                        "cleaned_text": res.cleaned_text,
                        "overall_score": res.overall_score,
                        "action": res.action,
                        "severity": res.severity,
                        "flagged_categories": res.flagged_categories,
                        "category_scores": res.category_scores,
                        "model_used": res.model_used,
                        "fast_passed": res.fast_passed,
                        "signals": res.signals,
                    }

                assert isinstance(data, dict)
                c1, c2 = st.columns([1.2, 1.8])
                with c1:
                    st.subheader("Moderation Decision")
                    render_action_badge(str(data["action"]), float(data["overall_score"]))
                    st.markdown(f"**Model Used:** `{data['model_used']}`")
                    st.markdown(f"**Flagged Categories:** `{data['flagged_categories'] or ['none']}`")
                    st.markdown("**Cleaned Text (Post-Preprocessing):**")
                    st.code(str(data["cleaned_text"]), language="text")
                    st.markdown("**Extracted Social Signals:**")
                    st.json(data.get("signals", {}))
                with c2:
                    st.subheader("Category Toxicity Breakdown")
                    render_category_breakdown(dict(data["category_scores"]))

    with tab_batch:
        st.subheader("Batch CSV Upload or Live Social Stream Simulation")
        col_csv, col_sim = st.columns(2)
        with col_csv:
            uploaded = st.file_uploader("Upload CSV with a `text` column", type=["csv"])
            if uploaded is not None:
                df = pd.read_csv(uploaded)
                if "text" not in df.columns:
                    st.error("CSV must contain a `text` column.")
                else:
                    st.write(f"Loaded {len(df)} rows.")
                    if st.button("Run Batch Moderation on CSV", type="primary"):
                        items = [
                            {
                                "text": str(row["text"]),
                                "platform": str(row.get("platform", "csv_upload")),
                                "post_id": str(row.get("post_id", "csv_batch")),
                                "author_handle": str(row.get("author", "anonymous")),
                            }
                            for _, row in df.head(500).iterrows()
                        ]
                        ok, resp = api_request("POST", "/predict/batch", json={"items": items})
                        if ok and isinstance(resp, dict):
                            st.success(
                                f"Processed {resp['total']} comments | Toxic: {resp['toxic_count']} | "
                                f"Borderline Review: {resp['borderline_count']}"
                            )
                            st.dataframe(pd.DataFrame(resp["results"]), width="stretch")
                        else:
                            local_clf = get_local_classifier()
                            preds = local_clf.predict_batch([it["text"] for it in items])
                            st.dataframe(
                                pd.DataFrame(
                                    [
                                        {
                                            "raw_text": p.raw_text,
                                            "cleaned_text": p.cleaned_text,
                                            "overall_score": p.overall_score,
                                            "action": p.action,
                                            "flagged_categories": ", ".join(p.flagged_categories),
                                        }
                                        for p in preds
                                    ]
                                ),
                                width="stretch",
                            )

        with col_sim:
            st.markdown("Replay sample X / Reddit / YouTube comments through the live pipeline:")
            sim_count = st.slider("Number of simulated comments", min_value=3, max_value=8, value=8)
            if st.button("Simulate Live Social Stream", width="stretch"):
                ok, resp = api_request("POST", f"/ingest/simulate?limit={sim_count}")
                if ok and isinstance(resp, dict):
                    st.success(
                        f"Streamed {resp['total']} comments | Toxic: {resp['toxic_count']} | "
                        f"Borderline: {resp['borderline_count']}"
                    )
                    if resp.get("hate_raid_alerts"):
                        st.error(f"Hate-Raid Spike Detected: {resp['hate_raid_alerts']}")
                    st.dataframe(
                        pd.DataFrame(resp["results"])[
                            ["platform", "post_id", "author_handle", "raw_text", "overall_score", "action"]
                        ],
                        width="stretch",
                    )
                else:
                    events = list(MockSocialStream().stream(limit=sim_count))
                    local_clf = get_local_classifier()
                    rows = []
                    for ev in events:
                        pred = local_clf.predict(ev.text)
                        rows.append(
                            {
                                "platform": ev.platform,
                                "post_id": ev.post_id,
                                "author": ev.author_handle,
                                "text": ev.text,
                                "score": pred.overall_score,
                                "action": pred.action,
                            }
                        )
                    st.dataframe(pd.DataFrame(rows), width="stretch")

    with tab_review:
        st.subheader("Human-in-the-Loop Borderline Queue (60%–80% Confidence)")
        if not health_ok:
            st.info("Start the FastAPI backend to manage persisted review queue items.")
        else:
            ok, queue = api_request("GET", "/review-queue")
            if ok and isinstance(queue, list):
                if not queue:
                    st.success("Review queue is empty.")
                for item in queue:
                    with st.expander(
                        f"[{item['platform']}] @{item['author_handle']} • Score: {item['overall_score']:.1%} • Post: {item['post_id']}",
                        expanded=True,
                    ):
                        st.write(f"**Raw:** {item['raw_text']}")
                        st.write(f"**Cleaned:** `{item['cleaned_text']}`")
                        st.write(f"**Flagged:** {item['flagged_categories']}")
                        notes = st.text_input("Moderator notes", key=f"notes_{item['id']}")
                        b1, b2 = st.columns(2)
                        if b1.button("✅ Approve Safe", key=f"approve_{item['id']}"):
                            api_request(
                                "POST",
                                f"/review/{item['id']}",
                                json={"decision": "approved_safe", "moderator_notes": notes or None},
                            )
                            st.rerun()
                        if b2.button("🚫 Confirm Toxic / Block", key=f"block_{item['id']}"):
                            api_request(
                                "POST",
                                f"/review/{item['id']}",
                                json={"decision": "confirmed_toxic", "moderator_notes": notes or None},
                            )
                            st.rerun()

    with tab_spikes:
        st.subheader("Real-Time Toxicity Spikes & Moderation Logs")
        if not health_ok:
            st.info("Start the FastAPI backend to view aggregated database metrics and hate-raid alerts.")
        else:
            ok_stats, stats = api_request("GET", "/analytics/spikes")
            ok_logs, logs = api_request("GET", "/logs?limit=100")
            if ok_stats and isinstance(stats, dict):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Processed", stats["total_processed"])
                m2.metric("Toxic Detected", stats["toxic_count"])
                m3.metric("Auto-Blocked", stats["blocked_count"])
                m4.metric("Pending Review", stats["pending_review_count"])

                if stats.get("hate_raid_alerts"):
                    st.error("Active Hate-Raid Alerts")
                    st.dataframe(pd.DataFrame(stats["hate_raid_alerts"]), width="stretch")
                else:
                    st.success("No active hate-raid spikes detected.")

            if ok_logs and isinstance(logs, list):
                render_spike_timeline(logs)
                if logs:
                    st.markdown("### Recent Moderation Logs")
                    st.dataframe(
                        pd.DataFrame(logs)[
                            ["id", "platform", "post_id", "author_handle", "raw_text", "overall_score", "action", "review_status"]
                        ],
                        width="stretch",
                    )


if __name__ == "__main__":
    main()
