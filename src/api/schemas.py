"""Pydantic request and response schemas for the toxicity detection API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Single social media comment inference request."""

    text: str = Field(..., min_length=1, max_length=5000, description="Raw social media post or comment.")
    platform: str = Field(default="social", max_length=32, description="Origin platform (x, reddit, youtube, etc.).")
    post_id: str = Field(default="global_feed", max_length=128, description="Parent post or thread identifier.")
    author_handle: str = Field(default="anonymous", max_length=128, description="Author username or ID.")


class BatchPredictRequest(BaseModel):
    """Batch inference request for concurrent webhook or CSV moderation."""

    items: list[PredictRequest] = Field(..., min_length=1, max_length=500)


class SocialSignals(BaseModel):
    """Extracted social media metadata signals."""

    hashtags: list[str] = Field(default_factory=list)
    emojis: list[str] = Field(default_factory=list)
    obfuscation_hits: int = 0
    uppercase_ratio: float = 0.0
    sentiment_score: float = 0.0
    used_ekphrasis: bool = False


class PredictionResponse(BaseModel):
    """Structured toxicity prediction and moderation action."""

    log_id: int | None = None
    platform: str
    post_id: str
    author_handle: str
    raw_text: str
    cleaned_text: str
    overall_score: float
    is_toxic: bool
    needs_human_review: bool
    action: Literal["allow", "review", "block"]
    severity: Literal["low", "borderline", "high", "critical"]
    flagged_categories: list[str]
    category_scores: dict[str, float]
    model_used: str
    fast_passed: bool = False
    signals: SocialSignals
    created_at: datetime


class HateRaidAlert(BaseModel):
    """Detected spike of concentrated toxic comments on a post."""

    post_id: str
    total_comments: int
    toxic_comments: int
    toxic_ratio: float
    avg_toxicity_score: float
    alert_level: Literal["warning", "critical"]


class BatchPredictResponse(BaseModel):
    """Response for batch predictions including any detected hate-raid alerts."""

    total: int
    toxic_count: int
    borderline_count: int
    results: list[PredictionResponse]
    hate_raid_alerts: list[HateRaidAlert] = Field(default_factory=list)


class ModerationLogItem(BaseModel):
    """Database record representation for moderation logs and review queue."""

    id: int
    platform: str
    post_id: str
    author_handle: str
    raw_text: str
    cleaned_text: str
    overall_score: float
    is_toxic: bool
    needs_human_review: bool
    action: str
    severity: str
    flagged_categories: list[str]
    category_scores: dict[str, float]
    model_used: str
    review_status: str
    moderator_notes: str | None = None
    created_at: datetime


class HumanReviewRequest(BaseModel):
    """Human moderator decision for borderline flagged comments."""

    decision: Literal["approved_safe", "confirmed_toxic"]
    moderator_notes: str | None = Field(default=None, max_length=500)


class AnalyticsSummaryResponse(BaseModel):
    """Aggregated moderation stats and active hate-raid alerts."""

    total_processed: int
    toxic_count: int
    blocked_count: int
    pending_review_count: int
    avg_toxicity_score: float
    hate_raid_alerts: list[HateRaidAlert]
    category_counts: dict[str, int]
    platform_counts: dict[str, int]


class HealthResponse(BaseModel):
    """API and ML backend health status."""

    status: str
    database_connected: bool
    backends: dict[str, Any]
