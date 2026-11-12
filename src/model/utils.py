"""Helper functions for confidence thresholding, calibration, and spike detection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

TOXICITY_CATEGORIES: tuple[str, ...] = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)

DEFAULT_CATEGORY_THRESHOLDS: dict[str, float] = {
    "toxic": 0.50,
    "severe_toxic": 0.45,
    "obscene": 0.55,
    "threat": 0.40,
    "insult": 0.50,
    "identity_hate": 0.40,
}


@dataclass(slots=True)
class ModerationDecision:
    """Routing decision derived from calibrated toxicity probabilities."""

    overall_score: float
    is_toxic: bool
    needs_human_review: bool
    action: str  # "allow" | "review" | "block"
    flagged_categories: list[str]
    severity: str  # "low" | "borderline" | "high" | "critical"


def calibrate_probabilities(
    raw_scores: Mapping[str, float],
    temperature: float = 1.0,
) -> dict[str, float]:
    """Clamp and temperature-scale category probabilities into [0.0, 1.0]."""
    calibrated: dict[str, float] = {}
    safe_temp = max(temperature, 1e-3)
    for category in TOXICITY_CATEGORIES:
        val = float(raw_scores.get(category, 0.0))
        val = max(0.0, min(1.0, val))
        if safe_temp != 1.0 and 0.0 < val < 1.0:
            # Power-scaling calibration that preserves 0 and 1 endpoints
            val = val ** (1.0 / safe_temp)
        calibrated[category] = round(val, 4)
    return calibrated


def classify_moderation_action(
    category_scores: Mapping[str, float],
    toxicity_threshold: float = 0.50,
    borderline_low: float = 0.60,
    borderline_high: float = 0.80,
    category_thresholds: Mapping[str, float] | None = None,
) -> ModerationDecision:
    """Map category scores into moderation flags and human-review routing.

    Comments whose peak toxicity probability falls within [borderline_low, borderline_high]
    (default 60%-80%) are routed to the human moderator review queue (`action="review"`).
    """
    thresholds = dict(DEFAULT_CATEGORY_THRESHOLDS)
    if category_thresholds:
        thresholds.update(category_thresholds)

    flagged = [
        cat
        for cat in TOXICITY_CATEGORIES
        if category_scores.get(cat, 0.0) >= thresholds.get(cat, toxicity_threshold)
    ]

    overall_score = round(max((float(category_scores.get(c, 0.0)) for c in TOXICITY_CATEGORIES), default=0.0), 4)
    in_borderline_band = borderline_low <= overall_score <= borderline_high

    if overall_score > borderline_high or category_scores.get("threat", 0.0) >= 0.70 or category_scores.get("identity_hate", 0.0) >= 0.75:
        action = "block"
        severity = "critical" if overall_score >= 0.90 else "high"
        is_toxic = True
        needs_human_review = False
    elif in_borderline_band:
        action = "review"
        severity = "borderline"
        is_toxic = overall_score >= toxicity_threshold
        needs_human_review = True
    elif overall_score >= toxicity_threshold or flagged:
        action = "review"
        severity = "borderline"
        is_toxic = True
        needs_human_review = True
    else:
        action = "allow"
        severity = "low"
        is_toxic = False
        needs_human_review = False

    return ModerationDecision(
        overall_score=overall_score,
        is_toxic=is_toxic,
        needs_human_review=needs_human_review,
        action=action,
        flagged_categories=flagged,
        severity=severity,
    )


def detect_hate_raid_spikes(
    records: Iterable[Mapping[str, object]],
    min_comments: int = 3,
    toxic_ratio_threshold: float = 0.50,
) -> list[dict[str, object]]:
    """Identify posts/threads experiencing concentrated toxic bursts ("hate raids")."""
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in records:
        post_id = str(row.get("post_id") or "global_feed")
        grouped[post_id].append(row)

    alerts: list[dict[str, object]] = []
    for post_id, items in grouped.items():
        total = len(items)
        if total < min_comments:
            continue
        toxic_count = sum(1 for item in items if bool(item.get("is_toxic")))
        avg_score = sum(float(item.get("overall_score", 0.0)) for item in items) / total
        toxic_ratio = toxic_count / total
        if toxic_ratio >= toxic_ratio_threshold:
            alerts.append(
                {
                    "post_id": post_id,
                    "total_comments": total,
                    "toxic_comments": toxic_count,
                    "toxic_ratio": round(toxic_ratio, 4),
                    "avg_toxicity_score": round(avg_score, 4),
                    "alert_level": "critical" if toxic_ratio >= 0.75 else "warning",
                }
            )
    alerts.sort(key=lambda a: (float(a["toxic_ratio"]), int(a["toxic_comments"])), reverse=True)
    return alerts
