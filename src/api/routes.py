"""FastAPI routes for asynchronous toxicity prediction, moderation logs, and analytics."""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import text as sql_text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.api.db import ModerationLog, get_db
from src.api.schemas import (
    AnalyticsSummaryResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HateRaidAlert,
    HealthResponse,
    HumanReviewRequest,
    ModerationLogItem,
    PredictionResponse,
    PredictRequest,
    SocialSignals,
)
from src.data_pipeline.stream_ingest import MockSocialStream
from src.model.classifier import PredictionResult, ToxicityClassifier
from src.model.utils import TOXICITY_CATEGORIES, detect_hate_raid_spikes

router = APIRouter()
DbSession = Annotated[Session, Depends(get_db)]


def get_classifier(request: Request) -> ToxicityClassifier:
    """Retrieve the shared classifier instance stored on `app.state`."""
    classifier: ToxicityClassifier | None = getattr(request.app.state, "classifier", None)
    if classifier is None:
        classifier = ToxicityClassifier()
        request.app.state.classifier = classifier
    return classifier


def _persist_prediction(
    db: Session,
    item: PredictRequest,
    result: PredictionResult,
) -> ModerationLog:
    review_status = "pending" if result.needs_human_review else "auto_resolved"
    record = ModerationLog(
        platform=item.platform,
        post_id=item.post_id,
        author_handle=item.author_handle,
        raw_text=result.raw_text,
        cleaned_text=result.cleaned_text,
        overall_score=result.overall_score,
        is_toxic=result.is_toxic,
        needs_human_review=result.needs_human_review,
        action=result.action,
        severity=result.severity,
        flagged_categories=result.flagged_categories,
        category_scores=result.category_scores,
        model_used=result.model_used,
        review_status=review_status,
        created_at=datetime.now(UTC),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def _to_prediction_response(
    record: ModerationLog,
    result: PredictionResult,
) -> PredictionResponse:
    return PredictionResponse(
        log_id=record.id,
        platform=record.platform,
        post_id=record.post_id,
        author_handle=record.author_handle,
        raw_text=result.raw_text,
        cleaned_text=result.cleaned_text,
        overall_score=result.overall_score,
        is_toxic=result.is_toxic,
        needs_human_review=result.needs_human_review,
        action=result.action,  # type: ignore[arg-type]
        severity=result.severity,  # type: ignore[arg-type]
        flagged_categories=result.flagged_categories,
        category_scores=result.category_scores,
        model_used=result.model_used,
        fast_passed=result.fast_passed,
        signals=SocialSignals(**result.signals),
        created_at=record.created_at,
    )


def _to_log_item(record: ModerationLog) -> ModerationLogItem:
    return ModerationLogItem(
        id=record.id,
        platform=record.platform,
        post_id=record.post_id,
        author_handle=record.author_handle,
        raw_text=record.raw_text,
        cleaned_text=record.cleaned_text,
        overall_score=record.overall_score,
        is_toxic=record.is_toxic,
        needs_human_review=record.needs_human_review,
        action=record.action,
        severity=record.severity,
        flagged_categories=list(record.flagged_categories or []),
        category_scores=dict(record.category_scores or {}),
        model_used=record.model_used,
        review_status=record.review_status,
        moderator_notes=record.moderator_notes,
        created_at=record.created_at,
    )


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(
    request: Request,
    db: DbSession,
) -> HealthResponse:
    """Return service health, loaded ML backends, and DB status."""
    classifier = get_classifier(request)
    db_ok = True
    try:
        db.execute(sql_text("SELECT 1"))
    except SQLAlchemyError:
        db_ok = False
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        database_connected=db_ok,
        backends=classifier.active_backends,
    )


@router.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_toxicity(
    payload: PredictRequest,
    request: Request,
    db: DbSession,
) -> PredictionResponse:
    """Asynchronously classify a single social media comment and log the decision."""
    classifier = get_classifier(request)
    result = await asyncio.to_thread(classifier.predict, payload.text)
    record = _persist_prediction(db, payload, result)
    return _to_prediction_response(record, result)


@router.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(
    payload: BatchPredictRequest,
    request: Request,
    db: DbSession,
) -> BatchPredictResponse:
    """Process a batch of comments concurrently and detect hate-raid spikes."""
    classifier = get_classifier(request)
    texts = [item.text for item in payload.items]
    results = await asyncio.to_thread(classifier.predict_batch, texts)

    responses: list[PredictionResponse] = []
    spike_inputs: list[dict[str, object]] = []
    for item, result in zip(payload.items, results):
        record = _persist_prediction(db, item, result)
        responses.append(_to_prediction_response(record, result))
        spike_inputs.append(
            {
                "post_id": item.post_id,
                "is_toxic": result.is_toxic,
                "overall_score": result.overall_score,
            }
        )

    alerts = [HateRaidAlert(**alert) for alert in detect_hate_raid_spikes(spike_inputs)]
    return BatchPredictResponse(
        total=len(responses),
        toxic_count=sum(1 for r in responses if r.is_toxic),
        borderline_count=sum(1 for r in responses if r.needs_human_review),
        results=responses,
        hate_raid_alerts=alerts,
    )


@router.post("/ingest/simulate", response_model=BatchPredictResponse, tags=["Streaming"])
async def simulate_social_stream(
    request: Request,
    db: DbSession,
    limit: int = Query(default=8, ge=1, le=50),
) -> BatchPredictResponse:
    """Replay sample social media events through the live classifier for demonstration."""
    events = list(MockSocialStream().stream(limit=limit))
    batch_payload = BatchPredictRequest(
        items=[
            PredictRequest(
                text=ev.text,
                platform=ev.platform,
                post_id=ev.post_id,
                author_handle=ev.author_handle,
            )
            for ev in events
        ]
    )
    return await predict_batch(batch_payload, request, db)


@router.get("/logs", response_model=list[ModerationLogItem], tags=["Moderation"])
async def list_moderation_logs(
    db: DbSession,
    action: str | None = Query(default=None),
    post_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[ModerationLogItem]:
    """Fetch recent moderation log entries ordered newest-first."""
    query = db.query(ModerationLog)
    if action:
        query = query.filter(ModerationLog.action == action)
    if post_id:
        query = query.filter(ModerationLog.post_id == post_id)
    records = query.order_by(ModerationLog.id.desc()).limit(limit).all()
    return [_to_log_item(r) for r in records]


@router.get("/review-queue", response_model=list[ModerationLogItem], tags=["Moderation"])
async def list_review_queue(
    db: DbSession,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[ModerationLogItem]:
    """Return borderline (60%-80% confidence) comments awaiting human moderation."""
    records = (
        db.query(ModerationLog)
        .filter(ModerationLog.needs_human_review.is_(True), ModerationLog.review_status == "pending")
        .order_by(ModerationLog.overall_score.desc(), ModerationLog.id.desc())
        .limit(limit)
        .all()
    )
    return [_to_log_item(r) for r in records]


@router.post("/review/{log_id}", response_model=ModerationLogItem, tags=["Moderation"])
async def submit_human_review(
    log_id: int,
    payload: HumanReviewRequest,
    db: DbSession,
) -> ModerationLogItem:
    """Apply a human moderator decision to a borderline flagged comment."""
    record = db.query(ModerationLog).filter(ModerationLog.id == log_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail=f"Moderation log {log_id} not found.")

    record.review_status = payload.decision
    record.needs_human_review = False
    if payload.decision == "approved_safe":
        record.action = "allow"
        record.is_toxic = False
    else:
        record.action = "block"
        record.is_toxic = True
    record.moderator_notes = payload.moderator_notes
    db.commit()
    db.refresh(record)
    return _to_log_item(record)


@router.get("/analytics/spikes", response_model=AnalyticsSummaryResponse, tags=["Analytics"])
async def get_analytics_summary(
    db: DbSession,
    limit: int = Query(default=500, ge=10, le=2000),
) -> AnalyticsSummaryResponse:
    """Compute aggregated toxicity metrics and hate-raid alerts across recent logs."""
    records = db.query(ModerationLog).order_by(ModerationLog.id.desc()).limit(limit).all()
    total = len(records)
    if total == 0:
        return AnalyticsSummaryResponse(
            total_processed=0,
            toxic_count=0,
            blocked_count=0,
            pending_review_count=0,
            avg_toxicity_score=0.0,
            hate_raid_alerts=[],
            category_counts={cat: 0 for cat in TOXICITY_CATEGORIES},
            platform_counts={},
        )

    cat_counter: Counter[str] = Counter({cat: 0 for cat in TOXICITY_CATEGORIES})
    platform_counter: Counter[str] = Counter()
    spike_rows: list[dict[str, object]] = []
    toxic_count = 0
    blocked_count = 0
    pending_count = 0
    score_sum = 0.0

    for rec in records:
        score_sum += rec.overall_score
        if rec.is_toxic:
            toxic_count += 1
        if rec.action == "block":
            blocked_count += 1
        if rec.needs_human_review and rec.review_status == "pending":
            pending_count += 1
        for cat in rec.flagged_categories or []:
            cat_counter[cat] += 1
        platform_counter[rec.platform] += 1
        spike_rows.append(
            {
                "post_id": rec.post_id,
                "is_toxic": rec.is_toxic,
                "overall_score": rec.overall_score,
            }
        )

    alerts = [HateRaidAlert(**alert) for alert in detect_hate_raid_spikes(spike_rows)]
    return AnalyticsSummaryResponse(
        total_processed=total,
        toxic_count=toxic_count,
        blocked_count=blocked_count,
        pending_review_count=pending_count,
        avg_toxicity_score=round(score_sum / total, 4),
        hate_raid_alerts=alerts,
        category_counts=dict(cat_counter),
        platform_counts=dict(platform_counter),
    )
