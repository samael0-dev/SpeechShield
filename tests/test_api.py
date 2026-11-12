"""Integration tests for the FastAPI toxicity moderation endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.db import build_engine_and_session, get_db
from src.api.main import create_app
from src.model.classifier import ToxicityClassifier


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    db_path = tmp_path / "test_moderation.db"
    _, session_factory = build_engine_and_session(f"sqlite:///{db_path}")

    classifier = ToxicityClassifier(
        backend="heuristic",
        baseline_weights_path=tmp_path / "missing.joblib",
        transformer_weights_dir=tmp_path / "missing_tx",
    )
    app = create_app(classifier=classifier)

    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database_connected"] is True
    assert body["backends"]["heuristic_ready"] is True


def test_predict_safe_and_toxic_comments(client: TestClient) -> None:
    safe_resp = client.post(
        "/predict",
        json={
            "text": "Thanks for sharing this awesome tutorial!",
            "platform": "reddit",
            "post_id": "thread_1",
            "author_handle": "helpful_dev",
        },
    )
    assert safe_resp.status_code == 200
    safe_body = safe_resp.json()
    assert safe_body["action"] == "allow"
    assert safe_body["is_toxic"] is False

    toxic_resp = client.post(
        "/predict",
        json={
            "text": "Shut up you @ss $lut I will hurt you",
            "platform": "x",
            "post_id": "thread_2",
            "author_handle": "troll_user",
        },
    )
    assert toxic_resp.status_code == 200
    toxic_body = toxic_resp.json()
    assert toxic_body["is_toxic"] is True
    assert toxic_body["action"] in {"review", "block"}
    assert "obscene" in toxic_body["flagged_categories"]


def test_borderline_review_queue_workflow(client: TestClient) -> None:
    # Single-hit insult with clown emoji lands in the 60%-80% borderline review band
    resp = client.post(
        "/predict",
        json={
            "text": "You are such a clown 🤡",
            "platform": "x",
            "post_id": "post_borderline",
            "author_handle": "edgy_user",
        },
    )
    assert resp.status_code == 200
    pred = resp.json()
    assert pred["needs_human_review"] is True
    assert pred["action"] == "review"
    log_id = pred["log_id"]

    queue_resp = client.get("/review-queue")
    assert queue_resp.status_code == 200
    queue_items = queue_resp.json()
    assert any(item["id"] == log_id for item in queue_items)

    review_resp = client.post(
        f"/review/{log_id}",
        json={"decision": "approved_safe", "moderator_notes": "Friendly banter in context"},
    )
    assert review_resp.status_code == 200
    reviewed = review_resp.json()
    assert reviewed["review_status"] == "approved_safe"
    assert reviewed["action"] == "allow"
    assert reviewed["needs_human_review"] is False


def test_batch_predict_and_hate_raid_analytics(client: TestClient) -> None:
    batch_resp = client.post(
        "/predict/batch",
        json={
            "items": [
                {"text": "Go kill yourself loser", "platform": "x", "post_id": "raid_post", "author_handle": "u1"},
                {"text": "You are a stupid idiot", "platform": "x", "post_id": "raid_post", "author_handle": "u2"},
                {"text": "Shut up @ss $lut", "platform": "x", "post_id": "raid_post", "author_handle": "u3"},
                {"text": "Great stream today!", "platform": "x", "post_id": "calm_post", "author_handle": "u4"},
            ]
        },
    )
    assert batch_resp.status_code == 200
    batch_body = batch_resp.json()
    assert batch_body["total"] == 4
    assert batch_body["toxic_count"] == 3
    assert any(alert["post_id"] == "raid_post" for alert in batch_body["hate_raid_alerts"])

    analytics_resp = client.get("/analytics/spikes")
    assert analytics_resp.status_code == 200
    stats = analytics_resp.json()
    assert stats["total_processed"] >= 4
    assert any(alert["post_id"] == "raid_post" for alert in stats["hate_raid_alerts"])


def test_simulate_stream_endpoint(client: TestClient) -> None:
    resp = client.post("/ingest/simulate?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 5

    logs_resp = client.get("/logs?limit=10")
    assert logs_resp.status_code == 200
    assert len(logs_resp.json()) >= 5


def test_baseline_model_training_and_fastpass(tmp_path: Path) -> None:
    from scripts.scrape_data import build_starter_dataset, preprocess_dataframe
    from scripts.train_model import train_baseline

    df = preprocess_dataframe(build_starter_dataset())
    weights_path = tmp_path / "baseline.joblib"
    metrics = train_baseline(df, output_path=weights_path, model_type="logreg")
    assert "toxic" in metrics
    assert weights_path.exists()

    clf = ToxicityClassifier(
        backend="hybrid",
        baseline_weights_path=weights_path,
        transformer_weights_dir=tmp_path / "no_tx",
        fast_pass_safe_threshold=0.45,
    )
    assert clf.active_backends["baseline_loaded"] is True
    safe_pred = clf.predict("Thanks for sharing this benchmark, super helpful!")
    assert safe_pred.model_used in {"baseline-fastpass", "baseline-tfidf"}
    toxic_pred = clf.predict("Shut up you pathetic idiot loser")
    assert toxic_pred.overall_score >= 0.45

