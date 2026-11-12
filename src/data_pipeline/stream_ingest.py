"""Social media stream ingestion adapters (Twitter/X, Reddit, Kafka, RabbitMQ, and local replay)."""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SocialCommentEvent:
    """Canonical representation of an ingested social media post or comment."""

    event_id: str
    platform: str
    author_handle: str
    text: str
    post_id: str = "global_feed"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class MockSocialStream:
    """Generates or replays realistic social media comments for local development."""

    DEFAULT_EVENTS: ClassVar[list[dict[str, str]]] = [
        {"platform": "x", "author": "gamer_alex", "post_id": "post_101", "text": "GG everyone, that match was super intense! 🙌🔥"},
        {"platform": "reddit", "author": "dev_maya", "post_id": "post_101", "text": "Thanks for sharing the detailed breakdown, really helpful."},
        {"platform": "x", "author": "troll_99", "post_id": "post_101", "text": "You are such a pathetic clown 🤡 #TrashPlayer"},
        {"platform": "youtube", "author": "anon_user", "post_id": "post_202", "text": "Shut up you stupid @ss $lut nobody asked for your opinion"},
        {"platform": "x", "author": "bot_raid_1", "post_id": "post_303", "text": "Go back to where you came from #IdentityHate nobody wants your kind here"},
        {"platform": "reddit", "author": "critic_sam", "post_id": "post_202", "text": "Honestly this update broke half the features, pretty annoying tbh."},
        {"platform": "x", "author": "toxic_42", "post_id": "post_303", "text": "I will find where you live and hurt you, watch your back"},
        {"platform": "youtube", "author": "kind_viewer", "post_id": "post_101", "text": "Love the energy in this community ❤️ keep going!"},
    ]

    def __init__(self, source_path: str | Path | None = None) -> None:
        self.source_path = Path(source_path) if source_path else None

    def stream(self, limit: int | None = None, delay_seconds: float = 0.0) -> Iterator[SocialCommentEvent]:
        """Yield `SocialCommentEvent` items from a local CSV/JSONL file or built-in samples."""
        if self.source_path and self.source_path.exists():
            yield from self._stream_file(self.source_path, limit=limit, delay_seconds=delay_seconds)
            return

        for idx, item in enumerate(self.DEFAULT_EVENTS):
            if limit is not None and idx >= limit:
                break
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            yield SocialCommentEvent(
                event_id=f"sim_{idx + 1}",
                platform=item["platform"],
                author_handle=item["author"],
                post_id=item["post_id"],
                text=item["text"],
            )

    def _stream_file(
        self, path: Path, limit: int | None = None, delay_seconds: float = 0.0
    ) -> Iterator[SocialCommentEvent]:
        count = 0
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if limit is not None and count >= limit:
                        break
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    count += 1
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)
                    yield SocialCommentEvent(
                        event_id=str(row.get("id", f"file_{idx + 1}")),
                        platform=str(row.get("platform", "social")),
                        author_handle=str(row.get("author", "anonymous")),
                        post_id=str(row.get("post_id", "feed")),
                        text=str(row.get("text", "")),
                    )
        elif path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    if limit is not None and count >= limit:
                        break
                    count += 1
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)
                    yield SocialCommentEvent(
                        event_id=str(row.get("id", f"csv_{idx + 1}")),
                        platform=str(row.get("platform", "social")),
                        author_handle=str(row.get("author", "anonymous")),
                        post_id=str(row.get("post_id", "feed")),
                        text=str(row.get("text", "")),
                    )


class SocialStreamIngestor:
    """Adapter for live Twitter/X (Tweepy), Reddit (PRAW), and Kafka/RabbitMQ publishing."""

    def __init__(
        self,
        kafka_bootstrap: str | None = None,
        kafka_topic: str = "social-comments-raw",
        rabbitmq_url: str | None = None,
    ) -> None:
        self.kafka_bootstrap = kafka_bootstrap or os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        self.kafka_topic = kafka_topic
        self.rabbitmq_url = rabbitmq_url or os.getenv("RABBITMQ_URL")
        self._kafka_producer: Any = None

    def _get_kafka_producer(self) -> Any:
        if self._kafka_producer is not None:
            return self._kafka_producer
        try:
            from kafka import KafkaProducer

            self._kafka_producer = KafkaProducer(
                bootstrap_servers=self.kafka_bootstrap,
                value_serializer=lambda v: v.encode("utf-8"),
            )
            return self._kafka_producer
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("Kafka producer unavailable (%s); events will not be published.", exc)
            return None

    def publish_to_kafka(self, event: SocialCommentEvent) -> bool:
        """Send an ingested social event to Kafka if configured."""
        producer = self._get_kafka_producer()
        if producer is None:
            return False
        producer.send(self.kafka_topic, value=event.to_json())
        producer.flush()
        return True

    def fetch_reddit_comments(self, subreddit: str, limit: int = 25) -> list[SocialCommentEvent]:
        """Fetch latest comments from a subreddit via PRAW."""
        try:
            import praw
        except ImportError as exc:
            raise RuntimeError("praw is not installed. Run `pip install praw`.") from exc

        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT", "social-toxicity-detector/1.0")
        if not client_id or not client_secret:
            raise ValueError("REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set.")

        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        events: list[SocialCommentEvent] = []
        for comment in reddit.subreddit(subreddit).comments(limit=limit):
            events.append(
                SocialCommentEvent(
                    event_id=str(comment.id),
                    platform="reddit",
                    author_handle=str(comment.author or "[deleted]"),
                    post_id=str(comment.link_id),
                    text=str(comment.body),
                )
            )
        return events

    def fetch_twitter_recent(self, query: str, max_results: int = 20) -> list[SocialCommentEvent]:
        """Fetch recent posts from X/Twitter API v2 via Tweepy."""
        try:
            import tweepy
        except ImportError as exc:
            raise RuntimeError("tweepy is not installed. Run `pip install tweepy`.") from exc

        bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        if not bearer_token:
            raise ValueError("TWITTER_BEARER_TOKEN must be set.")

        client = tweepy.Client(bearer_token=bearer_token)
        response = client.search_recent_tweets(
            query=query,
            max_results=max(10, min(max_results, 100)),
            tweet_fields=["author_id", "created_at", "conversation_id"],
        )
        events: list[SocialCommentEvent] = []
        for tweet in response.data or []:
            events.append(
                SocialCommentEvent(
                    event_id=str(tweet.id),
                    platform="x",
                    author_handle=str(tweet.author_id or "unknown"),
                    post_id=str(tweet.conversation_id or tweet.id),
                    text=str(tweet.text),
                    created_at=tweet.created_at.isoformat() if tweet.created_at else datetime.now(UTC).isoformat(),
                )
            )
        return events
