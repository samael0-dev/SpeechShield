"""SQLAlchemy persistence layer for flagged comments, chat logs, and human review queue."""

from __future__ import annotations

import os
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def _default_database_url() -> str:
    url = os.getenv("DATABASE_URL", "sqlite:///./data/toxicity_logs.db")
    if url.startswith("sqlite:///"):
        db_file = Path(url.replace("sqlite:///", "", 1))
        db_file.parent.mkdir(parents=True, exist_ok=True)
    return url


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class ModerationLog(Base):
    """Stores processed social comments, toxicity breakdown, and human review status."""

    __tablename__ = "moderation_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform: Mapped[str] = mapped_column(String(32), default="social", index=True)
    post_id: Mapped[str] = mapped_column(String(128), default="global_feed", index=True)
    author_handle: Mapped[str] = mapped_column(String(128), default="anonymous")
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    cleaned_text: Mapped[str] = mapped_column(Text, nullable=False)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    is_toxic: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    needs_human_review: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    action: Mapped[str] = mapped_column(String(16), default="allow", index=True)
    severity: Mapped[str] = mapped_column(String(16), default="low")
    flagged_categories: Mapped[list[str]] = mapped_column(JSON, default=list)
    category_scores: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    model_used: Mapped[str] = mapped_column(String(128), default="heuristic-fallback")
    review_status: Mapped[str] = mapped_column(String(32), default="auto_resolved", index=True)
    moderator_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        index=True,
    )


def build_engine_and_session(database_url: str | None = None) -> tuple[object, sessionmaker[Session]]:
    """Create SQLAlchemy engine and session factory."""
    url = database_url or _default_database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args, future=True)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)
    return engine, session_factory


_ENGINE, SessionLocal = build_engine_and_session()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a scoped SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
