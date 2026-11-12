"""FastAPI application entrypoint and lifecycle management."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.model.classifier import ToxicityClassifier

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def create_app(classifier: ToxicityClassifier | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if getattr(application.state, "classifier", None) is None:
            application.state.classifier = classifier or ToxicityClassifier(
                toxicity_threshold=_float_env("TOXICITY_THRESHOLD", 0.50),
                borderline_low=_float_env("BORDERLINE_LOW", 0.60),
                borderline_high=_float_env("BORDERLINE_HIGH", 0.80),
                fast_pass_safe_threshold=_float_env("FAST_PASS_SAFE_THRESHOLD", 0.20),
            )
        yield

    application = FastAPI(
        title="Social Media Toxicity Detector API",
        description="Asynchronous social media hate speech and toxicity detection service.",
        version="0.1.0",
        lifespan=lifespan,
    )

    if classifier is not None:
        application.state.classifier = classifier

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router)
    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("APP_ENV", "development") == "development",
    )
