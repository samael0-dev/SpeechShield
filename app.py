"""Hugging Face Spaces entrypoint serving the SpeechShield Tailwind UI + FastAPI + optional Gradio."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn

from src.api.main import app as fastapi_app
from src.model.classifier import ToxicityClassifier

logger = logging.getLogger(__name__)


def ensure_baseline_weights() -> None:
    """Generate starter dataset and baseline weights on first boot when missing."""
    backend = os.getenv("MODEL_BACKEND", "hybrid").lower()
    if backend == "heuristic":
        return

    weights_path = Path(os.getenv("BASELINE_WEIGHTS_PATH", "data/weights/baseline_model.joblib"))
    if weights_path.exists():
        return

    try:
        from scripts.train_model import (
            DEFAULT_DATA_PATH,
            load_or_create_training_data,
            train_baseline,
        )

        logger.info("Baseline weights missing at %s; training starter model...", weights_path)
        df = load_or_create_training_data(DEFAULT_DATA_PATH)
        train_baseline(df, output_path=weights_path, model_type="logreg")
    except (OSError, RuntimeError, ValueError, ImportError) as exc:  # pragma: no cover - startup safety fallback
        logger.warning("Could not bootstrap baseline weights (%s); falling back to heuristic.", exc)


ensure_baseline_weights()

try:
    import gradio as gr

    def _predict_gradio(text: str) -> tuple[str, dict[str, float]]:
        classifier: ToxicityClassifier | None = getattr(fastapi_app.state, "classifier", None)
        if classifier is None:
            classifier = ToxicityClassifier()
            fastapi_app.state.classifier = classifier
        res = classifier.predict(text)
        summary = f"Action: {res.action.upper()} | Overall: {res.overall_score:.1%} | Model: {res.model_used}"
        return summary, res.category_scores

    demo = gr.Interface(
        fn=_predict_gradio,
        inputs=gr.Textbox(lines=3, label="Social Post / Comment"),
        outputs=[
            gr.Textbox(label="Moderation Verdict"),
            gr.Label(label="Category Scores"),
        ],
        title="SpeechShield — Companion Inspector",
        description="Full Tailwind CSS console is served at `/`.",
    )
    app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")
except ImportError:
    app = fastapi_app


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
    )
