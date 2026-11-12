"""Toxicity classifier combining fast-pass baseline filtering and BERTweet/RoBERTa inference."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from src.data_pipeline.cleaner import CleanedTextResult, SocialTextCleaner
from src.model.utils import (
    TOXICITY_CATEGORIES,
    ModerationDecision,
    calibrate_probabilities,
    classify_moderation_action,
)

logger = logging.getLogger(__name__)

CATEGORY_LEXICONS: dict[str, set[str]] = {
    "severe_toxic": {"kill yourself", "kys", "die", "hang yourself", "burn in hell"},
    "threat": {"kill", "murder", "shoot", "stab", "bomb", "hurt you", "watch your back", "find where you live"},
    "obscene": {"fuck", "fucking", "shit", "shitty", "bitch", "asshole", "ass", "dick", "cunt", "slut", "whore"},
    "insult": {"idiot", "stupid", "moron", "dumb", "loser", "pathetic", "trash", "garbage", "clown", "scum", "ugly"},
    "identity_hate": {
        "nigger", "faggot", "retard", "retarded", "racist", "xenophobe",
        "your kind", "go back to where you came from", "identity hate",
    },
}


@dataclass(slots=True)
class PredictionResult:
    """Complete prediction payload for a single social media text."""

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
    fast_passed: bool = False
    signals: dict[str, Any] = field(default_factory=dict)


class ToxicityClassifier:
    """Multi-stage social media toxicity classifier.

    Execution order in `hybrid` mode:
    1. Clean social text (hashtags, emojis, contractions, obfuscated slurs).
    2. If a trained baseline model exists, compute fast-pass probability; when
       below `fast_pass_safe_threshold`, short-circuit as safe (`fast_passed=True`).
    3. Otherwise run the fine-tuned BERTweet/RoBERTa transformer (if available),
       falling back to the trained baseline model or deterministic lexicon scoring.
    """

    def __init__(
        self,
        backend: str | None = None,
        transformer_model_name: str | None = None,
        baseline_weights_path: str | Path | None = None,
        transformer_weights_dir: str | Path | None = None,
        toxicity_threshold: float = 0.50,
        borderline_low: float = 0.60,
        borderline_high: float = 0.80,
        fast_pass_safe_threshold: float = 0.20,
        allow_hub_download: bool = False,
        cleaner: SocialTextCleaner | None = None,
    ) -> None:
        self.backend = (backend or os.getenv("MODEL_BACKEND", "hybrid")).lower()
        self.transformer_model_name = transformer_model_name or os.getenv(
            "TRANSFORMER_MODEL_NAME", "vinai/bertweet-base"
        )
        self.baseline_weights_path = Path(
            baseline_weights_path or os.getenv("BASELINE_WEIGHTS_PATH", "data/weights/baseline_model.joblib")
        )
        self.transformer_weights_dir = Path(
            transformer_weights_dir or os.getenv("TRANSFORMER_WEIGHTS_DIR", "data/weights/bertweet_toxicity")
        )
        self.toxicity_threshold = toxicity_threshold
        self.borderline_low = borderline_low
        self.borderline_high = borderline_high
        self.fast_pass_safe_threshold = fast_pass_safe_threshold
        self.allow_hub_download = allow_hub_download
        self.cleaner = cleaner or SocialTextCleaner(use_ekphrasis=True)

        self._baseline_bundle: dict[str, Any] | None = None
        self._transformer_pipeline: Any = None
        self._load_available_models()

    def _load_available_models(self) -> None:
        if self.backend in {"hybrid", "baseline"} and self.baseline_weights_path.exists():
            try:
                self._baseline_bundle = joblib.load(self.baseline_weights_path)
                logger.info("Loaded baseline model from %s", self.baseline_weights_path)
            except (OSError, ValueError, RuntimeError, EOFError) as exc:
                logger.warning("Failed to load baseline weights (%s): %s", self.baseline_weights_path, exc)

        if self.backend in {"hybrid", "transformer"}:
            local_ready = self.transformer_weights_dir.exists() and any(self.transformer_weights_dir.iterdir())
            if local_ready or self.allow_hub_download:
                self._load_transformer(local_ready)

    def _load_transformer(self, local_ready: bool) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )

            source = str(self.transformer_weights_dir) if local_ready else self.transformer_model_name
            device = 0 if torch.cuda.is_available() else -1
            tokenizer = AutoTokenizer.from_pretrained(source, normalization=True)
            model = AutoModelForSequenceClassification.from_pretrained(source)
            self._transformer_pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                top_k=None,
                truncation=True,
                max_length=128,
                device=device,
            )
            logger.info("Loaded transformer classifier from %s", source)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("Transformer unavailable (%s); continuing with baseline/heuristic.", exc)
            self._transformer_pipeline = None

    @property
    def active_backends(self) -> dict[str, bool]:
        """Report which ML model backends are currently loaded in memory."""
        return {
            "baseline_loaded": self._baseline_bundle is not None,
            "transformer_loaded": self._transformer_pipeline is not None,
            "heuristic_ready": True,
        }

    def _predict_baseline(self, cleaned_text: str) -> dict[str, float] | None:
        if not self._baseline_bundle:
            return None
        vectorizer = self._baseline_bundle["vectorizer"]
        classifiers = self._baseline_bundle["classifiers"]
        features = vectorizer.transform([cleaned_text])
        scores: dict[str, float] = {}
        for cat in TOXICITY_CATEGORIES:
            clf = classifiers.get(cat)
            if clf is None:
                scores[cat] = 0.0
                continue
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(features)[0]
                scores[cat] = float(probs[1]) if len(probs) > 1 else float(probs[0])
            else:
                pred = float(clf.predict(features)[0])
                scores[cat] = max(0.0, min(1.0, pred))
        return calibrate_probabilities(scores)

    def _predict_transformer(self, cleaned_text: str) -> dict[str, float] | None:
        if self._transformer_pipeline is None:
            return None
        raw_outputs = self._transformer_pipeline(cleaned_text)
        items = raw_outputs[0] if raw_outputs and isinstance(raw_outputs[0], list) else raw_outputs
        scores = {cat: 0.0 for cat in TOXICITY_CATEGORIES}
        for entry in items:
            label = str(entry.get("label", "")).lower().replace("-", "_").replace(" ", "_")
            score = float(entry.get("score", 0.0))
            if label in scores:
                scores[label] = score
            elif label in {"hate", "hateful", "toxic", "label_1", "offensive"}:
                scores["toxic"] = max(scores["toxic"], score)
        return calibrate_probabilities(scores)

    @staticmethod
    def _predict_heuristic(cleaned: CleanedTextResult) -> dict[str, float]:
        """Deterministic lexicon + social-signal scorer used when no trained weights exist."""
        text = cleaned.cleaned_text
        words = set(re.findall(r"[a-z']+", text))
        scores: dict[str, float] = {cat: 0.02 for cat in TOXICITY_CATEGORIES}

        for category, phrases in CATEGORY_LEXICONS.items():
            hits = 0
            for phrase in phrases:
                if " " in phrase:
                    if phrase in text:
                        hits += 1
                elif phrase in words:
                    hits += 1
            if hits > 0:
                base = 0.56 + min(0.38, (hits - 1) * 0.16)
                scores[category] = max(scores[category], base)

        # Social emphasis boosters (ALL CAPS shouting, obfuscation, emojis like clown face, negative sentiment)
        emphasis_boost = 0.0
        if cleaned.uppercase_ratio >= 0.6 and len(cleaned.raw_text) >= 8:
            emphasis_boost += 0.08
        if cleaned.obfuscation_hits > 0:
            emphasis_boost += min(0.15, 0.08 * cleaned.obfuscation_hits)
        if "clown face" in text or "middle finger" in text:
            emphasis_boost += 0.08
            scores["insult"] = max(scores["insult"], 0.62)
        if cleaned.sentiment_score < -0.4:
            emphasis_boost += 0.05
        emphasis_boost = min(0.22, emphasis_boost)

        peak_non_toxic = max(scores[c] for c in TOXICITY_CATEGORIES if c != "toxic")
        if peak_non_toxic >= 0.45:
            scores["toxic"] = max(scores["toxic"], min(0.98, peak_non_toxic + emphasis_boost))
        elif cleaned.sentiment_score < -0.6 and emphasis_boost > 0:
            scores["toxic"] = max(scores["toxic"], 0.45 + emphasis_boost)

        return calibrate_probabilities(scores)

    def predict(self, text: str) -> PredictionResult:
        """Clean and classify a single comment with calibrated thresholds."""
        cleaned = self.cleaner.clean(text)
        model_used = "heuristic-fallback"
        fast_passed = False
        category_scores: dict[str, float] | None = None

        if self.backend == "baseline":
            category_scores = self._predict_baseline(cleaned.cleaned_text)
            if category_scores is not None:
                model_used = "baseline-tfidf"
        elif self.backend == "transformer":
            category_scores = self._predict_transformer(cleaned.cleaned_text)
            if category_scores is not None:
                model_used = f"transformer:{self.transformer_model_name}"
        elif self.backend == "hybrid":
            baseline_scores = self._predict_baseline(cleaned.cleaned_text)
            if baseline_scores is not None:
                baseline_peak = max(baseline_scores.values(), default=0.0)
                if baseline_peak < self.fast_pass_safe_threshold:
                    category_scores = baseline_scores
                    model_used = "baseline-fastpass"
                    fast_passed = True
                else:
                    transformer_scores = self._predict_transformer(cleaned.cleaned_text)
                    if transformer_scores is not None:
                        category_scores = transformer_scores
                        model_used = f"transformer:{self.transformer_model_name}"
                    else:
                        category_scores = baseline_scores
                        model_used = "baseline-tfidf"
            else:
                transformer_scores = self._predict_transformer(cleaned.cleaned_text)
                if transformer_scores is not None:
                    category_scores = transformer_scores
                    model_used = f"transformer:{self.transformer_model_name}"

        if category_scores is None:
            category_scores = self._predict_heuristic(cleaned)
            model_used = "heuristic-fallback"

        decision: ModerationDecision = classify_moderation_action(
            category_scores=category_scores,
            toxicity_threshold=self.toxicity_threshold,
            borderline_low=self.borderline_low,
            borderline_high=self.borderline_high,
        )

        return PredictionResult(
            raw_text=cleaned.raw_text,
            cleaned_text=cleaned.cleaned_text,
            overall_score=decision.overall_score,
            is_toxic=decision.is_toxic,
            needs_human_review=decision.needs_human_review,
            action=decision.action,
            severity=decision.severity,
            flagged_categories=decision.flagged_categories,
            category_scores=category_scores,
            model_used=model_used,
            fast_passed=fast_passed,
            signals={
                "hashtags": cleaned.hashtags,
                "emojis": cleaned.emojis,
                "obfuscation_hits": cleaned.obfuscation_hits,
                "uppercase_ratio": cleaned.uppercase_ratio,
                "sentiment_score": cleaned.sentiment_score,
                "used_ekphrasis": cleaned.used_ekphrasis,
            },
        )

    def predict_batch(self, texts: Sequence[str]) -> list[PredictionResult]:
        """Classify a batch of social media comments."""
        return [self.predict(t) for t in texts]
