"""Training script for Baseline (TF-IDF + LogisticRegression/SVM) and Transformer (BERTweet/RoBERTa)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.scrape_data import build_starter_dataset, preprocess_dataframe
from src.model.utils import TOXICITY_CATEGORIES

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_social_comments.csv"
DEFAULT_BASELINE_PATH = PROJECT_ROOT / "data" / "weights" / "baseline_model.joblib"
DEFAULT_TRANSFORMER_DIR = PROJECT_ROOT / "data" / "weights" / "bertweet_toxicity"


def load_or_create_training_data(data_path: Path) -> pd.DataFrame:
    """Load preprocessed CSV or generate starter training data if missing."""
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        print(f"Dataset not found at {data_path}; generating starter social dataset...")
        df = preprocess_dataframe(build_starter_dataset())
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)

    if "cleaned_text" not in df.columns:
        df = preprocess_dataframe(df)
    df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
    for cat in TOXICITY_CATEGORIES:
        if cat not in df.columns:
            df[cat] = 0
        df[cat] = df[cat].fillna(0).astype(int)
    return df


def train_baseline(
    df: pd.DataFrame,
    output_path: Path,
    model_type: str = "logreg",
) -> dict[str, dict[str, float]]:
    """Train multi-label TF-IDF baseline classifiers and report Macro-F1 / Recall."""
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["toxic"] if df["toxic"].nunique() > 1 else None,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=25000,
        min_df=1,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_df["cleaned_text"])
    X_val = vectorizer.transform(val_df["cleaned_text"])

    classifiers: dict[str, object] = {}
    metrics: dict[str, dict[str, float]] = {}

    for cat in TOXICITY_CATEGORIES:
        y_train = train_df[cat].to_numpy()
        y_val = val_df[cat].to_numpy()

        if len(np.unique(y_train)) < 2:
            print(f"[Baseline] Skipping '{cat}' (single class in training split).")
            continue

        if model_type == "svm":
            base_svc = LinearSVC(class_weight="balanced", random_state=42)
            clf: object = CalibratedClassifierCV(base_svc, cv=3)
        else:
            clf = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=2.0,
                random_state=42,
            )

        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)

        precision = float(precision_score(y_val, preds, zero_division=0))
        recall = float(recall_score(y_val, preds, zero_division=0))
        macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))

        classifiers[cat] = clf
        metrics[cat] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "macro_f1": round(macro_f1, 4),
        }
        print(
            f"[Baseline | {cat:13s}] Precision={precision:.4f} Recall={recall:.4f} Macro-F1={macro_f1:.4f}"
        )

    if "toxic" in classifiers:
        toxic_preds = classifiers["toxic"].predict(X_val)  # type: ignore[union-attr]
        print("\nClassification Report ('toxic'):")
        print(classification_report(val_df["toxic"].to_numpy(), toxic_preds, zero_division=0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifiers": classifiers, "metrics": metrics}, output_path)
    print(f"Saved baseline bundle -> {output_path}")
    return metrics


def train_transformer(
    df: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    epochs: int = 2,
    batch_size: int = 8,
    use_lora: bool = False,
) -> None:
    """Fine-tune BERTweet or RoBERTa for multi-label social toxicity detection."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    label_cols = list(TOXICITY_CATEGORIES)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name, normalization=True)

    def to_hf_dataset(frame: pd.DataFrame) -> Dataset:
        records = {
            "text": frame["cleaned_text"].tolist(),
            "labels": frame[label_cols].astype(float).values.tolist(),
        }
        ds = Dataset.from_dict(records)
        return ds.map(
            lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128),
            batched=True,
        )

    train_ds = to_hf_dataset(train_df)
    val_ds = to_hf_dataset(val_df)

    id2label = {idx: label for idx, label in enumerate(label_cols)}
    label2id = {label: idx for idx, label in enumerate(label_cols)}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_cols),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "value"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        logits, labels = eval_pred
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        return {
            "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        use_cpu=not torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved fine-tuned transformer -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline or transformer social toxicity models.")
    parser.add_argument("--mode", choices=["baseline", "transformer"], default="baseline")
    parser.add_argument("--baseline-type", choices=["logreg", "svm"], default="logreg")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--baseline-out", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--transformer-model", default="vinai/bertweet-base")
    parser.add_argument("--transformer-out", type=Path, default=DEFAULT_TRANSFORMER_DIR)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--use-lora", action="store_true")
    args = parser.parse_args()

    df = load_or_create_training_data(args.data_path)
    if args.mode == "baseline":
        train_baseline(df, output_path=args.baseline_out, model_type=args.baseline_type)
    else:
        train_transformer(
            df=df,
            model_name=args.transformer_model,
            output_dir=args.transformer_out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_lora=args.use_lora,
        )


if __name__ == "__main__":
    main()
