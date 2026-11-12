<div align="center">

```text
███████╗██████╗ ███████╗███████╗ ██████╗██╗  ██╗███████╗██╗  ██╗██╗███████╗██╗     ██████╗ 
██╔════╝██╔══██╗██╔════╝██╔════╝██╔════╝██║  ██║██╔════╝██║  ██║██║██╔════╝██║     ██╔══██╗
███████╗██████╔╝█████╗  █████╗  ██║     ███████║███████╗███████║██║█████╗  ██║     ██║  ██║
╚════██║██╔═══╝ ██╔══╝  ██╔══╝  ██║     ██╔══██║╚════██║██╔══██║██║██╔══╝  ██║     ██║  ██║
███████║██║     ███████╗███████╗╚██████╗██║  ██║███████║██║  ██║██║███████╗███████╗██████╔╝
╚══════╝╚═╝     ╚══════╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═════╝ 
```

### *Real-Time Social Media Toxicity & Hate-Speech Moderation Platform*

**An intelligent, multi-stage social moderation system built for high-velocity internet dialects.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

</div>

---

## Overview

**SpeechShield** is an end-to-end social media toxicity and hate-speech detection platform designed to moderate high-velocity, unstructured online discussions across X (Twitter), Reddit, YouTube, and community feeds.

Traditional academic English classifiers struggle with social media dialects—emojis (`🤡`), camel-cased hashtags (`#IdentityHate`), leetspeak and obfuscated slurs (`$lut`, `@ss`, `b!tch`), elongations (`whyyyyy`), and sudden coordinated harassment bursts. SpeechShield solves this through a social-aware text preprocessing pipeline, a fast-pass traditional ML filter, transformer inference (`BERTweet` / `RoBERTa`), asynchronous FastAPI endpoints, SQLAlchemy audit persistence, and an interactive HTML + Tailwind CSS moderation dashboard.

### What SpeechShield Delivers

- **Social Dialect Preprocessing (`src/data_pipeline/cleaner.py`)** — Translates emojis to descriptive text, unpacks camel-case hashtags, expands contractions/shorthand, restores obfuscated slurs, and extracts emphasis/sentiment signals.
- **Two-Stage Hybrid Classification (`src/model/classifier.py`)** — Uses a lightweight TF-IDF + Logistic Regression / SVM fast-pass filter to clear clearly safe comments before escalating suspicious posts to BERTweet / RoBERTa.
- **Calibrated Multi-Label Scoring (`src/model/utils.py`)** — Scores 6 toxicity dimensions (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`) with category-specific thresholds.
- **Human-in-the-Loop Borderline Queue** — Automatically routes ambiguous comments in the **60%–80%** confidence band (`needs_human_review=True`) for human moderator review instead of blindly auto-blocking.
- **Hate-Raid Spike Detection** — Aggregates per-post toxicity ratios across concurrent batches and logs to detect coordinated toxic raids (`warning` / `critical`).
- **Asynchronous API & Persistence (`src/api/`)** — Serves concurrent FastAPI prediction, batch webhook, review, and analytics endpoints backed by SQLAlchemy (SQLite locally, PostgreSQL in Docker).
- **Live Moderation Dashboard (`frontend/index.html`)** — Dark/Light Tailwind CSS console featuring single-comment inspection, interactive SVG charts, borderline review actions, and spike telemetry.

---

## Architecture & System Workflow

SpeechShield is organized around two connected pipelines: a **Social Text & Hybrid Inference Pipeline** and an **Async Moderation, Persistence & Dashboard Pipeline**.

### 1. Social Preprocessing & Hybrid Inference Pipeline

Rather than sending raw noisy social text straight to a heavy transformer, every comment flows through a structured multi-stage pipeline:

| Stage | Component | Responsibility |
| --- | --- | --- |
| **01** | **Raw Social Input** | Accepts posts/comments with platform (`x`, `reddit`, `youtube`), `post_id`, and `author_handle`. |
| **02** | **Emoji & URL Normalization** | Translates Unicode emojis via `emoji` (`🤡` → `clown face`) and normalizes URLs to `<url>`. |
| **03** | **Slur De-Obfuscation** | Restores symbol/leetspeak slurs (`@ss`, `$lut`, `b!tch`, `f*ck`) before `@user` mention masking. |
| **04** | **Hashtag & Slang Unpacking** | Splits `#IdentityHate` → `Identity Hate`, expands contractions (`stfu`, `don't`), collapses elongations, and optionally applies `ekphrasis`. |
| **05** | **Signal Extraction** | Computes `obfuscation_hits`, `uppercase_ratio`, `exclamation_count`, and lexicon `sentiment_score`. |
| **06A** | **Fast-Pass Baseline Filter** | Evaluates TF-IDF + Logistic Regression / calibrated SVM (`baseline_model.joblib`). Scores below `FAST_PASS_SAFE_THRESHOLD` (`0.20`) short-circuit as safe (`baseline-fastpass`). |
| **06B** | **Transformer Deep Check** | Escalates non-cleared comments to fine-tuned `vinai/bertweet-base` / RoBERTa (`data/weights/bertweet_toxicity/`) when available, with baseline/lexicon fallback. |
| **07** | **Threshold Calibration** | Evaluates category thresholds and assigns action: `allow`, `review` (60%–80% borderline), or `block`. |
| **08** | **Persistence & Spike Check** | Logs results to `moderation_logs` and evaluates thread-level hate-raid spike alerts. |

#### Multi-Label Toxicity Categories & Thresholds

| Category | Default Threshold | Escalation Rule |
| --- | --- | --- |
| `toxic` | `0.50` | General toxicity baseline |
| `severe_toxic` | `0.45` | Severe abuse / self-harm encouragement |
| `obscene` | `0.55` | Profanity & explicit language |
| `threat` | `0.40` | Auto-`block` when `>= 0.70` |
| `insult` | `0.50` | Targeted personal attacks |
| `identity_hate` | `0.40` | Auto-`block` when `>= 0.75` |

> **Human-in-the-Loop Policy:** Comments whose peak probability falls in **`[0.60, 0.80]`** are routed to `action="review"` and queued at `GET /review-queue` so moderators can confirm (`confirmed_toxic`) or clear (`approved_safe`) them via `POST /review/{log_id}`.

---

### 2. Backend API, Streaming & Moderation Dashboard

#### Core API Endpoints (`src/api/routes.py`)

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Returns API health, SQLAlchemy DB status, and loaded ML backends. |
| `POST` | `/predict` | Async inference for a single comment; cleans, classifies, and logs to DB. |
| `POST` | `/predict/batch` | Concurrent batch inference (up to 500 items) with hate-raid spike detection. |
| `POST` | `/ingest/simulate` | Replays sample social events (`MockSocialStream`) through the live pipeline. |
| `GET` | `/logs` | Lists recent `moderation_logs` entries (filterable by `action`, `post_id`). |
| `GET` | `/review-queue` | Returns pending borderline (60%–80%) items ordered by score. |
| `POST` | `/review/{log_id}` | Records human moderator decision (`approved_safe` or `confirmed_toxic`) + notes. |
| `GET` | `/analytics/spikes` | Aggregates category counts, platform counts, and active hate-raid alerts. |

#### Core Modules

| Module | Description |
| --- | --- |
| **Social Text Cleaner** ([cleaner.py](src/data_pipeline/cleaner.py)) | Emoji translation, hashtag splitting, contraction expansion, obfuscated slur normalization, and sentiment scoring. |
| **Stream Ingestor** ([stream_ingest.py](src/data_pipeline/stream_ingest.py)) | Adapters for X/Twitter (`tweepy`), Reddit (`praw`), Apache Kafka (`kafka-python`), and `MockSocialStream`. |
| **Hybrid Toxicity Classifier** ([classifier.py](src/model/classifier.py)) | Supports `hybrid`, `baseline`, `transformer`, and `heuristic` backends with automatic fast-pass routing. |
| **Threshold & Spike Utils** ([utils.py](src/model/utils.py)) | Probability temperature calibration, moderation decision mapping, and per-`post_id` hate-raid detection. |
| **FastAPI Service** ([main.py](src/api/main.py)) | Async lifespan startup, CORS middleware, and SQLAlchemy `ModerationLog` persistence. |
| **Moderation Dashboard** ([index.html](frontend/index.html)) | HTML + Tailwind CSS 3-tab UI: Live Text Analyzer, Borderline Review Queue, and Spike Monitor. |

---

## Technology Stack

| Layer | Technologies |
| --- | --- |
| **Backend & API** | Python 3.10+, FastAPI, Pydantic v2, Uvicorn, HTTPX |
| **Deep Learning & NLP** | PyTorch, Hugging Face `transformers` (`vinai/bertweet-base`, `roberta-base`), `peft` (LoRA), `datasets` |
| **Traditional ML & Preprocessing** | Scikit-Learn (`TfidfVectorizer`, `LogisticRegression`, `LinearSVC`, `CalibratedClassifierCV`), `imbalanced-learn`, `joblib`, `emoji`, `ekphrasis`, NLTK |
| **Database & Streaming** | SQLAlchemy 2.0, SQLite (local default), PostgreSQL (`psycopg2-binary`), Apache Kafka, Tweepy, PRAW |
| **Frontend Dashboard** | HTML5, Tailwind CSS, Vanilla JavaScript, Inline SVG Charts |
| **Containerization & QA** | Docker, Docker Compose (PostgreSQL 16 + Apache Kafka 3.7 KRaft), Pytest, Ruff |

---

## Supported Datasets & Training Modes

SpeechShield supports ingesting local starter data, Hugging Face benchmarks, or live social feeds via [scrape_data.py](scripts/scrape_data.py) and training models via [train_model.py](scripts/train_model.py):

- **Recommended Benchmarks**: Jigsaw Toxic Comment Classification, `civil_comments`, HateBenchSet, RealToxicityPrompts, and Cyberbully-Detection-Dataset.
- **Baseline Models (`--mode baseline`)**: Trains multi-label TF-IDF + Logistic Regression (`--baseline-type logreg`) or calibrated SVM (`--baseline-type svm`) with `class_weight="balanced"`, reporting Precision, Recall, and Macro-F1 per category and saving `data/weights/baseline_model.joblib`.
- **Transformer Fine-Tuning (`--mode transformer`)**: Fine-tunes `vinai/bertweet-base` or `roberta-base` with Hugging Face `Trainer` and optional LoRA (`--use-lora`), saving weights to `data/weights/bertweet_toxicity/`.

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- Git & `pip`
- Docker & Docker Compose *(optional, for containerized deployment with PostgreSQL & Kafka)*

### Step 1 — Clone the repository

```bash
git clone https://github.com/samael0-dev/SpeechShield.git
cd SpeechShield
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
```

**Windows (PowerShell):**

```powershell
venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure environment variables

Copy the template environment file:

```bash
cp .env.example .env
```

**Windows PowerShell:**

```powershell
Copy-Item .env.example .env
```

> **Note:** Never commit `.env` files, API bearer tokens, or database credentials. By default, `.env.example` uses local SQLite (`sqlite:///./data/toxicity_logs.db`) and `MODEL_BACKEND=hybrid`.

### Step 5 — Generate starter data & train the baseline model

```bash
python scripts/scrape_data.py --source sample
python scripts/train_model.py --mode baseline --baseline-type logreg
```

*(Optional)* Fine-tune BERTweet with LoRA on a Hugging Face dataset:

```bash
python scripts/scrape_data.py --source huggingface --dataset-name civil_comments --limit 5000
python scripts/train_model.py --mode transformer --transformer-model vinai/bertweet-base --epochs 3 --batch-size 16 --use-lora
```

### Step 6 — Start the FastAPI server & dashboard

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- Moderation Dashboard: `http://localhost:8000/`
- Interactive Swagger UI: `http://localhost:8000/docs`
- ReDoc UI: `http://localhost:8000/redoc`

---

## Docker Deployment

Launch the full stack (**PostgreSQL 16**, **Apache Kafka 3.7 KRaft**, **FastAPI API**, and **Nginx HTML/Tailwind Frontend**) using Docker Compose:

```bash
docker compose up --build
```

- FastAPI Docs: `http://localhost:8000/docs`
- Moderation Dashboard: `http://localhost:8501`

---

## Testing & Code Quality

SpeechShield includes automated unit and FastAPI integration tests covering social text cleaning, obfuscated slur restoration, toxicity prediction, borderline human review workflows, stream simulation, hate-raid spike analytics, and baseline training.

### Run Pytest

```bash
python -m pytest -v
```

### Run Ruff

Check linting across all source, script, frontend, and test modules:

```bash
python -m ruff check src scripts tests frontend
```

---

## Known Limitations

- **Starter Dataset Size:** The built-in `--source sample` generator creates 56 unique starter rows for immediate offline smoke-testing; real production deployments should train on full benchmarks (e.g., Jigsaw, `civil_comments`, or Cyberbully-Detection-Dataset) for calibrated multi-label recall across rare subcategories (`threat`, `severe_toxic`, `identity_hate`).
- **Optional Transformer Hub Downloads:** By default, `ToxicityClassifier` loads local weights from `data/weights/bertweet_toxicity/` without making unprompted Hugging Face network downloads at API startup (`allow_hub_download=False`).
- **`ekphrasis` Python Version Compatibility:** `ekphrasis` is loaded optionally with a safe fallback to the built-in social text cleaner (`SocialTextCleaner`) on environments where `ekphrasis` is not installed.

---

## Project Structure

```text
SpeechShield/
├── data/
│   ├── raw/                             # Raw JSON/CSV dumps from X, Reddit, or public datasets
│   ├── processed/                       # Cleaned datasets ready for training
│   └── weights/                         # Saved baseline (.joblib) & fine-tuned transformer weights
│
├── notebooks/
│   ├── 01_data_exploration.ipynb        # Class balance, platform & length exploration
│   ├── 02_preprocessing_experiments.ipynb
│   └── 03_baseline_model_training.ipynb
│
├── src/
│   ├── api/
│   │   ├── db.py                        # SQLAlchemy engine & ModerationLog model
│   │   ├── main.py                      # FastAPI application entrypoint & lifespan
│   │   ├── routes.py                    # /predict, /review-queue, /analytics/spikes endpoints
│   │   └── schemas.py                   # Pydantic v2 request/response models
│   ├── data_pipeline/
│   │   ├── cleaner.py                   # Emoji, hashtag, contraction & slur cleaning
│   │   └── stream_ingest.py             # Tweepy, PRAW, Kafka & MockSocialStream adapters
│   └── model/
│       ├── classifier.py                # Hybrid fast-pass baseline + BERTweet/RoBERTa classifier
│       └── utils.py                     # Threshold calibration & hate-raid spike detection
│
├── frontend/
│   ├── index.html                       # Tailwind CSS moderation dashboard
│   └── assets/
│       ├── app.js                       # API integration, Dark/Light toggle & SVG charts
│       └── style.css                    # Typography & scrollbar refinements
│
├── scripts/
│   ├── scrape_data.py                   # Dataset ingestion & preprocessing CLI
│   └── train_model.py                   # Baseline & BERTweet/LoRA training CLI
│
├── tests/
│   ├── conftest.py                      # Pytest path configuration
│   ├── test_api.py                      # FastAPI & hybrid classifier integration tests
│   └── test_cleaner.py                  # Social media text cleaner unit tests
│
├── .env.example                         # Environment variable template
├── .gitignore                           # Ignores data/, weights/, .env, and local DBs
├── docker-compose.yml                   # Orchestrates PostgreSQL, Kafka, API & Frontend
├── Dockerfile.api                       # FastAPI container image
├── Dockerfile.frontend                  # Nginx container image for HTML/Tailwind UI
├── requirements.txt                     # Python dependencies
└── README.md
```
