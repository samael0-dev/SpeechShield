# Social Media Toxicity Detector

An end-to-end AI/ML moderation system tailored for high-velocity social media text (X/Twitter, Reddit, YouTube, and community chats). It combines social media dialect preprocessing, a fast-pass traditional ML filter, fine-tuned BERTweet/RoBERTa inference, asynchronous FastAPI endpoints, SQLAlchemy logging, and a Streamlit moderation dashboard with hate-raid spike detection and a 60%–80% borderline human review queue.

## Project Structure

```text
├── data/
│   ├── raw/                        # Raw JSON/CSV dumps from X/Reddit/datasets
│   ├── processed/                  # Cleaned datasets ready for training
│   └── weights/                    # Saved baseline (.joblib) & transformer weights
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_experiments.ipynb
│   └── 03_baseline_model_training.ipynb
├── src/
│   ├── api/                        # FastAPI server, Pydantic schemas, DB logging
│   ├── data_pipeline/              # Social text cleaner (emoji, hashtags, slurs) & stream adapters
│   └── model/                      # Hybrid classifier & threshold calibration
├── frontend/                       # Streamlit moderation dashboard & Altair charts
├── scripts/                        # Data ingestion & model training scripts
├── tests/                          # Unit & integration tests
├── .env.example
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.frontend
└── requirements.txt
```

## Quick Start (Local Development)

1. **Configure environment variables** (optional; defaults to SQLite + hybrid inference):
   ```bash
   cp .env.example .env
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate starter data & train the fast-pass baseline model**:
   ```bash
   python scripts/scrape_data.py --source sample
   python scripts/train_model.py --mode baseline --baseline-type logreg
   ```

4. **Start the FastAPI backend**:
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```

5. **Start the Streamlit moderation dashboard** (in a second terminal):
   ```bash
   streamlit run frontend/app.py
   ```

## Fine-Tuning BERTweet / RoBERTa

To fine-tune `vinai/bertweet-base` (or `roberta-base` / `cardiffnlp/twitter-roberta-base-hate-latest`) with optional LoRA (`--use-lora`):

```bash
python scripts/train_model.py \
  --mode transformer \
  --transformer-model vinai/bertweet-base \
  --epochs 3 \
  --batch-size 16 \
  --use-lora
```

Saved weights in `data/weights/bertweet_toxicity/` are automatically picked up by `ToxicityClassifier` in `hybrid` and `transformer` modes.

## API Endpoints

- `GET /health` — Service status, database connectivity, and loaded backends.
- `POST /predict` — Async classification for a single comment.
- `POST /predict/batch` — Batch classification with hate-raid spike alerts.
- `POST /ingest/simulate` — Replay sample social media events into the pipeline.
- `GET /logs` — Retrieve stored moderation history.
- `GET /review-queue` — List borderline comments (60%–80% confidence) awaiting human review.
- `POST /review/{log_id}` — Submit human moderator approval or block decision.
- `GET /analytics/spikes` — Aggregated category counts and active hate-raid alerts.

## Running Tests

```bash
pytest -v
```

## Docker Deployment

```bash
docker compose up --build
```

- FastAPI docs: `http://localhost:8000/docs`
- Streamlit dashboard: `http://localhost:8501`
