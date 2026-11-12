FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY --chown=user:user src ./src
COPY --chown=user:user scripts ./scripts
COPY --chown=user:user frontend ./frontend
COPY --chown=user:user data ./data

RUN mkdir -p /app/data/raw /app/data/processed /app/data/weights \
    && chown -R user:user /app/data

USER user

RUN python scripts/scrape_data.py --source sample \
    && python scripts/train_model.py --mode baseline

EXPOSE 7860

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
