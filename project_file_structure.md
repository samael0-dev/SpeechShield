# Social Media Toxicity Detector - File Structure

```text
social-toxicity-detector/
├── data/                       # Local data storage (Add to .gitignore)
│   ├── raw/                    # Raw JSON/CSV dumps from Twitter/Reddit
│   ├── processed/              # Cleaned, tokenized datasets ready for training
│   └── weights/                # Saved model weights (e.g., fine-tuned BERTweet)
│
├── notebooks/                  # Jupyter notebooks for experimentation
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_experiments.ipynb
│   └── 03_baseline_model_training.ipynb
│
├── src/                        # Core backend and ML logic
│   ├── __init__.py
│   ├── api/                    # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI instance and server startup
│   │   ├── routes.py           # Endpoints (e.g., /predict, /health)
│   │   └── schemas.py          # Pydantic models for API request/response validation
│   │
│   ├── data_pipeline/          # Social media text processing
│   │   ├── __init__.py
│   │   ├── cleaner.py          # ekphrasis and emoji translation logic
│   │   └── stream_ingest.py    # Kafka/RabbitMQ or Tweepy/PRAW ingestion scripts
│   │
│   └── model/                  # Machine Learning inference and loading
│       ├── __init__.py
│       ├── classifier.py       # Class to load BERTweet/RoBERTa and run predictions
│       └── utils.py            # Helper functions for confidence thresholding
│
├── frontend/                   # User Interface (Streamlit or Next.js)
│   ├── app.py                  # Streamlit main application file (if using Streamlit)
│   ├── components/             # Reusable UI components
│   └── assets/                 # Images, custom CSS
│
├── scripts/                    # Utility scripts for batch jobs
│   ├── train_model.py          # Script to fine-tune the transformer model
│   └── scrape_data.py          # Script to batch-download historical data
│
├── tests/                      # Unit and integration tests
│   ├── test_api.py             # Test FastAPI endpoints
│   └── test_cleaner.py         # Test social media slang and emoji parser
│
├── .env.example                # Template for environment variables (API keys, DB URIs)
├── .gitignore                  # Files to exclude from version control (data/, .env, __pycache__)
├── docker-compose.yml          # Orchestrates the API, Frontend, and message broker (Kafka)
├── Dockerfile.api              # Docker configuration for the FastAPI backend
├── Dockerfile.frontend         # Docker configuration for the Dashboard
├── requirements.txt            # Python dependencies (PyTorch, FastAPI, transformers, ekphrasis, etc.)
└── README.md                   # Project documentation and setup instructions
```

### Directory Highlights:
*   **`src/data_pipeline/`**: This is where your specialized social media tools live. `cleaner.py` will handle unpacking hashtags and converting emojis before the text is sent to the model.
*   **`src/api/`**: Contains your FastAPI application, which will asynchronously process incoming text streams and return toxicity scores.
*   **`data/`**: Kept locally for training and testing but should be excluded from Git to avoid uploading massive datasets or proprietary model weights.