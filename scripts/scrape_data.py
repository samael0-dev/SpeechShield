"""Batch data ingestion and social text preprocessing script.

Supports:
- `sample`: Creates a balanced starter social-media multi-label dataset locally.
- `huggingface`: Downloads public toxicity/hate-speech benchmarks via `datasets`.
- `reddit`: Pulls live subreddit comments via PRAW (requires credentials).
- `twitter`: Pulls recent tweets via Tweepy (requires bearer token).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.cleaner import SocialTextCleaner
from src.data_pipeline.stream_ingest import SocialStreamIngestor
from src.model.utils import TOXICITY_CATEGORIES

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def build_starter_dataset() -> pd.DataFrame:
    """Create a diverse, balanced starter social-media dataset with unique samples."""
    rows: list[dict[str, object]] = [
        # Clean / friendly / gaming banter (non-toxic)
        {"platform": "x", "text": "GG everyone! That final round was insane 🙌🔥", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "reddit", "text": "Thanks for sharing this benchmark, super helpful for my research.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "youtube", "text": "Awesome tutorial ❤️ really cleared up how transformers work!", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "x", "text": "Honestly this patch broke my favorite combo smh, hope they fix it soon.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "reddit", "text": "I disagree with your point on the architecture, FastAPI scales better here imo.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "x", "text": "Bro you got cooked in that 1v1 lol, rematch tomorrow! 😄 #GamingNight", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "youtube", "text": "Could you share the GitHub repo link in the description?", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "reddit", "text": "Solid analysis, though I think the latency numbers need another pass.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "x", "text": "Happy Friday folks! Wishing everyone a relaxing weekend 🌟", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "youtube", "text": "This soundtrack slaps so hard, listening on repeat.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "reddit", "text": "Have you tried calibrating the threshold before deploying?", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "x", "text": "Not a fan of the UI redesign tbh, felt cleaner before.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "x", "text": "Just pushed a clean PR for the streaming consumer #OpenSource", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "reddit", "text": "Great writeup! Appreciate the reproducible evaluation metrics.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "youtube", "text": "Subscribed immediately, your explanation of attention heads is crystal clear.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "x", "text": "Coffee first, debugging Docker compose second ☕💻", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "reddit", "text": "Anyone else seeing higher recall with class_weight balanced?", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "youtube", "text": "That comeback in overtime gave me chills, well played both squads!", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "x", "text": "Friendly reminder to back up your model weights before retraining ✨", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "reddit", "text": "I prefer Streamlit for fast prototypes and Next.js for customer dashboards.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "x", "text": "Wait that plot twist in episode 5 actually caught me off guard 😂", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "youtube", "text": "Thanks for adding timestamps in the pinned comment, super convenient!", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "reddit", "text": "The new dataset release covers 34 identity groups which helps nuance testing.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"platform": "x", "text": "Congrats on shipping v1.0! Proud of the whole team 🚀", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        # Insult / cyberbullying
        {"platform": "x", "text": "You are such a pathetic clown 🤡 nobody cares what you think #TrashTake", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        {"platform": "youtube", "text": "How can someone be this stupid and clueless? Total moron.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        {"platform": "reddit", "text": "Your code is absolute garbage and you are an idiot for posting it.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        {"platform": "x", "text": "Look at this loser crying online again, embarrassing af.", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
        {"platform": "youtube", "text": "What a dumb take from a hopeless idiot, delete your channel.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        {"platform": "reddit", "text": "Stop acting like an ignorant clown, your arguments are pure trash.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        {"platform": "x", "text": "Nobody likes you, pathetic wannabe influencer 🤡", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        {"platform": "youtube", "text": "You look ugly and sound completely brainless on camera.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        # Obscene / obfuscated slurs
        {"platform": "x", "text": "Shut up you @ss $lut nobody asked for your garbage opinion", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
        {"platform": "youtube", "text": "stfu you f*cking b!tch gtfo of this thread", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
        {"platform": "reddit", "text": "What a load of sh!tty bullshit, you dumb @sshole", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
        {"platform": "x", "text": "f u c k off you worthless piece of trash", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
        {"platform": "youtube", "text": "Get out of here you wh0re, nobody wants your b!tching", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
        {"platform": "reddit", "text": "Stop talking sh!t you absolute d1ckhead, nobody buys your lies", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
        {"platform": "x", "text": "You are a f**king pathetic c*nt, shut your mouth already", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
        {"platform": "youtube", "text": "Nobody cares about this a$$wipe and his stupid bullshit video", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
        # Threats / severe toxicity
        {"platform": "x", "text": "I will find where you live and hurt you, watch your back", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
        {"platform": "youtube", "text": "Just go kill yourself already, kys nobody wants you alive", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 1, "identity_hate": 0},
        {"platform": "x", "text": "Someone ought to shoot you on sight, delete your account or die", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
        {"platform": "reddit", "text": "Hope you burn in hell, I am going to stab anyone defending this", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
        {"platform": "x", "text": "K!ll y0urself tonight, the world would be better without you", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 1, "identity_hate": 0},
        {"platform": "youtube", "text": "I swear I will hunt you down and beat you senseless", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
        {"platform": "reddit", "text": "Go hang yourself loser, you deserve to die painfully", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 1, "identity_hate": 0},
        {"platform": "x", "text": "Wait until I catch you outside, I am going to murder you", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
        # Identity hate
        {"platform": "x", "text": "Go back to where you came from #IdentityHate we do not want your kind here", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 1},
        {"platform": "reddit", "text": "All people from that religion are scum and should be banned from society", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 1},
        {"platform": "youtube", "text": "Typical r3tard behavior from your kind, disgusting xenophobe garbage", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 1},
        {"platform": "x", "text": "Shut up you f@ggot, your community ruins everything #HateSpeech", "toxic": 1, "severe_toxic": 1, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 1},
        {"platform": "reddit", "text": "People of your race are inferior scum and do not belong in this country", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 1},
        {"platform": "youtube", "text": "Women do not belong in tech, go back to the kitchen stupid @ss", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 1},
        {"platform": "x", "text": "Get these immigrants out of our city, filthy parasites #GoHome", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 1},
        {"platform": "reddit", "text": "Religious zealots like your group are toxic garbage ruining the forum", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 1},
    ]
    return pd.DataFrame(rows)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply `SocialTextCleaner` across all text rows and ensure label columns exist."""
    cleaner = SocialTextCleaner(use_ekphrasis=True)
    cleaned_texts: list[str] = []
    sentiments: list[float] = []
    for text in df["text"].astype(str):
        res = cleaner.clean(text)
        cleaned_texts.append(res.cleaned_text)
        sentiments.append(res.sentiment_score)

    out = df.copy()
    out["cleaned_text"] = cleaned_texts
    out["sentiment_score"] = sentiments
    for cat in TOXICITY_CATEGORIES:
        if cat not in out.columns:
            out[cat] = 0
        out[cat] = out[cat].fillna(0).astype(int)
    return out


def fetch_huggingface_dataset(dataset_name: str, limit: int) -> pd.DataFrame:
    """Load a Hugging Face toxicity/hate-speech dataset and normalize schema."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=f"train[:{limit}]")
    df = ds.to_pandas()
    if "comment_text" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"comment_text": "text"})
    if "tweet" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"tweet": "text"})
    if "label" in df.columns and "toxic" not in df.columns:
        df["toxic"] = (df["label"].astype(int) > 0).astype(int)
    df["platform"] = dataset_name
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape, load, and preprocess social toxicity data.")
    parser.add_argument(
        "--source",
        choices=["sample", "huggingface", "reddit", "twitter"],
        default="sample",
        help="Data ingestion source.",
    )
    parser.add_argument("--dataset-name", default="civil_comments", help="Hugging Face dataset ID.")
    parser.add_argument("--query", default="gaming", help="Reddit subreddit or Twitter search query.")
    parser.add_argument("--limit", type=int, default=2000, help="Maximum number of rows to fetch.")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.source == "sample":
        raw_df = build_starter_dataset()
        raw_path = RAW_DIR / "sample_social_comments.csv"
    elif args.source == "huggingface":
        raw_df = fetch_huggingface_dataset(args.dataset_name, args.limit)
        raw_path = RAW_DIR / f"{args.dataset_name.replace('/', '_')}.csv"
    elif args.source == "reddit":
        events = SocialStreamIngestor().fetch_reddit_comments(args.query, limit=args.limit)
        raw_df = pd.DataFrame([{"platform": e.platform, "author": e.author_handle, "post_id": e.post_id, "text": e.text} for e in events])
        raw_path = RAW_DIR / f"reddit_{args.query}.csv"
    else:
        events = SocialStreamIngestor().fetch_twitter_recent(args.query, max_results=args.limit)
        raw_df = pd.DataFrame([{"platform": e.platform, "author": e.author_handle, "post_id": e.post_id, "text": e.text} for e in events])
        raw_path = RAW_DIR / "twitter_recent.csv"

    raw_df.to_csv(raw_path, index=False)
    processed_df = preprocess_dataframe(raw_df)
    processed_path = PROCESSED_DIR / "cleaned_social_comments.csv"
    processed_df.to_csv(processed_path, index=False)
    print(f"Saved raw dataset ({len(raw_df)} rows) -> {raw_path}")
    print(f"Saved cleaned dataset ({len(processed_df)} rows) -> {processed_path}")


if __name__ == "__main__":
    main()
