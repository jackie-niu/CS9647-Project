import os
import pandas as pd

REQUIRED_COLS = ["id", "title", "text", "url", "source", "publish_date"]

def load_news_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure required columns exist
    missing = [c for c in ["id", "title", "text", "url", "source"] if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing}. Found: {list(df.columns)}")

    # Normalize types
    df["title"] = df["title"].fillna("").astype(str)
    df["text"]  = df["text"].fillna("").astype(str)
    df["url"]   = df["url"].fillna("").astype(str)
    df["source"] = df["source"].fillna("").astype(str)

    # Combine title + text (main model input)
    df["combined_text"] = (df["title"].str.strip() + "\n\n" + df["text"].str.strip()).str.strip()

    # Drop empty combined text
    df = df[df["combined_text"].str.len() > 0].reset_index(drop=True)
    return df


def load_fakenewsnet_kaggle(root_dir: str, cache_csv: str | None = None) -> pd.DataFrame:
    # Define file paths for the 4 CSVs (fake/real for BuzzFeed and PolitiFact)
    files = {
        ("BuzzFeed", 1): os.path.join(root_dir, "FakeNewsNet", "BuzzFeed_fake_news_content.csv"),
        ("BuzzFeed", 0): os.path.join(root_dir, "FakeNewsNet", "BuzzFeed_real_news_content.csv"),
        ("PolitiFact", 1): os.path.join(root_dir, "FakeNewsNet", "PolitiFact_fake_news_content.csv"),
        ("PolitiFact", 0): os.path.join(root_dir, "FakeNewsNet", "PolitiFact_real_news_content.csv"),
    }

    frames = []
    for (dataset, label), path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

        df = load_news_csv(path)
        df["dataset"] = dataset
        df["label"] = int(label)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    # Keep only relevant columns
    keep = [c for c in ["dataset", "id", "title", "text", "combined_text", "url", "source", "publish_date", "label"] if c in out.columns]
    out = out[keep].copy()

    if cache_csv:
        os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
        out.to_csv(cache_csv, index=False)

    return out