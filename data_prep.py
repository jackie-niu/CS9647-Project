import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_or_build_raw_fakenewsnet(raw_csv_cache_path: str) -> pd.DataFrame:
    # Load raw dataset from cache if exists, else build from FakeNewsNet repo CSVs and cache it
    if os.path.exists(raw_csv_cache_path):
        print(f"[data] Loading cached raw dataset: {raw_csv_cache_path}")
        return pd.read_csv(raw_csv_cache_path)

    print("[data] Building raw dataset from FakeNewsNet repo CSVs...")
    
    # Expect repo already cloned in current working dir as ./FakeNewsNet
    base = "FakeNewsNet/dataset"
    files = {
        ("politifact", "fake"): f"{base}/politifact_fake.csv",
        ("politifact", "real"): f"{base}/politifact_real.csv",
        ("gossipcop", "fake"):  f"{base}/gossipcop_fake.csv",
        ("gossipcop", "real"):  f"{base}/gossipcop_real.csv",
    }

    # Load each CSV, add source and label columns, and concatenate
    dfs = []
    for (source, kind), path in files.items():
        df = pd.read_csv(path)
        df["source"] = source
        df["label"] = 1 if kind == "fake" else 0
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    # Keep only relevant columns if they exist, and save to cache
    keep_cols = [c for c in ["id", "title", "url", "tweet_ids", "source", "label"] if c in data.columns]
    data = data[keep_cols].copy()

    data.to_csv(raw_csv_cache_path, index=False)
    print(f"[data] Saved raw dataset: {raw_csv_cache_path} (rows={len(data)})")
    return data

def filter_and_sample(data: pd.DataFrame, subset: str, n: int, seed: int) -> pd.DataFrame:
    # Filter dataset by subset and sample n rows if n > 0 and less than total. Subset can be 'politifact', 'gossipcop', or 'both'
    df = data.copy()
    if subset in ("politifact", "gossipcop"):
        df = df[df["source"] == subset].copy()
    elif subset != "both":
        raise ValueError("subset must be 'politifact', 'gossipcop', or 'both'.")

    # Sample if requested
    if n > 0 and n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # must have url
    df = df.dropna(subset=["url"]).reset_index(drop=True)
    return df

# Split the dataset into train/val/test with stratification on label, using the provided seed for reproducibility
def split_train_val_test(df: pd.DataFrame, seed: int):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)