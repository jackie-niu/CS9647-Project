import os
import pandas as pd


def load_pheme_csv(root_dir: str, csv_rel_path: str = "PHEME/pheme_reactions_3col.csv",
                   cache_csv: str | None = None) -> pd.DataFrame:

    # Load the PHEME dataset from a CSV file, with some cleaning and validation.
    csv_path = os.path.join(root_dir, csv_rel_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find PHEME CSV at: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"PHEME CSV missing columns: {missing}. Found: {list(df.columns)}")

    out = pd.DataFrame()
    out["text"] = df["text"].fillna("").astype(str).str.strip()

    # source text (keeping optional for now)
    # out["source_text"] = df["source_text"].fillna("").astype(str).str.strip() if "source_text" in df.columns else ""

    # label is already 0/1 after create_pheme.ipynb processing
    out["is_misinformation"] = pd.to_numeric(df["label"], errors="coerce")

    # clean just in case, and ensure it's int
    out = out.dropna(subset=["is_misinformation"]).copy()
    out["is_misinformation"] = out["is_misinformation"].astype(int)

    # keep only 0/1
    out = out[out["is_misinformation"].isin([0, 1])].copy()

    # drop empty text
    out = out[out["text"].str.len() > 0].reset_index(drop=True)

    if cache_csv:
        os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
        out.to_csv(cache_csv, index=False)

    return out