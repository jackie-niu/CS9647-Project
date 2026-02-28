import os
import pandas as pd
from tqdm.auto import tqdm
from newspaper import Article

def fetch_article_text(url: str) -> str | None:
    # Fetch article text using newspaper. Returns None if fetching/parsing fails or text is too short
    try:
        a = Article(url)
        a.download()
        a.parse()
        txt = ((a.title or "") + "\n" + (a.text or "")).strip()
        if len(txt) < 50:
            return None
        return txt
    except Exception:
        return None

def align_to_sample(scraped: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    # Align scraped data to the sample by URL. If 'text' column is missing in scraped, add it as NA
    if "news_url" not in sample.columns:
        raise ValueError("Sample must contain 'news_url' column.")
    if "text" not in scraped.columns:
        scraped = scraped.copy()
        scraped["text"] = pd.NA
    scraped_urls = scraped[["news_url", "text"]].drop_duplicates()
    merged = sample.merge(scraped_urls, on="news_url", how="left")
    return merged

def scrape_with_resume(sample: pd.DataFrame, scraped_cache_path: str, partial_cache_path: str, save_every: int = 50,) -> pd.DataFrame:
    # Scrape article texts for the given sample, with resume capability 
    # Uses scraped_cache_path for final cache and partial_cache_path for intermediate saves
    if os.path.exists(scraped_cache_path):
        print(f"[scrape] Loading fully scraped cache: {scraped_cache_path}")
        scraped = pd.read_parquet(scraped_cache_path)
        scraped = align_to_sample(scraped, sample)
        return scraped.dropna(subset=["text"]).reset_index(drop=True)

    if os.path.exists(partial_cache_path):
        print(f"[scrape] Resuming partial scrape: {partial_cache_path}")
        scraped = pd.read_parquet(partial_cache_path)
        scraped = align_to_sample(scraped, sample)
    else:
        scraped = sample.copy()
        scraped["text"] = pd.NA

    # Identify missing texts to scrape
    missing_idx = scraped.index[scraped["text"].isna()].tolist()
    print(f"[scrape] Need to scrape {len(missing_idx)} / {len(scraped)}")

    # Scrape missing articles with progress bar and periodic saves
    for k, i in enumerate(tqdm(missing_idx, desc="Scraping")):
        news_url = scraped.at[i, "news_url"]
        scraped.at[i, "text"] = fetch_article_text(news_url)

        if (k + 1) % save_every == 0:
            scraped.to_parquet(partial_cache_path, index=False)

    # Finalize
    scraped_clean = scraped.dropna(subset=["text"]).reset_index(drop=True)
    scraped_clean.to_parquet(scraped_cache_path, index=False)
    print(f"[scrape] Saved scraped cache: {scraped_cache_path} (rows={len(scraped_clean)})")

    return scraped_clean