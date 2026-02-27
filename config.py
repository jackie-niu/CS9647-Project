import os

DEFAULT_CACHE_ROOT = "/content/drive/MyDrive/fakenewsnet_cache"

# Cache locations
def paths(cache_root: str):
    os.makedirs(cache_root, exist_ok=True)
    return {
        "raw_csv": os.path.join(cache_root, "fakenewsnet_raw.csv"),
        "scraped_parquet": os.path.join(cache_root, "fakenewsnet_scraped.parquet"),
        "scrape_partial": os.path.join(cache_root, "fakenewsnet_scrape_partial.parquet"),
        "tokenized_root": os.path.join(cache_root, "tokenized"),
        "ckpt_root": os.path.join(cache_root, "model_ckpts"),
    }