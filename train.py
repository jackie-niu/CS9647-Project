import os
import argparse
import pandas as pd

import torch
import evaluate
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from config import DEFAULT_CACHE_ROOT, paths
from data_prep import load_or_build_raw_fakenewsnet, filter_and_sample, split_train_val_test
from scrape_cache import scrape_with_resume
from baseline import run_baselines
from eval_utils import hf_compute_metrics_factory, print_classification

# Ensure repo is cloned before data prep to access CSVs, and before scrape to save cache in repo dir. 
def ensure_repo_cloned():
    if not os.path.exists("FakeNewsNet"):
        print("[setup] Cloning FakeNewsNet repo...")
        os.system("rm -rf FakeNewsNet")
        os.system("git clone -q https://github.com/KaiDMML/FakeNewsNet.git")

# Convert a DataFrame with 'text' and 'label' columns to a HuggingFace Dataset
def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))

# Tokenize the datasets with the specified tokenizer and max length, and cache the tokenized datasets to disk for future runs. If cached versions exist, load from disk instead of re-tokenizing.
def tokenize_and_cache(train_df, val_df, test_df, tokenizer_name, max_len, tokenized_root):
    safe_name = tokenizer_name.replace("/", "_")
    base_dir = os.path.join(tokenized_root, f"{safe_name}_len{max_len}")
    train_path = os.path.join(base_dir, "train")
    val_path   = os.path.join(base_dir, "val")
    test_path  = os.path.join(base_dir, "test")

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print("[tok] Loading tokenized datasets from cache...")
        train_tok = load_from_disk(train_path).with_format("torch")
        val_tok   = load_from_disk(val_path).with_format("torch")
        test_tok  = load_from_disk(test_path).with_format("torch")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return tokenizer, train_tok, val_tok, test_tok

    print("[tok] Tokenizing and caching datasets...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Define tokenization function for mapping over the datasets
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_len)

    train_tok = to_hf_dataset(train_df).map(tok, batched=True).remove_columns(["text"]).with_format("torch")
    val_tok   = to_hf_dataset(val_df).map(tok, batched=True).remove_columns(["text"]).with_format("torch")
    test_tok  = to_hf_dataset(test_df).map(tok, batched=True).remove_columns(["text"]).with_format("torch")

    os.makedirs(base_dir, exist_ok=True)
    train_tok.save_to_disk(train_path)
    val_tok.save_to_disk(val_path)
    test_tok.save_to_disk(test_path)
    print(f"[tok] Saved tokenized datasets to: {base_dir}")

    return tokenizer, train_tok, val_tok, test_tok

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", type=str, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--subset", type=str, default="both", choices=["politifact", "gossipcop", "both"])
    parser.add_argument("--n", type=int, default=5000, help="Number of rows to sample (0 = all)")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_len", type=int, default=256)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_bs", type=int, default=16)
    parser.add_argument("--eval_bs", type=int, default=32)

    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--run_baselines", action="store_true")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    # Sanity check GPU availability and print GPU info if available
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    p = paths(args.cache_root)
    os.makedirs(p["tokenized_root"], exist_ok=True)
    os.makedirs(p["ckpt_root"], exist_ok=True)

    ensure_repo_cloned()

    # 1) Raw dataset
    raw = load_or_build_raw_fakenewsnet(p["raw_csv"])
    sample = filter_and_sample(raw, subset=args.subset, n=args.n, seed=args.seed)

    # 2) Scrape
    scraped = scrape_with_resume(
        sample=sample,
        scraped_cache_path=p["scraped_parquet"],
        partial_cache_path=p["scrape_partial"],
        save_every=50,
    )

    # 3) Split
    train_df, val_df, test_df = split_train_val_test(scraped, seed=args.seed)
    print(f"[split] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # Baseline
    if args.run_baselines:
        base_results = run_baselines(train_df, test_df)
        print("\n[baseline results]")
        for k, v in base_results.items():
            print(k, v)

    # 4) Tokenize (cached)
    tokenizer, train_tok, val_tok, test_tok = tokenize_and_cache(
        train_df, val_df, test_df,
        tokenizer_name=args.model,
        max_len=args.max_len,
        tokenized_root=p["tokenized_root"],
    )

    # 5) Train with HuggingFace Trainer 
    metric_f1 = evaluate.load("f1")
    metric_acc = evaluate.load("accuracy")
    metric_precision = evaluate.load("precision")
    metric_recall = evaluate.load("recall")
    compute_metrics = hf_compute_metrics_factory(metric_acc, metric_f1, metric_precision, metric_recall)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    safe_model = args.model.replace("/", "_")
    run_dir = os.path.join(p["ckpt_root"], f"{safe_model}_len{args.max_len}_{args.subset}_n{args.n}")
    os.makedirs(run_dir, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=run_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_total_limit=args.save_total_limit,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print(f"[train] checkpoints: {run_dir}")
    trainer.train(resume_from_checkpoint=args.resume if args.resume else None)

    print("\n[test eval]")
    test_metrics = trainer.evaluate(test_tok)
    print(test_metrics)

    # 6) Error analysis
    pred = trainer.predict(test_tok)
    y_true = pred.label_ids
    y_pred = pred.predictions.argmax(axis=1)
    print_classification(y_true, y_pred)

    # 7) Export best model
    export_dir = os.path.join(run_dir, "best_model_export")
    trainer.save_model(export_dir)
    tokenizer.save_pretrained(export_dir)
    print(f"[export] Saved model+tokenizer to: {export_dir}")

if __name__ == "__main__":
    main()