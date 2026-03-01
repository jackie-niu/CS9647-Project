import os
import argparse
import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from data_prep import load_fakenewsnet_kaggle
from baseline import run_baselines
from eval_utils import summarize_predictions

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=".", help="Project root containing FakeNewsNet/ folder")
    parser.add_argument("--cache_root", type=str, default="/content/drive/MyDrive/fakenewsnet_cache")

    parser.add_argument("--use_dataset", type=str, default="both", choices=["both", "BuzzFeed", "PolitiFact"])
    parser.add_argument("--text_col", type=str, default="combined_text", choices=["combined_text", "text", "title"])

    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--run_baselines", action="store_true")

    args = parser.parse_args()

    # Ensure relative paths resolve from this file’s directory
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    os.chdir(PROJECT_ROOT)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gpu_ok = torch.cuda.is_available()
    print("GPU available:", gpu_ok)

    os.makedirs(args.cache_root, exist_ok=True)
    raw_cache = os.path.join(args.cache_root, "kaggle_fakenewsnet_combined.csv")

    # -------- Load dataset (cached) --------
    if os.path.exists(raw_cache):
        df = pd.read_csv(raw_cache)
        print(f"[data] Loaded cached dataset: {raw_cache} (rows={len(df)})")
    else:
        df = load_fakenewsnet_kaggle(args.root_dir, cache_csv=raw_cache)
        print(f"[data] Built and cached dataset: {raw_cache} (rows={len(df)})")

    if args.use_dataset != "both":
        df = df[df["dataset"] == args.use_dataset].reset_index(drop=True)

    # Some rows may still have empty text_col after cleaning; drop just in case
    df[args.text_col] = df[args.text_col].fillna("").astype(str)
    df = df[df[args.text_col].str.strip().str.len() > 0].reset_index(drop=True)

    # -------- Split train/val/test (stratified) --------
    y = df["label"].to_numpy()
    train_df, temp_df = train_test_split(
        df, test_size=(args.test_size + args.val_size),
        random_state=args.seed, stratify=y
    )
    y_temp = temp_df["label"].to_numpy()
    rel_test = args.test_size / (args.test_size + args.val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test,
        random_state=args.seed, stratify=y_temp
    )

    print(f"[split] train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print("[split] train label counts:", train_df["label"].value_counts().to_dict())

    # -------- Baselines --------
    if args.run_baselines:
        base = run_baselines(train_df, test_df, text_col=args.text_col)
        print("\n[baseline] TF-IDF + Logistic Regression\n", base["tfidf_lr"]["report"])
        print("[baseline] LR metrics:", {k: base["tfidf_lr"][k] for k in ["accuracy","precision","recall","f1"]})

        print("\n[baseline] TF-IDF + Linear SVM\n", base["tfidf_svm"]["report"])
        print("[baseline] SVM metrics:", {k: base["tfidf_svm"][k] for k in ["accuracy","precision","recall","f1"]})

    # -------- Transformer fine-tuning --------
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_len
        )

    # HuggingFace datasets need column named "text"
    train_hf = Dataset.from_pandas(train_df[[args.text_col, "label"]].rename(columns={args.text_col: "text"}).reset_index(drop=True))
    val_hf   = Dataset.from_pandas(val_df[[args.text_col, "label"]].rename(columns={args.text_col: "text"}).reset_index(drop=True))
    test_hf  = Dataset.from_pandas(test_df[[args.text_col, "label"]].rename(columns={args.text_col: "text"}).reset_index(drop=True))

    train_tok = train_hf.map(tok, batched=True).remove_columns(["text"]).rename_column("label", "labels")
    val_tok   = val_hf.map(tok, batched=True).remove_columns(["text"]).rename_column("label", "labels")
    test_tok  = test_hf.map(tok, batched=True).remove_columns(["text"]).rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    run_dir = os.path.join(args.cache_root, "runs", f"{args.model.replace('/','_')}_{args.use_dataset}_{args.text_col}")
    os.makedirs(run_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=run_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=gpu_ok,
        report_to="none",
        save_total_limit=2,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # -------- Evaluate + error analysis --------
    pred = trainer.predict(test_tok)
    logits = pred.predictions
    y_true = test_df["label"].to_numpy()
    y_pred = np.argmax(logits, axis=1)

    cm, report, err_view = summarize_predictions(
        test_df.reset_index(drop=True),
        y_true, y_pred,
        text_col=args.text_col,
        k=25
    )

    print("\n[transformer] Confusion matrix:\n", cm)
    print("\n[transformer] Classification report:\n", report)

    err_path = os.path.join(run_dir, "misclassified_examples.csv")
    err_view.to_csv(err_path, index=False)
    print(f"\n[error-analysis] Saved misclassified examples to: {err_path}")


if __name__ == "__main__":
    main()