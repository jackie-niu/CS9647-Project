import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import transformers
from packaging import version
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data_prep import load_pheme_csv
from baseline import run_baselines
from eval_utils import summarize_predictions

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=".", help="Project root containing PHEME/ folder")
    parser.add_argument("--pheme_csv", type=str, default="PHEME/pheme_reactions_3col.csv")
    parser.add_argument("--cache_root", type=str, default="/content/drive/MyDrive/pheme_cache")

    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_len", type=int, default=128)  
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--run_baselines", action="store_true")
    args = parser.parse_args()

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    os.chdir(PROJECT_ROOT)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gpu_ok = torch.cuda.is_available()
    print("GPU available:", gpu_ok)

    os.makedirs(args.cache_root, exist_ok=True)
    raw_cache = os.path.join(args.cache_root, "pheme_clean.csv")

    # Load dataset
    if os.path.exists(raw_cache):
        df = pd.read_csv(raw_cache)
        print(f"[data] Loaded cached dataset: {raw_cache} (rows={len(df)})")
    else:
        df = load_pheme_csv(args.root_dir, csv_rel_path=args.pheme_csv, cache_csv=raw_cache)
        print(f"[data] Built and cached dataset: {raw_cache} (rows={len(df)})")

    # Basic sanity
    print("[data] label counts:", df["is_misinformation"].value_counts().to_dict())

    # Split into train/val/test
    y = df["is_misinformation"].to_numpy()
    train_df, temp_df = train_test_split(
        df, test_size=(args.test_size + args.val_size),
        random_state=args.seed, stratify=y
    )
    y_temp = temp_df["is_misinformation"].to_numpy()
    rel_test = args.test_size / (args.test_size + args.val_size)

    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test,
        random_state=args.seed, stratify=y_temp
    )

    print(f"[split] train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print("[split] train label counts:", train_df["is_misinformation"].value_counts().to_dict())

    # Baselines
    if args.run_baselines:
        base = run_baselines(train_df, test_df, text_col="text", label_col="is_misinformation")
        print("\n[baseline] TF-IDF + Logistic Regression\n", base["tfidf_lr"]["report"])
        print("[baseline] LR metrics:", {k: base["tfidf_lr"][k] for k in ["accuracy","precision","recall","f1"]})

        print("\n[baseline] TF-IDF + Linear SVM\n", base["tfidf_svm"]["report"])
        print("[baseline] SVM metrics:", {k: base["tfidf_svm"][k] for k in ["accuracy","precision","recall","f1"]})

    # Save baseline confusion matrices for later analysis
    baseline_dir = os.path.join(args.cache_root, "baseline_outputs")
    os.makedirs(baseline_dir, exist_ok=True)

    np.save(os.path.join(baseline_dir, "lr_confusion_matrix.npy"), base["tfidf_lr"]["confusion_matrix"])
    np.save(os.path.join(baseline_dir, "svm_confusion_matrix.npy"), base["tfidf_svm"]["confusion_matrix"])

    print("[baseline] Saved confusion matrices to:", baseline_dir)

    with open(os.path.join(baseline_dir, "lr_report.txt"), "w") as f:
        f.write(base["tfidf_lr"]["report"])

    with open(os.path.join(baseline_dir, "svm_report.txt"), "w") as f:
        f.write(base["tfidf_svm"]["report"])

    # Transformer-based model fine-tuning
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

    # HuggingFace dataset expects labels column = "labels" later
    train_hf = Dataset.from_pandas(train_df[["text", "is_misinformation", "source_text"]].rename(columns={"is_misinformation": "label"}).reset_index(drop=True))
    val_hf   = Dataset.from_pandas(val_df[["text", "is_misinformation", "source_text"]].rename(columns={"is_misinformation": "label"}).reset_index(drop=True))
    test_hf  = Dataset.from_pandas(test_df[["text", "is_misinformation", "source_text"]].rename(columns={"is_misinformation": "label"}).reset_index(drop=True))

    train_tok = train_hf.map(tok, batched=True).remove_columns(["text", "source_text"]).rename_column("label", "labels")
    val_tok   = val_hf.map(tok, batched=True).remove_columns(["text", "source_text"]).rename_column("label", "labels")
    test_tok  = test_hf.map(tok, batched=True).remove_columns(["text", "source_text"]).rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    run_dir = os.path.join(args.cache_root, "runs", f"{args.model.replace('/','_')}_pheme")
    os.makedirs(run_dir, exist_ok=True)

    # Transformers version compatibility for eval arg
    
    use_new = version.parse(transformers.__version__) >= version.parse("4.46.0")

    kwargs = dict(
        output_dir=run_dir,
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
    if use_new:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Test evaluation and error analysis
    pred = trainer.predict(test_tok)
    logits = pred.predictions
    y_true = test_df["is_misinformation"].to_numpy()
    y_pred = np.argmax(logits, axis=1)

    cm, report, err_view = summarize_predictions(
        test_df.reset_index(drop=True),
        y_true, y_pred,
        text_col="text",
        k=30
    )

    print("\n[transformer] Confusion matrix:\n", cm)
    print("\n[transformer] Classification report:\n", report)

    err_path = os.path.join(run_dir, "misclassified_examples.csv")
    err_view.to_csv(err_path, index=False)
    print(f"\n[error-analysis] Saved misclassified examples to: {err_path}")


if __name__ == "__main__":
    main()