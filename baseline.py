import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score


def run_baselines(train_df, test_df, text_col="combined_text"):
    X_train = train_df[text_col].fillna("").astype(str).tolist()
    X_test  = test_df[text_col].fillna("").astype(str).tolist()
    y_train = train_df["label"].to_numpy()
    y_test  = test_df["label"].to_numpy()

    if len(np.unique(y_train)) < 2:
        raise ValueError(f"Baseline training split has only one class: {np.unique(y_train)}")

    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)

    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=3000, n_jobs=-1, class_weight="balanced")
    lr.fit(Xtr, y_train)
    pred_lr = lr.predict(Xte)
    results["tfidf_lr"] = {
        "accuracy": float(accuracy_score(y_test, pred_lr)),
        "precision": float(precision_score(y_test, pred_lr)),
        "recall": float(recall_score(y_test, pred_lr)),
        "f1": float(f1_score(y_test, pred_lr)),
        "report": classification_report(y_test, pred_lr, digits=4),
    }

    # Linear SVM
    svm = LinearSVC(class_weight="balanced")
    svm.fit(Xtr, y_train)
    pred_svm = svm.predict(Xte)
    results["tfidf_svm"] = {
        "accuracy": float(accuracy_score(y_test, pred_svm)),
        "precision": float(precision_score(y_test, pred_svm)),
        "recall": float(recall_score(y_test, pred_svm)),
        "f1": float(f1_score(y_test, pred_svm)),
        "report": classification_report(y_test, pred_svm, digits=4),
    }

    return results