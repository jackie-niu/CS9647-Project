import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def run_baselines(train_df, test_df):
    X_train, y_train = train_df["text"].tolist(), train_df["label"].to_numpy()
    X_test,  y_test  = test_df["text"].tolist(),  test_df["label"].to_numpy()

    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)

    results = {}

    lr = LogisticRegression(max_iter=2000, n_jobs=-1)
    lr.fit(Xtr, y_train)
    pred_lr = lr.predict(Xte)
    results["tfidf_lr"] = {
        "accuracy": accuracy_score(y_test, pred_lr),
        "f1": f1_score(y_test, pred_lr),
        "precision": precision_score(y_test, pred_lr),
        "recall": recall_score(y_test, pred_lr),
    }

    svm = LinearSVC()
    svm.fit(Xtr, y_train)
    pred_svm = svm.predict(Xte)
    results["tfidf_svm"] = {
        "accuracy": accuracy_score(y_test, pred_svm),
        "f1": f1_score(y_test, pred_svm),
        "precision": precision_score(y_test, pred_svm),
        "recall": recall_score(y_test, pred_svm),
    }

    return results