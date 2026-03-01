import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def summarize_predictions(df, y_true, y_pred, text_col="text", k=25):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    # Misclassified examples for error analysis
    err = df.copy()
    err = err.assign(y_true=y_true, y_pred=y_pred)
    err = err[err["y_true"] != err["y_pred"]].copy()
    err["text_snippet"] = err[text_col].astype(str).str.replace("\n", " ").str.slice(0, 300)

    # return top-k errors
    err_view = err[["dataset", "source", "headline", "y_true", "y_pred", "text_snippet"]].head(k)

    return cm, report, err_view