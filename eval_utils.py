import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def summarize_predictions(df, y_true, y_pred, text_col="text", k=30):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    # Error analysis
    err = df.copy()
    err = err.assign(y_true=y_true, y_pred=y_pred)
    err = err[err["y_true"] != err["y_pred"]].copy()

    # Add text snippets for error analysis
    err["text_snippet"] = err[text_col].astype(str).str.replace("\n", " ").str.slice(0, 280)
    if "source_text" in err.columns:
        err["source_snippet"] = err["source_text"].astype(str).str.replace("\n", " ").str.slice(0, 180)

    cols = ["y_true", "y_pred", "text_snippet"]
    if "source_snippet" in err.columns:
        cols.insert(0, "source_snippet")

    err_view = err[cols].head(k)
    return cm, report, err_view