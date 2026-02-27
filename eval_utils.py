import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Helper function to print confusion matrix and classification report for given true and predicted labels
def print_classification(y_true, y_pred):
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification report:\n", classification_report(y_true, y_pred, digits=3))

# Function for HuggingFace Trainer
# Given metric objects for accuracy, f1, precision, and recall
def hf_compute_metrics_factory(metric_acc, metric_f1, metric_precision, metric_recall):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1": metric_f1.compute(predictions=preds, references=labels, average="binary")["f1"],
            "precision": metric_precision.compute(predictions=preds, references=labels, average="binary")["precision"],
            "recall": metric_recall.compute(predictions=preds, references=labels, average="binary")["recall"],
        }
    return compute_metrics