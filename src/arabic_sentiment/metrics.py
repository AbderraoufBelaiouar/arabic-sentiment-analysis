"""Evaluation metrics used during training and reporting."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute accuracy and macro-F1 from Trainer's `EvalPrediction`.

    The model outputs logits of shape (batch, num_labels).
    We take argmax to get the predicted class index.
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }
