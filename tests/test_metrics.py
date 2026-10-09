"""Unit tests for the metrics module."""

import numpy as np

from arabic_sentiment.metrics import compute_metrics


class _FakeEvalPred:
    """Mimics transformers EvalPrediction as a simple (preds, labels) tuple."""

    def __init__(self, predictions, labels):
        self.predictions = predictions
        self.label_ids = labels

    def __iter__(self):
        return iter((self.predictions, self.label_ids))


def test_perfect_predictions():
    # 3-class: all correct
    logits = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
    labels = np.array([0, 1, 2])
    result = compute_metrics(_FakeEvalPred(logits, labels))
    assert result["accuracy"] == 1.0
    assert result["f1_macro"] == 1.0


def test_all_wrong_predictions():
    logits = np.array([[10, 0, 0], [10, 0, 0], [10, 0, 0]])
    labels = np.array([1, 2, 2])
    result = compute_metrics(_FakeEvalPred(logits, labels))
    assert result["accuracy"] == 0.0


def test_partial_predictions():
    logits = np.array([[10, 0, 0], [0, 10, 0], [10, 0, 0]])
    labels = np.array([0, 1, 2])
    result = compute_metrics(_FakeEvalPred(logits, labels))
    assert 0 < result["accuracy"] < 1
    assert 0 < result["f1_macro"] < 1
