"""
The SentimentClassifier class — the heart of the system.

Provides:
- OOP SentimentClassifier with load-once / predict-many lifecycle
- @timed decorator for latency logging
- WeightedTrainer for training with class-balanced CE loss
- Helpers for computing class weights
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
)
from arabert.preprocess import ArabertPreprocessor
from sklearn.utils.class_weight import compute_class_weight

from arabic_sentiment.config import settings, LABEL_NAMES, NUM_LABELS

logger = logging.getLogger(__name__)

# Label mapping — index → human-readable name
LABEL_MAP = {i: name for i, name in enumerate(LABEL_NAMES)}


# ──────────────────────────────────────────────
# @timed decorator
# ──────────────────────────────────────────────


def timed(func):
    """Decorator that logs how long a function takes."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} took {elapsed:.4f}s")
        return result

    return wrapper


# ──────────────────────────────────────────────
# SentimentClassifier (inference)
# ──────────────────────────────────────────────


class SentimentClassifier:
    """
    Arabic sentiment classifier using AraBERT.

    Usage:
        classifier = SentimentClassifier()
        classifier.load()               # once at startup
        result = classifier.predict_one("المنتج رائع")
    """

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.preprocessor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = False

    @timed
    def load(self) -> None:
        """Load the model from disk into memory (called once at startup)."""
        logger.info(f"Loading model from {settings.model_path}...")

        self.preprocessor = ArabertPreprocessor(settings.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            settings.model_path
        )
        self.model.to(self.device)
        self.model.eval()

        self._loaded = True
        logger.info(f"Model loaded on {self.device}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @timed
    def predict_one(self, text: str) -> dict[str, Any]:
        """Predict sentiment for a single Arabic text."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        cleaned = self.preprocessor.preprocess(text)

        inputs = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=settings.max_seq_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)
        confidence, predicted_class = torch.max(probabilities, dim=-1)

        label = LABEL_MAP[predicted_class.item()]

        return {
            "label": label,
            "confidence": round(confidence.item(), 4),
        }

    @timed
    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predict sentiment for multiple texts at once."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        cleaned = [self.preprocessor.preprocess(t) for t in texts]

        inputs = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=settings.max_seq_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)
        confidences, predicted_classes = torch.max(probabilities, dim=-1)

        results = []
        for i in range(len(texts)):
            results.append(
                {
                    "label": LABEL_MAP[predicted_classes[i].item()],
                    "confidence": round(confidences[i].item(), 4),
                }
            )
        return results


# ──────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────


def load_model_and_tokenizer(
    model_name: str | None = None,
    num_labels: int = NUM_LABELS,
):
    """Load a pretrained AraBERT model, tokenizer, and preprocessor for training."""
    name = model_name or settings.model_name
    preprocessor = ArabertPreprocessor(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
    return model, tokenizer, preprocessor


def compute_class_weights(labels: list[int]) -> torch.Tensor:
    """Compute balanced class weights and return as a float tensor."""
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=np.array(labels),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.tensor(weights, dtype=torch.float).to(device)


class WeightedTrainer(Trainer):
    """Trainer sub-class that applies class-balanced weights to the CE loss."""

    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss
