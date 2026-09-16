"""Artifact loading, kept separate from the API so tests can use it directly."""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from tickets import config


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


@dataclass
class LoadedModel:
    pipeline: Pipeline
    svd: TruncatedSVD
    classes: list[str]
    version: str
    metrics: dict

    def predict(self, texts: list[str]) -> list[Prediction]:
        probabilities = self.pipeline.predict_proba(texts)
        labels = list(self.pipeline.classes_)
        return [
            Prediction(
                label=labels[int(np.argmax(row))],
                confidence=float(np.max(row)),
                probabilities={label: float(p) for label, p in zip(labels, row, strict=True)},
            )
            for row in probabilities
        ]

    def embed(self, texts: list[str]) -> np.ndarray:
        """Project text into the same LSA space the drift reference was built in."""
        matrix = self.pipeline.named_steps["tfidf"].transform(
            self.pipeline.named_steps["normalize"].transform(texts)
        )
        vectors = self.svd.transform(matrix)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


def load() -> LoadedModel:
    """Fail loudly: a service without its model is worse than one that won't start."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model at {config.MODEL_PATH}. Run `python -m tickets.train` first."
        )
    bundle = joblib.load(config.MODEL_PATH)
    metrics = (
        json.loads(config.METRICS_PATH.read_text()) if config.METRICS_PATH.exists() else {}
    )
    return LoadedModel(
        pipeline=bundle["pipeline"],
        svd=bundle["svd"],
        classes=bundle["classes"],
        version=bundle["model_version"],
        metrics=metrics,
    )
