"""Training entry point.  Run: `python -m tickets.train`

Writes four artifacts. The last one is what makes text drift monitoring possible:

  artifacts/model.joblib          pipeline (normalise -> tfidf -> logreg) + LSA embedder
  artifacts/metrics.json          held-out metrics + version, read by the quality gate
  artifacts/reference_sample.csv  training texts, for length/OOV baselines
  artifacts/drift_reference.json  embedding centroid, calibrated thresholds, class mix
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import UTC, datetime

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from tickets import config, data
from tickets.text import normalize_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


def build_pipeline() -> Pipeline:
    """Normalisation and vectorisation live inside the model, so they cannot diverge."""
    return Pipeline(
        [
            ("normalize", FunctionTransformer(normalize_batch, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1_000, C=4.0, class_weight="balanced",
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "nogit"


def _unit_embeddings(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise so that 1 - dot product is cosine distance."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _window_centroid_distance(embeddings: np.ndarray, centroid: np.ndarray) -> float:
    """Cosine distance between a window's own centroid and the training centroid.

    Note what this is *not*: the mean per-document distance to the centroid. That
    measures how spread out the window is, not where it sits — a drifted batch that
    is more uniform than training data scores *lower* on it, so drift hides as an
    improvement. Comparing centroid to centroid measures the thing we actually care
    about, which is whether the traffic has moved.
    """
    mean_vector = embeddings.mean(axis=0)
    mean_vector = mean_vector / max(float(np.linalg.norm(mean_vector)), 1e-12)
    return float(1.0 - mean_vector @ centroid)


def _calibrate_embedding_threshold(
    embeddings: np.ndarray, centroid: np.ndarray, window: int,
    rng: np.random.Generator, n_boot: int = 500,
) -> float:
    """How far can a window centroid sit from the training centroid by pure chance?

    Bootstrap windows out of the training set, measure each window's centroid
    distance, and take the 99.5th percentile. The alerting threshold is derived from
    data rather than guessed, and it is recalibrated on every retrain, so it always
    belongs to the model actually being served.
    """
    distances = [
        _window_centroid_distance(embeddings[rng.integers(0, len(embeddings), size=window)],
                                  centroid)
        for _ in range(n_boot)
    ]
    return float(np.quantile(distances, 0.995))


def _calibrate_oov_threshold(oov_rates: np.ndarray) -> float:
    """Same principle for vocabulary. The training baseline is near zero by
    construction (the vectorizer was fitted on this text), so a fixed 25% would let a
    real vocabulary shift run for weeks. Alert on a multiple of the baseline, with a
    floor so noise alone cannot trip it."""
    return float(max(3.0 * np.mean(oov_rates), 0.05))


def train(n_rows: int = 8_000) -> dict:
    df = data.generate(n_rows=n_rows)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2,
        random_state=config.RANDOM_SEED, stratify=df["label"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = {
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "per_class_f1": {c: float(report[c]["f1-score"]) for c in config.CLASSES},
    }

    # --- Embedding space (LSA over the same TF-IDF the classifier sees) ---------
    # Reusing the fitted vectorizer keeps the drift monitor in the model's own view
    # of the text, and costs no extra dependency or model download.
    tfidf = pipeline.named_steps["tfidf"]
    normalized_train = normalize_batch(X_train)
    train_matrix = tfidf.transform(normalized_train)
    svd = TruncatedSVD(n_components=config.EMBEDDING_DIM, random_state=config.RANDOM_SEED)
    embeddings = _unit_embeddings(svd.fit_transform(train_matrix))
    centroid = embeddings.mean(axis=0)
    centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
    rng = np.random.default_rng(config.RANDOM_SEED)
    embedding_threshold = _calibrate_embedding_threshold(
        embeddings, centroid, config.DRIFT_WINDOW_SIZE, rng
    )

    # --- Baselines the drift monitor compares live traffic against -------------
    unigrams = {term for term in tfidf.vocabulary_ if " " not in term}
    tokenize = tfidf.build_tokenizer()
    token_counts = [len(tokenize(t)) for t in normalized_train]
    oov_rates = np.array([
        sum(tok not in unigrams for tok in tokenize(t)) / max(len(tokenize(t)), 1)
        for t in normalized_train[:2_000]
    ])
    oov_threshold = _calibrate_oov_threshold(oov_rates)

    version = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{_git_sha()}"
    payload = {
        "model_version": version,
        "trained_at": datetime.now(UTC).isoformat(),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "vocabulary_size": int(len(tfidf.vocabulary_)),
        "classes": config.CLASSES,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=config.CLASSES
        ).tolist(),
        "thresholds": {
            "macro_f1": config.MIN_MACRO_F1,
            "accuracy": config.MIN_ACCURACY,
            "per_class_f1": config.MIN_PER_CLASS_F1,
        },
    }

    drift_reference = {
        "embedding_centroid": centroid.tolist(),
        "embedding_threshold": embedding_threshold,
        "class_distribution": {
            c: float((y_train == c).mean()) for c in config.CLASSES
        },
        "oov_rate_baseline": float(np.mean(oov_rates)),
        "oov_threshold": oov_threshold,
        "token_count_mean": float(np.mean(token_counts)),
    }

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "svd": svd, "model_version": version, "classes": config.CLASSES},
        config.MODEL_PATH,
    )
    config.METRICS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    config.DRIFT_REFERENCE_PATH.write_text(json.dumps(drift_reference, indent=2) + "\n")

    reference = X_train.sample(
        min(2_000, len(X_train)), random_state=config.RANDOM_SEED
    ).to_frame("text")
    # Store normalised text: the PII is already scrubbed, so the artifact is safe to keep.
    reference["text"] = normalize_batch(reference["text"])
    reference["n_chars"] = reference["text"].str.len()
    reference["n_tokens"] = [len(tokenize(t)) for t in reference["text"]]
    reference.to_csv(config.REFERENCE_PATH, index=False)

    log.info("model_version=%s macro_f1=%.4f accuracy=%.4f", version, metrics["macro_f1"],
             metrics["accuracy"])
    log.info("per-class f1: %s", {k: round(v, 3) for k, v in metrics["per_class_f1"].items()})
    log.info("calibrated thresholds: embedding=%.5f oov=%.4f (baseline %.4f)",
             embedding_threshold, oov_threshold, float(np.mean(oov_rates)))
    log.info("artifacts written to %s", config.ARTIFACTS_DIR.resolve())
    return payload


if __name__ == "__main__":
    train()
