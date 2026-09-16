"""Service metrics and text drift detection.

Tabular drift is easy: compare each column's distribution. Text has no columns, so
this module watches five different windows onto the same question — "is the traffic
still the traffic the model was trained on?"

  1. embedding drift   mean cosine distance from the training centroid (LSA space)
  2. OOV rate          share of tokens the vectorizer has never seen
  3. length drift      KS test on token counts per message
  4. class drift       predicted-class mix vs the training mix (total variation)
  5. confidence drift  mean confidence and low-confidence rate

None of them needs ground-truth labels, which is the point: labels arrive weeks late,
if ever. Each catches a different failure, and they disagree usefully — new vocabulary
spikes OOV, a viral incident moves the class mix, a new language moves everything.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram
from scipy import stats

from tickets import config

log = logging.getLogger("monitoring")

# --- Service metrics --------------------------------------------------------
PREDICTIONS_TOTAL = Counter(
    "tickets_predictions_total", "Predictions served", ["model_version", "predicted_class"]
)
PREDICTION_ERRORS = Counter(
    "tickets_prediction_errors_total", "Failed prediction requests", ["reason"]
)
REJECTED_REQUESTS = Counter(
    "tickets_rejected_requests_total",
    "Requests rejected by schema validation, by offending field",
    ["field"],
)
PREDICTION_LATENCY = Histogram(
    "tickets_prediction_latency_seconds",
    "Model inference latency",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
INPUT_LENGTH = Histogram(
    "tickets_input_tokens",
    "Token count of incoming tickets",
    buckets=(5, 10, 20, 40, 80, 160, 320, 640),
)
MODEL_INFO = Gauge("tickets_model_info", "Loaded model, 1 per version", ["model_version"])
MODEL_MACRO_F1 = Gauge("tickets_model_offline_macro_f1", "Macro F1 on held-out data at training")

# --- Drift metrics ----------------------------------------------------------
EMBEDDING_DRIFT = Gauge(
    "tickets_embedding_drift_distance",
    "Cosine distance between the live window centroid and the training centroid",
)
EMBEDDING_DRIFT_THRESHOLD = Gauge(
    "tickets_embedding_drift_threshold", "Calibrated alerting threshold for embedding drift"
)
OOV_RATE = Gauge("tickets_oov_token_rate", "Share of live tokens unseen during training")
OOV_RATE_BASELINE = Gauge("tickets_oov_token_rate_baseline", "OOV rate measured on training data")
OOV_THRESHOLD = Gauge(
    "tickets_oov_token_rate_threshold", "Calibrated alerting threshold for OOV rate"
)
LENGTH_DRIFT_SCORE = Gauge("tickets_length_drift_ks_statistic", "KS statistic on token counts")
LENGTH_DRIFT_PVALUE = Gauge("tickets_length_drift_p_value", "KS p-value on token counts")
CLASS_DRIFT = Gauge(
    "tickets_class_distribution_drift",
    "Total variation distance between predicted and training class mix",
)
CLASS_SHARE = Gauge(
    "tickets_predicted_class_share", "Share of the live window per class", ["predicted_class"]
)
MEAN_CONFIDENCE = Gauge("tickets_mean_confidence", "Mean top-class probability over the window")
LOW_CONFIDENCE_RATE = Gauge(
    "tickets_low_confidence_rate",
    f"Share of predictions below {config.LOW_CONFIDENCE_THRESHOLD} confidence",
)
DRIFT_DETECTED = Gauge(
    "tickets_drift_detected", "1 if this drift signal is currently firing", ["signal"]
)
DRIFTED_SIGNAL_COUNT = Gauge(
    "tickets_drifted_signal_count", "How many of the five drift signals are firing"
)
DRIFT_WINDOW_FILL = Gauge("tickets_drift_window_rows", "Requests buffered for drift detection")

SIGNALS = ("embedding", "oov", "length", "class_mix", "confidence")


class DriftMonitor:
    """Rolling window of live requests, compared against the training reference."""

    def __init__(self, reference: pd.DataFrame, drift_reference: dict, tokenizer, unigrams: set):
        self.reference_tokens = reference["n_tokens"].to_numpy()
        self.centroid = np.asarray(drift_reference["embedding_centroid"], dtype=float)
        self.embedding_threshold = float(drift_reference["embedding_threshold"])
        self.class_distribution = drift_reference["class_distribution"]
        self.oov_baseline = float(drift_reference["oov_rate_baseline"])
        self.oov_threshold = float(
            drift_reference.get("oov_threshold", config.OOV_RATE_THRESHOLD)
        )
        self.tokenizer = tokenizer
        self.unigrams = unigrams

        self._embeddings: deque[np.ndarray] = deque(maxlen=config.DRIFT_WINDOW_SIZE)
        self._tokens: deque[int] = deque(maxlen=config.DRIFT_WINDOW_SIZE)
        self._oov: deque[float] = deque(maxlen=config.DRIFT_WINDOW_SIZE)
        self._labels: deque[str] = deque(maxlen=config.DRIFT_WINDOW_SIZE)
        self._confidence: deque[float] = deque(maxlen=config.DRIFT_WINDOW_SIZE)
        self._since_recompute = 0
        self._lock = threading.Lock()

        EMBEDDING_DRIFT_THRESHOLD.set(self.embedding_threshold)
        OOV_RATE_BASELINE.set(self.oov_baseline)
        OOV_THRESHOLD.set(self.oov_threshold)
        for signal in SIGNALS:
            DRIFT_DETECTED.labels(signal).set(0)

    def token_stats(self, normalized_text: str) -> tuple[int, float]:
        tokens = self.tokenizer(normalized_text)
        if not tokens:
            return 0, 0.0
        oov = sum(token not in self.unigrams for token in tokens) / len(tokens)
        return len(tokens), oov

    def observe(
        self, normalized_text: str, embedding: np.ndarray, label: str, confidence: float
    ) -> None:
        n_tokens, oov_rate = self.token_stats(normalized_text)
        INPUT_LENGTH.observe(n_tokens)
        with self._lock:
            self._embeddings.append(embedding)
            self._tokens.append(n_tokens)
            self._oov.append(oov_rate)
            self._labels.append(label)
            self._confidence.append(confidence)
            self._since_recompute += 1
            DRIFT_WINDOW_FILL.set(len(self._embeddings))
            ready = (
                len(self._embeddings) >= config.DRIFT_WINDOW_SIZE
                and self._since_recompute >= config.DRIFT_RECOMPUTE_EVERY
            )
            if not ready:
                return
            self._since_recompute = 0
            snapshot = (
                np.vstack(self._embeddings),
                np.array(self._tokens),
                np.array(self._oov),
                list(self._labels),
                np.array(self._confidence),
            )
        self._recompute(*snapshot)

    def _recompute(self, embeddings, tokens, oov, labels, confidence) -> None:
        firing = {}

        # 1. Embedding drift: where the window sits, not how spread out it is.
        #    Mean per-document distance would measure spread, and a tighter-than-usual
        #    batch of drifted text would then score as *less* drifted.
        mean_vector = embeddings.mean(axis=0)
        mean_vector = mean_vector / max(float(np.linalg.norm(mean_vector)), 1e-12)
        distance = float(1.0 - mean_vector @ self.centroid)
        EMBEDDING_DRIFT.set(distance)
        firing["embedding"] = distance > self.embedding_threshold

        # 2. Vocabulary drift. Text's equivalent of an unannounced schema change.
        oov_rate = float(oov.mean())
        OOV_RATE.set(oov_rate)
        firing["oov"] = oov_rate > self.oov_threshold

        # 3. Length drift. Cheap, and catches format changes (a new UI, a bot, a template).
        ks = stats.ks_2samp(self.reference_tokens, tokens)
        LENGTH_DRIFT_SCORE.set(float(ks.statistic))
        LENGTH_DRIFT_PVALUE.set(float(ks.pvalue))
        firing["length"] = bool(ks.pvalue < config.DRIFT_P_VALUE_THRESHOLD)

        # 4. Prediction drift: the class mix the model is producing.
        live_share = pd.Series(labels).value_counts(normalize=True)
        total_variation = 0.5 * sum(
            abs(float(live_share.get(c, 0.0)) - self.class_distribution.get(c, 0.0))
            for c in set(self.class_distribution) | set(live_share.index)
        )
        CLASS_DRIFT.set(total_variation)
        for c in config.CLASSES:
            CLASS_SHARE.labels(c).set(float(live_share.get(c, 0.0)))
        firing["class_mix"] = total_variation > config.CLASS_DRIFT_THRESHOLD

        # 5. Confidence drift. A model getting unsure of itself is the earliest sign
        #    that inputs have moved somewhere it was never trained.
        MEAN_CONFIDENCE.set(float(confidence.mean()))
        low_rate = float((confidence < config.LOW_CONFIDENCE_THRESHOLD).mean())
        LOW_CONFIDENCE_RATE.set(low_rate)
        firing["confidence"] = low_rate > config.LOW_CONFIDENCE_RATE_THRESHOLD

        for signal in SIGNALS:
            DRIFT_DETECTED.labels(signal).set(int(firing[signal]))
        DRIFTED_SIGNAL_COUNT.set(sum(firing.values()))
        if any(firing.values()):
            log.warning("drift signals firing: %s", [s for s, on in firing.items() if on])


def load_drift_monitor(loaded_model) -> DriftMonitor | None:
    """Built from the served model itself, so monitoring always matches what is serving."""
    if not (config.REFERENCE_PATH.exists() and config.DRIFT_REFERENCE_PATH.exists()):
        log.warning("drift reference artifacts missing; drift monitoring disabled")
        return None
    tfidf = loaded_model.pipeline.named_steps["tfidf"]
    return DriftMonitor(
        reference=pd.read_csv(config.REFERENCE_PATH),
        drift_reference=json.loads(config.DRIFT_REFERENCE_PATH.read_text()),
        tokenizer=tfidf.build_tokenizer(),
        unigrams={term for term in tfidf.vocabulary_ if " " not in term},
    )
