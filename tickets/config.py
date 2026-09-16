"""Every threshold and path in one place."""

from __future__ import annotations

import os
from pathlib import Path

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
REFERENCE_PATH = ARTIFACTS_DIR / "reference_sample.csv"
DRIFT_REFERENCE_PATH = ARTIFACTS_DIR / "drift_reference.json"

RANDOM_SEED = 42

CLASSES = ["billing", "technical", "account", "shipping"]

# --- Text handling ----------------------------------------------------------
MAX_TEXT_CHARS = 5_000
MIN_TEXT_CHARS = 3
EMBEDDING_DIM = 64

# --- Quality gate -----------------------------------------------------------
# CI refuses to ship a model below these. Macro-F1, not accuracy: it weights
# every class equally, so a model that ignores a rare class cannot hide.
# The ceiling on this corpus is ~0.86: 4% of labels are noisy and 20% of tickets are
# ambiguous or vague by construction. Thresholds sit just under a good model, not at
# an aspirational number nobody can reach — an unreachable gate gets disabled, and a
# disabled gate protects nothing.
MIN_MACRO_F1 = 0.80
MIN_ACCURACY = 0.80
MIN_PER_CLASS_F1 = 0.75

# --- Drift detection --------------------------------------------------------
DRIFT_WINDOW_SIZE = int(os.environ.get("DRIFT_WINDOW_SIZE", "200"))
DRIFT_RECOMPUTE_EVERY = int(os.environ.get("DRIFT_RECOMPUTE_EVERY", "50"))
DRIFT_P_VALUE_THRESHOLD = float(os.environ.get("DRIFT_P_VALUE_THRESHOLD", "0.01"))
# Total variation distance between predicted-class shares and the training mix.
CLASS_DRIFT_THRESHOLD = float(os.environ.get("CLASS_DRIFT_THRESHOLD", "0.2"))
# Out-of-vocabulary token rate. Vocabulary moving is text's version of a schema change.
# Only a fallback: the real threshold is calibrated against the training corpus at
# train time and written into drift_reference.json.
OOV_RATE_THRESHOLD = float(os.environ.get("OOV_RATE_THRESHOLD", "0.05"))
# Below this, a prediction counts as low-confidence.
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("LOW_CONFIDENCE_THRESHOLD", "0.55"))
LOW_CONFIDENCE_RATE_THRESHOLD = float(os.environ.get("LOW_CONFIDENCE_RATE_THRESHOLD", "0.25"))
