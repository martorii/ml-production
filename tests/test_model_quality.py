"""The quality gate.

This is what stands between a bad model and production. CI runs it on every change;
if the freshly trained model falls below the thresholds in tickets/config.py, the
build fails and no image is ever built.
"""

from __future__ import annotations

import json

import pytest

from tickets import config
from tickets.model import load


@pytest.fixture(scope="module")
def metrics() -> dict:
    assert config.METRICS_PATH.exists(), "run `python -m tickets.train` first"
    return json.loads(config.METRICS_PATH.read_text())


def test_macro_f1_above_threshold(metrics):
    score = metrics["metrics"]["macro_f1"]
    assert score >= config.MIN_MACRO_F1, (
        f"macro F1 {score:.3f} is below the {config.MIN_MACRO_F1} gate — "
        "this model must not ship"
    )


def test_accuracy_above_threshold(metrics):
    score = metrics["metrics"]["accuracy"]
    assert score >= config.MIN_ACCURACY, f"accuracy {score:.3f} below {config.MIN_ACCURACY}"


@pytest.mark.parametrize("label", config.CLASSES)
def test_every_class_is_served(metrics, label):
    """Aggregate scores hide a class the model has quietly given up on.
    Per-class gates are what stop the rarest queue from being silently abandoned."""
    score = metrics["metrics"]["per_class_f1"][label]
    assert score >= config.MIN_PER_CLASS_F1, f"{label} F1 {score:.3f} too low"


def test_beats_majority_class_baseline(metrics):
    assert metrics["metrics"]["accuracy"] > 1 / len(config.CLASSES) + 0.2


def test_artifact_is_loadable_and_versioned():
    loaded = load()
    assert loaded.version == json.loads(config.METRICS_PATH.read_text())["model_version"]
    assert sorted(loaded.classes) == sorted(config.CLASSES)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("i was charged twice for my subscription and want a refund", "billing"),
        ("the app crashes with a 500 error every time i upload", "technical"),
        ("i cannot log in and the password reset email never arrives", "account"),
        ("my parcel is stuck at the depot and tracking never updates", "shipping"),
    ],
)
def test_behaviour_on_unambiguous_tickets(text, expected):
    """Aggregate metrics hide directional bugs. A textbook ticket per class must route
    correctly, or something is wrong regardless of what macro F1 says."""
    prediction = load().predict([text])[0]
    assert prediction.label == expected
    assert prediction.confidence > 0.5


def test_probabilities_are_well_formed():
    prediction = load().predict(["the invoice does not match the quoted price"])[0]
    assert set(prediction.probabilities) == set(config.CLASSES)
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-6


def test_invariance_to_formatting():
    """Case, whitespace and an order reference must not change the routing decision.
    This is a robustness property of the normaliser, tested through the model."""
    loaded = load()
    plain = loaded.predict(["my parcel is stuck at the depot and tracking never updates"])[0]
    noisy = loaded.predict(
        ["MY PARCEL   is stuck at the DEPOT and tracking never updates (ORD-77123)"]
    )[0]
    assert plain.label == noisy.label


def test_embeddings_are_unit_length():
    embeddings = load().embed(["the dashboard is broken", "refund my duplicate payment"])
    assert embeddings.shape == (2, config.EMBEDDING_DIM)
    assert all(abs(float((e**2).sum()) - 1.0) < 1e-6 for e in embeddings)
