"""Drift tests: quiet on stable traffic, loud on shifted traffic.

A drift monitor that never fires and one that always fires are equally useless, so
both directions are tested. These are the tests that would catch a threshold someone
'temporarily' loosened.
"""

from __future__ import annotations

import pytest

from tickets import config, data
from tickets.model import load
from tickets.monitoring import DRIFT_DETECTED, SIGNALS, load_drift_monitor
from tickets.text import normalize


@pytest.fixture(scope="module")
def loaded():
    return load()


def _feed(monitor, loaded, texts) -> None:
    predictions = loaded.predict(texts)
    embeddings = loaded.embed(texts)
    for text, prediction, embedding in zip(texts, predictions, embeddings, strict=True):
        monitor.observe(normalize(text), embedding, prediction.label, prediction.confidence)


def _firing() -> set[str]:
    return {
        sample.labels["signal"]
        for metric in DRIFT_DETECTED.collect()
        for sample in metric.samples
        if sample.value == 1
    }


def _reset() -> None:
    """The Prometheus registry is global; don't let one test's labels leak into the next."""
    DRIFT_DETECTED.clear()


def test_no_drift_on_the_training_distribution(loaded):
    _reset()
    monitor = load_drift_monitor(loaded)
    texts = data.generate(n_rows=config.DRIFT_WINDOW_SIZE, seed=11)["text"].tolist()
    _feed(monitor, loaded, texts)
    assert not _firing(), "stable traffic must not raise drift"


def test_drift_detected_on_shifted_traffic(loaded):
    _reset()
    monitor = load_drift_monitor(loaded)
    texts = data.generate(
        n_rows=config.DRIFT_WINDOW_SIZE, seed=12, shift=1.0
    )["text"].tolist()
    _feed(monitor, loaded, texts)
    firing = _firing()
    assert firing, "a fully shifted population must trigger drift"
    assert {"oov", "class_mix"} & firing, f"expected vocabulary or class drift, got {firing}"


def test_new_vocabulary_alone_raises_oov(loaded):
    """The signal that only text has: words the vectorizer has never seen."""
    _reset()
    monitor = load_drift_monitor(loaded)
    texts = [
        "zylotrix quantumpay helioscloud orbitsync glarbnix wozzlefrim bleepdorf"
    ] * config.DRIFT_WINDOW_SIZE
    _feed(monitor, loaded, texts)
    assert "oov" in _firing()


def test_embedding_drift_rises_under_shift(loaded):
    """Regression test for a real bug in the first version of this monitor.

    It measured the *mean distance of each document* to the training centroid, which
    is a measure of spread, not location — drifted traffic that was more uniform than
    training data scored lower, so drift showed up as an improvement. The statistic is
    now centroid-to-centroid, and this test fails if anyone changes it back.
    """
    from tickets.monitoring import EMBEDDING_DRIFT

    def measure(shift: float) -> float:
        _reset()
        monitor = load_drift_monitor(loaded)
        texts = data.generate(
            n_rows=config.DRIFT_WINDOW_SIZE, seed=21, shift=shift
        )["text"].tolist()
        _feed(monitor, loaded, texts)
        return EMBEDDING_DRIFT._value.get()

    stable, shifted = measure(0.0), measure(1.0)
    assert shifted > stable, f"drift must increase the statistic (got {shifted} vs {stable})"


def test_oov_threshold_is_calibrated_not_hardcoded(loaded):
    """The threshold must come from the training corpus, so a retrain re-derives it."""
    monitor = load_drift_monitor(loaded)
    assert monitor.oov_threshold >= 3 * monitor.oov_baseline
    assert 0.0 < monitor.oov_threshold < 1.0


def test_all_signals_are_registered(loaded):
    _reset()
    monitor = load_drift_monitor(loaded)
    _feed(monitor, loaded, data.generate(n_rows=config.DRIFT_WINDOW_SIZE, seed=13)["text"].tolist())
    reported = {
        sample.labels["signal"]
        for metric in DRIFT_DETECTED.collect()
        for sample in metric.samples
    }
    assert reported == set(SIGNALS)


def test_window_is_bounded(loaded):
    monitor = load_drift_monitor(loaded)
    _feed(
        monitor,
        loaded,
        data.generate(n_rows=config.DRIFT_WINDOW_SIZE * 2, seed=14)["text"].tolist(),
    )
    assert len(monitor._embeddings) == config.DRIFT_WINDOW_SIZE
