"""API contract tests: status codes, text-specific validation, metrics surface."""

from __future__ import annotations

import pytest

from tickets.config import MAX_TEXT_CHARS


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_ready(client):
    assert client.get("/ready").status_code == 200


def test_predict_happy_path(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "billing"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_version"]


@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},
        {"text": "   "},
        {"text": "hi"},
        {"text": "x" * (MAX_TEXT_CHARS + 1)},
        {"text": 42},
        {},
        {"text": "valid ticket text here", "priority": "high"},
    ],
    ids=["empty", "blank", "too-short", "too-long", "wrong-type", "missing", "extra-field"],
)
def test_invalid_input_is_rejected(client, payload):
    """Text needs stricter edge validation than numbers: unbounded strings are a cost
    and memory problem, and blank input gets classified confidently into nonsense."""
    assert client.post("/predict", json=payload).status_code == 422


def test_long_but_legal_text_is_accepted(client):
    payload = {"text": "the app crashes on upload. " * 100}
    assert client.post("/predict", json=payload).status_code == 200


def test_metrics_endpoint_exposes_service_and_drift_metrics(client, valid_payload):
    client.post("/predict", json=valid_payload)
    text = client.get("/metrics").text
    for metric in (
        "tickets_predictions_total",
        "tickets_prediction_latency_seconds",
        "tickets_model_info",
        "tickets_embedding_drift_threshold",
        "tickets_oov_token_rate_baseline",
    ):
        assert metric in text


def test_validation_error_body_is_serialisable(client):
    """A custom validator raising ValueError must still produce a JSON 422, not a 500."""
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["field"] == "text"
    assert "blank" in detail[0]["message"]


def test_rejected_requests_are_counted(client):
    """Bad input must be visible in metrics, not just rejected silently."""
    client.post("/predict", json={"text": ""})
    text = client.get("/metrics").text
    assert 'tickets_rejected_requests_total{field="text"}' in text
    assert 'tickets_prediction_errors_total{reason="validation_error"}' in text
