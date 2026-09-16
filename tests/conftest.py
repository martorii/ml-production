from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tickets import config


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    """Tests run against a real trained artifact; train one if CI hasn't already."""
    if not config.MODEL_PATH.exists():
        from tickets.train import train

        train(n_rows=4_000)


@pytest.fixture(scope="session")
def client():
    from tickets.api import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload() -> dict:
    return {"text": "hi team, i was charged twice for my subscription this month"}
