"""Normalisation is inside the model, so a change here changes predictions.
These tests pin the behaviour that training and serving both depend on."""

from __future__ import annotations

import pytest

from tickets.config import MAX_TEXT_CHARS
from tickets.text import normalize


@pytest.mark.parametrize(
    "raw,expected_fragment",
    [
        ("Contact me at Erik.M@example.com now", "<email>"),
        ("see https://example.com/help for details", "<url>"),
        ("my order ORD-88213 is late", "<ref>"),
        ("call me on 555 123 4567 today", "<num>"),
        ("you charged me 49.99 twice", "<num>"),
    ],
)
def test_pii_and_numbers_are_masked(raw, expected_fragment):
    result = normalize(raw)
    assert expected_fragment in result


def test_no_raw_pii_survives():
    result = normalize("email erik@example.com or call 555 123 4567 about ORD-99001")
    assert "erik@example.com" not in result
    assert "555" not in result
    assert "99001" not in result


def test_case_unicode_and_whitespace():
    assert normalize("  THE   App Crashed \n again ") == "the app crashed again"


def test_truncation_bounds_cost():
    assert len(normalize("spam " * 10_000)) <= MAX_TEXT_CHARS


def test_is_idempotent():
    """Normalising twice must not change the result, or serving could drift from training."""
    once = normalize("Charged 49.99 at https://shop.example.com ORD-12345")
    assert normalize(once) == once
