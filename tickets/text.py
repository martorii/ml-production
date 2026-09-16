"""Text normalisation, shared by training and serving.

This function is mounted *inside* the sklearn pipeline, so serving cannot
normalise differently from training — the most common source of train/serve skew
in NLP systems.

It also scrubs obvious PII. Two reasons, and the second is the one people forget:
  1. Emails, phone numbers and card-shaped digit runs should not be logged or
     end up in a drift reference sample sitting on disk.
  2. They are pure noise to the model. A phone number is never evidence that a
     ticket is about billing; leaving it in just inflates the vocabulary.
"""

from __future__ import annotations

import re
import unicodedata

from tickets.config import MAX_TEXT_CHARS

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_URL = re.compile(r"https?://\S+|www\.\S+")
_ORDER = re.compile(r"\b(?:ord|order|inv|ticket)[-#\s]?\d{4,}\b", re.IGNORECASE)
_LONG_DIGITS = re.compile(r"\b\d[\d\s-]{7,}\d\b")
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_WHITESPACE = re.compile(r"\s+")


def normalize(raw: str) -> str:
    """Lowercase, strip PII to placeholders, collapse whitespace, truncate."""
    text = unicodedata.normalize("NFKC", raw)[:MAX_TEXT_CHARS].lower()
    text = _URL.sub(" <url> ", text)
    text = _EMAIL.sub(" <email> ", text)
    text = _ORDER.sub(" <ref> ", text)
    text = _LONG_DIGITS.sub(" <num> ", text)
    # Keep small numbers as a placeholder too: "charged 49.99" and "charged 12.00"
    # should look the same to a bag-of-words model.
    text = _NUMBER.sub(" <num> ", text)
    return _WHITESPACE.sub(" ", text).strip()


def normalize_batch(texts) -> list[str]:
    """Pipeline-facing wrapper. Module-level so joblib can pickle a reference to it."""
    return [normalize(str(t)) for t in texts]
