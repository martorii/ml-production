"""Synthetic support-ticket corpus.

Deterministic and offline, with a `shift` knob so drift can be demonstrated rather
than described. `shift` does three things at once, mimicking how text distributions
actually move in production:

  * new vocabulary appears  (a product rename, new slang, a new payment provider)
  * messages get longer and noisier
  * the class mix changes   (a billing incident floods one queue)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tickets.config import CLASSES, RANDOM_SEED

# Per-class phrase banks. Vocabulary overlaps on purpose: "account" and "billing"
# both talk about charges, so the problem is not trivially separable.
PHRASES: dict[str, list[str]] = {
    "billing": [
        "i was charged twice for my subscription this month",
        "the invoice amount does not match the quoted price",
        "please refund the duplicate payment on my credit card",
        "why is there an extra fee of 12.99 on my bill",
        "my payment failed but the money left my bank account",
        "i want to cancel the plan and stop being billed",
        "the discount code was not applied at checkout",
        "can you send me a vat receipt for last month",
    ],
    "technical": [
        "the app crashes every time i open the reports page",
        "i keep getting a 500 error when i upload a file",
        "the dashboard has been loading forever since the update",
        "sync stopped working between my laptop and my phone",
        "the export button does nothing at all in chrome",
        "api requests time out after about thirty seconds",
        "the page is blank after i log in on safari",
        "notifications are not being delivered any more",
    ],
    "account": [
        "i cannot log in and the password reset email never arrives",
        "please delete my account and all associated data",
        "i need to change the email address on my profile",
        "two factor authentication is locking me out of my account",
        "how do i add a colleague as a second user",
        "my account was suspended without any explanation",
        "i want to transfer ownership of the workspace",
        "the login link says my session has expired",
    ],
    "shipping": [
        "my parcel has been stuck at the depot for a week",
        "the tracking number does not update at all",
        "the package arrived damaged and the box was open",
        "i need to change the delivery address before dispatch",
        "the courier left the package with a neighbour",
        "when will my order be shipped it has been ten days",
        "half of the items were missing from the delivery",
        "delivery was marked as completed but nothing arrived",
    ],
}

# Genuinely ambiguous tickets, each plausible under two labels. Real queues are
# full of these, and they are why a text classifier never reaches 100%.
AMBIGUOUS: dict[tuple[str, str], list[str]] = {
    ("billing", "shipping"): [
        "i paid for express delivery and it never arrived",
        "you charged me shipping costs for an order i cancelled",
    ],
    ("billing", "account"): [
        "i was billed after i closed my account",
        "my plan upgraded itself and i never authorised it",
    ],
    ("technical", "account"): [
        "the login page keeps throwing an error",
        "single sign on redirects me in a loop",
    ],
    ("technical", "shipping"): [
        "the tracking widget on the site shows nothing",
        "the order status page has been broken for days",
    ],
}

# Short, vague tickets: almost no signal, which is exactly what makes them realistic.
VAGUE = [
    "it does not work",
    "please help",
    "this is the third time i am writing",
    "nothing happens when i click",
    "can someone look into this",
]

OPENERS = ["hi team", "hello support", "good morning", "hey", "dear support", ""]
CLOSERS = [
    "thanks in advance",
    "please advise",
    "any help appreciated",
    "this is urgent",
    "regards",
    "",
]
NOISE = [
    "i have been a customer for three years",
    "i already tried the usual steps",
    "let me know if you need more details",
    "ticket reference ORD-88213",
    "you can reach me at user@example.com",
]

# Vocabulary that did not exist at training time. Drives the OOV signal.
NEW_VOCABULARY = [
    "the new quantumpay checkout flow is broken",
    "since migrating to helioscloud nothing works",
    "the zenithcard integration rejects my details",
    "your lumenbot assistant gave me wrong information",
    "the orbitsync beta keeps disconnecting",
]


def generate(n_rows: int = 4_000, seed: int = RANDOM_SEED, shift: float = 0.0) -> pd.DataFrame:
    """Return a DataFrame of `text` and `label`.

    shift=0.0 reproduces the training distribution; shift=1.0 is a clearly
    different population in vocabulary, length and class mix.
    """
    rng = np.random.default_rng(seed)

    weights = np.full(len(CLASSES), 1 / len(CLASSES))
    if shift:
        # A billing incident floods one queue.
        weights = weights + shift * np.array([0.45, -0.15, -0.15, -0.15])
        weights = np.clip(weights, 0.01, None)
        weights = weights / weights.sum()

    ambiguous_pairs = list(AMBIGUOUS)

    rows = []
    for _ in range(n_rows):
        label = str(rng.choice(CLASSES, p=weights))
        parts = [str(rng.choice(OPENERS))]

        roll = rng.random()
        if roll < 0.12:
            # Ambiguous ticket: the body only supports two labels, one of which is true.
            pair = ambiguous_pairs[int(rng.integers(len(ambiguous_pairs)))]
            label = pair[int(rng.integers(2))]
            parts.append(str(rng.choice(AMBIGUOUS[pair])))
        elif roll < 0.20:
            # Vague ticket: the label is barely recoverable from the text at all.
            parts.append(str(rng.choice(VAGUE)))
        else:
            parts.append(str(rng.choice(PHRASES[label])))
            # Cross-class contamination: real tickets mention more than one thing.
            if rng.random() < 0.35:
                other = str(rng.choice([c for c in CLASSES if c != label]))
                parts.append(str(rng.choice(PHRASES[other])))

        if rng.random() < 0.3 + 0.4 * shift:
            parts.append(str(rng.choice(NOISE)))
        if shift and rng.random() < 0.8 * shift:
            parts.append(str(rng.choice(NEW_VOCABULARY)))
        if shift and rng.random() < 0.5 * shift:
            parts.append(str(rng.choice(NOISE)))

        parts.append(str(rng.choice(CLOSERS)))
        text = " ".join(part for part in parts if part).strip()

        # Label noise: whoever tagged the historical tickets was human. A dataset
        # with a perfectly clean target teaches the wrong lesson about ceilings.
        if rng.random() < 0.04:
            label = str(rng.choice([c for c in CLASSES if c != label]))

        rows.append({"text": text, "label": label})

    return pd.DataFrame(rows)
