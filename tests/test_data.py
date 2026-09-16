"""Data contract tests: catch a broken corpus before it becomes a broken model."""

from __future__ import annotations

import pytest

from tickets import config, data


def test_schema_and_labels():
    df = data.generate(n_rows=500)
    assert list(df.columns) == ["text", "label"]
    assert set(df["label"]).issubset(set(config.CLASSES))
    assert (df["text"].str.len() >= config.MIN_TEXT_CHARS).all()


def test_all_classes_present_and_balanced():
    shares = data.generate(n_rows=4_000)["label"].value_counts(normalize=True)
    assert set(shares.index) == set(config.CLASSES)
    assert shares.min() > 0.15, "a class this rare makes macro-F1 unstable"


def test_generation_is_deterministic():
    assert data.generate(n_rows=200).equals(data.generate(n_rows=200))


def test_texts_are_not_duplicated_into_uselessness():
    """If the corpus is mostly repeats, the test split leaks into training."""
    df = data.generate(n_rows=2_000)
    assert df["text"].nunique() / len(df) > 0.5


@pytest.mark.parametrize("shift", [0.5, 1.0])
def test_shift_changes_vocabulary_length_and_class_mix(shift):
    base = data.generate(n_rows=2_000, seed=1)
    shifted = data.generate(n_rows=2_000, seed=1, shift=shift)

    base_vocab = set(" ".join(base["text"]).split())
    shifted_vocab = set(" ".join(shifted["text"]).split())
    assert shifted_vocab - base_vocab, "shifted data must introduce unseen vocabulary"
    assert shifted["text"].str.len().mean() > base["text"].str.len().mean()
    assert shifted["label"].value_counts(normalize=True).max() > (
        base["label"].value_counts(normalize=True).max()
    )
