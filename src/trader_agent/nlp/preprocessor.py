"""Text preprocessing utilities for financial NLP -- cleaning, normalization, tokenization."""

from __future__ import annotations


def clean_text(raw: str) -> str:
    """Strip HTML, normalize whitespace, and remove boilerplate from financial text."""
    raise NotImplementedError


def normalize_ticker_mentions(text: str) -> str:
    """Standardize $TICKER and cashtag mentions to a canonical form."""
    raise NotImplementedError
