"""Cheap mismatch checks to run after the similarity check passes. Stdlib only."""

from __future__ import annotations

import re

NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
NEGATION = re.compile(
    r"\b(not|no|never|without|cannot|can't|don't|doesn't|isn't|won't|shouldn't)\b", re.I
)
# Words that start with a capital somewhere other than the start of the sentence.
PROPER = re.compile(r"(?<=[a-z0-9,] )[A-Z][a-zA-Z0-9]+")


def numbers_differ(a: str, b: str) -> bool:
    return set(NUMBER.findall(a)) != set(NUMBER.findall(b))


def negation_differs(a: str, b: str) -> bool:
    return (len(NEGATION.findall(a)) % 2) != (len(NEGATION.findall(b)) % 2)


def names_differ(a: str, b: str) -> bool:
    return set(PROPER.findall(a)) != set(PROPER.findall(b))


def word_order_trap(a: str, b: str) -> bool:
    """Same words, different order: the case a bag-of-meanings embedding is worst at."""
    x, y = re.findall(r"\w+", a.lower()), re.findall(r"\w+", b.lower())
    return x != y and sorted(x) == sorted(y)


def blocks(a: str, b: str) -> bool:
    """True when a similarity hit should be refused and sent to the model."""
    return (
        numbers_differ(a, b)
        or negation_differs(a, b)
        or names_differ(a, b)
        or word_order_trap(a, b)
    )
