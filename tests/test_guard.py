import numpy as np
import pytest

import bench
import guard


@pytest.mark.parametrize(
    "a,b",
    [
        (
            "What is the fee for orders over 100 dollars?",
            "What is the fee for orders over 500 dollars?",
        ),
        ("Is the API available in 2026?", "Is the API available in 2025?"),
        ("Can I cancel my subscription?", "Can I not cancel my subscription?"),
        ("How do I move from Azure to AWS?", "How do I move from AWS to Azure?"),
        ("What is the best hotel in Paris?", "What is the best hotel in London?"),
    ],
)
def test_blocks_lookalikes_with_different_meaning(a, b):
    assert guard.blocks(a, b)


@pytest.mark.parametrize(
    "a,b",
    [
        ("How do I reset my password?", "What is the way to reset a password?"),
        ("What is the capital of France?", "what is the capital of France"),
        ("Fee for 100 dollars?", "What is the fee on 100 dollars?"),
    ],
)
def test_allows_real_paraphrases(a, b):
    assert not guard.blocks(a, b)


def test_double_negation_is_treated_as_no_negation():
    assert not guard.negation_differs("Is it not never allowed?", "Is it allowed?")


def test_identical_text_is_never_a_word_order_trap():
    assert not guard.word_order_trap("a b c", "a b c")


def test_sweep_counts_only_allowed_hits():
    sim = np.array([0.95, 0.95, 0.5, 0.95])
    labels = np.array([1, 0, 1, 0])
    no_guard = bench.sweep(sim, labels, np.array([True] * 4))
    guarded = bench.sweep(sim, labels, np.array([True, False, True, True]))
    row = {r["threshold"]: r for r in no_guard}[0.9]
    assert row["tpr"] == 0.5 and row["fpr"] == 1.0
    assert {r["threshold"]: r for r in guarded}[0.9]["fpr"] == 0.5


def test_precision_uses_the_base_rate():
    sim, labels = np.array([0.95, 0.5, 0.95, 0.5]), np.array([1, 1, 0, 0])
    row = {r["threshold"]: r for r in bench.sweep(sim, labels, np.array([True] * 4))}[0.9]
    # tpr .5, fpr .5, so precision equals the base rate
    assert row["precision_at_20pct_repeats"] == pytest.approx(0.2)


def test_auc_perfect_and_inverted():
    labels = np.array([1, 1, 0, 0])
    assert bench.auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 1.0
    assert bench.auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 0.0
