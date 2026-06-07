"""Heapsort: correct descending order, duplicate values, stability of contents."""
import random

from algorithms.heapsort import heapsort_desc, prioritize, Snippet


def test_descending_order():
    items = [(3, "c"), (9, "a"), (1, "e"), (7, "b"), (5, "d")]
    out = heapsort_desc(items)
    keys = [k for k, _ in out]
    assert keys == sorted(keys, reverse=True)
    assert keys == [9, 7, 5, 3, 1]


def test_duplicate_values():
    items = [(5, "a"), (5, "b"), (5, "c"), (2, "d"), (5, "e")]
    out = heapsort_desc(items)
    keys = [k for k, _ in out]
    assert keys == [5, 5, 5, 5, 2]
    # All payloads preserved (no loss/duplication).
    assert {p for _, p in out} == {"a", "b", "c", "d", "e"}


def test_single_and_empty():
    assert heapsort_desc([]) == []
    assert heapsort_desc([(42, "x")]) == [(42, "x")]


def test_large_random_matches_builtin():
    data = [(random.randint(0, 1000), i) for i in range(500)]
    out = heapsort_desc(list(data))
    keys = [k for k, _ in out]
    assert keys == sorted([k for k, _ in data], reverse=True)


def test_prioritize_orders_snippets():
    snips = [Snippet(0, "a", 10, 3.0), Snippet(1, "b", 10, 9.0),
             Snippet(2, "c", 10, 5.0)]
    ordered = prioritize(snips)
    assert [s.score for s in ordered] == [9.0, 5.0, 3.0]
    assert [s.idx for s in ordered] == [1, 2, 0]
