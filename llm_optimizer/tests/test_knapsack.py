"""0/1 Knapsack (memoized): optimal selection and capacity edge cases."""
from itertools import combinations

from algorithms.knapsack import knapsack_pack


def _brute_force(weights, values, cap):
    """Reference optimum by exhaustive subset search."""
    best = 0.0
    n = len(weights)
    for r in range(n + 1):
        for combo in combinations(range(n), r):
            w = sum(weights[i] for i in combo)
            if w <= cap:
                best = max(best, sum(values[i] for i in combo))
    return best


def test_classic_optimal_selection():
    w = [400, 300, 500, 200]
    v = [9, 4, 8, 6]
    r = knapsack_pack(w, v, capacity=1024)
    assert r.total_value == _brute_force(w, v, 1024)
    assert r.total_tokens <= 1024


def test_selection_respects_capacity():
    w = [10, 20, 30]
    v = [60, 100, 120]
    r = knapsack_pack(w, v, capacity=50)
    assert r.total_tokens <= 50
    assert r.total_value == _brute_force(w, v, 50)   # == 220 (items 1+2)


def test_zero_capacity_drops_everything():
    r = knapsack_pack([5, 8], [10, 20], capacity=0)
    assert r.selected_indices == []
    assert r.total_value == 0.0
    assert r.total_tokens == 0


def test_everything_fits():
    w, v = [1, 2, 3], [5, 5, 5]
    r = knapsack_pack(w, v, capacity=100)
    assert sorted(r.selected_indices) == [0, 1, 2]
    assert r.total_tokens == 6


def test_item_heavier_than_capacity_excluded():
    r = knapsack_pack([5, 1000], [1, 999], capacity=10)
    assert 1 not in r.selected_indices
    assert r.selected_indices == [0]


def test_empty_input():
    r = knapsack_pack([], [], capacity=100)
    assert r.selected_indices == []
    assert r.total_value == 0.0


def test_memo_telemetry_reported():
    r = knapsack_pack([400, 300, 500, 200], [9, 4, 8, 6], capacity=1024)
    assert r.states_computed > 0


def test_matches_brute_force_random():
    import random
    random.seed(7)
    for _ in range(25):
        n = random.randint(1, 8)
        w = [random.randint(1, 20) for _ in range(n)]
        v = [random.randint(1, 50) for _ in range(n)]
        cap = random.randint(0, 40)
        r = knapsack_pack(w, v, cap)
        assert r.total_value == _brute_force(w, v, cap)
        assert r.total_tokens <= cap
