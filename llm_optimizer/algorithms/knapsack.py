"""
Dynamic Programming  ->  0/1 Knapsack via Memory Functions (top-down memoization),
per Levitin Ch. 8.

Each text block is an item with:
    weight = token cost
    value  = relevance score
The LLM context window is the knapsack capacity. We choose the subset of blocks
that MAXIMISES total relevance without exceeding the token budget. This is the
classic 0/1 knapsack; we solve it top-down and cache subproblem answers in a
table (the "memory function") so each (item, capacity) state is solved once.

Complexity: O(n * W) time and space, where n = #blocks, W = token capacity.
(Pseudo-polynomial: it depends on the numeric value of W.)
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class PackResult:
    selected_indices: List[int] = field(default_factory=list)
    dropped_indices: List[int] = field(default_factory=list)
    total_value: float = 0.0
    total_tokens: int = 0
    states_computed: int = 0          # how many memo cells were filled (telemetry)


def knapsack_pack(weights: List[int], values: List[float], capacity: int) -> PackResult:
    n = len(weights)
    memo = {}                          # (i, cap) -> best value using items i..n-1

    def solve(i: int, cap: int) -> float:
        if i == n or cap <= 0:
            return 0.0
        key = (i, cap)
        if key in memo:
            return memo[key]
        # Option 1: skip item i
        best = solve(i + 1, cap)
        # Option 2: take item i (only if it fits)
        if weights[i] <= cap:
            take = values[i] + solve(i + 1, cap - weights[i])
            if take > best:
                best = take
        memo[key] = best
        return best

    best_value = solve(0, capacity)

    # Reconstruct the chosen set by re-walking the decisions using the memo.
    selected, dropped = [], []
    cap = capacity
    for i in range(n):
        # If keeping item i strictly beats skipping it, it was part of the optimum.
        if weights[i] <= cap and (values[i] + solve(i + 1, cap - weights[i])) > solve(i + 1, cap):
            selected.append(i)
            cap -= weights[i]
        else:
            dropped.append(i)

    return PackResult(
        selected_indices=selected,
        dropped_indices=dropped,
        total_value=round(best_value, 2),
        total_tokens=capacity - cap,
        states_computed=len(memo),
    )


if __name__ == "__main__":
    w = [400, 300, 500, 200]
    v = [9, 4, 8, 6]
    r = knapsack_pack(w, v, capacity=1024)
    print("selected:", r.selected_indices, "value:", r.total_value,
          "tokens:", r.total_tokens, "states:", r.states_computed)
