"""
Transform & Conquer  ->  Heapsort, per Levitin Ch. 6.

We "transform" an unordered list of text snippets into a max-heap (heap
construction by sifting down from the last parental node), then repeatedly
extract the maximum to produce a fully ordered list. Total time O(n log n),
in-place, no extra arrays.

Each element is a (key, payload) pair where key = relevance score. We sort
DESCENDING (highest relevance first) so the result feeds straight into the
knapsack packer.
"""
from dataclasses import dataclass
from typing import Any, List, Tuple


def _sift_down(a: List[Tuple[float, Any]], i: int, n: int):
    """Restore the max-heap property for subtree rooted at i within a[0..n-1]."""
    while True:
        largest = i
        left, right = 2 * i + 1, 2 * i + 2
        if left < n and a[left][0] > a[largest][0]:
            largest = left
        if right < n and a[right][0] > a[largest][0]:
            largest = right
        if largest == i:
            return
        a[i], a[largest] = a[largest], a[i]
        i = largest


def build_max_heap(a: List[Tuple[float, Any]]):
    """Bottom-up heap construction: sift down every parental node. O(n)."""
    n = len(a)
    for i in range(n // 2 - 1, -1, -1):
        _sift_down(a, i, n)


def heapsort_desc(items: List[Tuple[float, Any]]) -> List[Tuple[float, Any]]:
    """Return items sorted by key DESCENDING using classic heapsort.
    Implementation: heapsort naturally yields ascending order in place
    (repeatedly swap root to the end and shrink the heap), so we reverse."""
    a = list(items)
    n = len(a)
    build_max_heap(a)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]   # move current max to its final slot
        _sift_down(a, 0, end)         # restore heap over the shrunken range
    a.reverse()                       # ascending -> descending
    return a


@dataclass
class Snippet:
    idx: int          # original block index (used as recency proxy)
    text: str
    tokens: int       # weight for the knapsack
    score: float      # relevance value


def prioritize(snippets: List[Snippet]) -> List[Snippet]:
    keyed = [(s.score, s) for s in snippets]
    ordered = heapsort_desc(keyed)
    return [s for _, s in ordered]


if __name__ == "__main__":
    data = [Snippet(i, f"block {i}", 10 + i, sc)
            for i, sc in enumerate([3, 9, 1, 7, 5])]
    for s in prioritize(data):
        print(s.idx, s.score)
