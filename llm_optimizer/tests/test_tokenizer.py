"""Tokenizer: valid counts, monotonicity, empty handling, backend report."""
from tokenizer import count_tokens, tokenizer_backend


def test_empty_is_zero():
    assert count_tokens("") == 0


def test_positive_count_for_text():
    assert count_tokens("Hello world, this is a test.") > 0


def test_longer_text_more_tokens():
    short = count_tokens("budget")
    longer = count_tokens("budget review meeting next tuesday afternoon agenda")
    assert longer > short


def test_count_is_int():
    assert isinstance(count_tokens("some text"), int)


def test_backend_reported():
    backend = tokenizer_backend()
    assert backend.startswith("tiktoken:") or backend.startswith("heuristic")


def test_reasonable_magnitude():
    # ~ one sentence should land in a sane token range regardless of backend.
    n = count_tokens("The Q3 budget review is scheduled for next Tuesday.")
    assert 5 <= n <= 40
