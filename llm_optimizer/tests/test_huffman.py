"""Huffman: encode/decode round-trip, archive serialisation, edge cases."""
import struct

from algorithms.huffman import (
    huffman_compress, compress, decompress, verify_roundtrip,
)


def test_basic_roundtrip():
    text = "the quick brown fox jumps over the lazy dog"
    assert decompress(compress(text)) == text
    assert verify_roundtrip(text) is True


def test_empty_text():
    assert decompress(compress("")) == ""
    assert verify_roundtrip("") is True


def test_single_symbol_edge_case():
    text = "aaaaaaaaaa"
    assert verify_roundtrip(text) is True
    assert decompress(compress(text)) == text


def test_unicode_roundtrip():
    text = "héllo wörld — café ✓ 日本語 🚀"
    assert decompress(compress(text)) == text


def test_newlines_and_whitespace():
    text = "line one\n\nline two\twith tabs   and spaces\n"
    assert decompress(compress(text)) == text


def test_archive_is_self_describing():
    text = "compress me please"
    archive = compress(text)
    assert archive[:4] == b"HUF1"             # magic
    (hdr_len,) = struct.unpack(">I", archive[4:8])
    assert hdr_len > 0
    # Decoding uses only the archive bytes — no external state.
    assert decompress(archive) == text


def test_archive_smaller_than_naive_for_repetitive():
    text = "ab" * 500
    archive = compress(text)
    # Payload should be well under the 8-bits/char original for skewed input.
    assert len(archive) < len(text.encode("utf-8"))


def test_compress_reports_positive_ratio():
    r = huffman_compress("aaaaaaaaaabbbbbccc")
    assert r.ratio > 0
    assert r.compressed_bits < r.original_bits


def test_bad_magic_raises():
    import pytest
    with pytest.raises(ValueError):
        decompress(b"NOPE" + b"\x00" * 8)


def test_long_text_roundtrip():
    text = ("Design and analysis of algorithms. " * 100)
    assert verify_roundtrip(text) is True
