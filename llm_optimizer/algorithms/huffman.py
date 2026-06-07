"""
Greedy Technique  ->  Huffman Coding, per Levitin Ch. 9.

We build optimal prefix-free codes by a greedy rule: repeatedly merge the two
lowest-frequency nodes into a parent until one tree remains. Characters that
occur often get short codes; rare ones get long codes, minimising total encoded
length.

Frequency counting is done with a frequency array / dictionary (the "counting
sort" style tally the brief mentions). We then report real compression
telemetry: original size, encoded size, and percentage saved.
"""
import heapq
import json
import struct
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass(order=True)
class _Node:
    freq: int
    order: int                       # tie-breaker to keep heap deterministic
    char: Optional[str] = field(compare=False, default=None)
    left: "Optional[_Node]" = field(compare=False, default=None)
    right: "Optional[_Node]" = field(compare=False, default=None)


@dataclass
class HuffmanResult:
    codes: Dict[str, str]
    encoded_bits: str
    original_bits: int               # 8 bits per UTF-8 byte of source
    compressed_bits: int             # length of the packed payload
    tree_lines: List[str]            # printable tree for the demo panel
    @property
    def ratio(self) -> float:
        if self.original_bits == 0:
            return 0.0
        return 100.0 * (1 - self.compressed_bits / self.original_bits)


def _count_frequencies(text: str) -> Counter:
    return Counter(text)             # tally pass over the characters


def _build_tree(freqs: Counter) -> Optional[_Node]:
    heap, order = [], 0
    for ch, f in freqs.items():
        heapq.heappush(heap, _Node(f, order, char=ch))
        order += 1
    if not heap:
        return None
    while len(heap) > 1:
        a = heapq.heappop(heap)      # two least-frequent (greedy choice)
        b = heapq.heappop(heap)
        heapq.heappush(heap, _Node(a.freq + b.freq, order, left=a, right=b))
        order += 1
    return heap[0]


def _assign_codes(node: Optional[_Node]) -> Dict[str, str]:
    if node is None:
        return {}
    codes: Dict[str, str] = {}
    # Single-symbol edge case: give it a 1-bit code.
    if node.char is not None:
        return {node.char: "0"}

    def walk(n: _Node, prefix: str):
        if n.char is not None:
            codes[n.char] = prefix or "0"
            return
        if n.left:
            walk(n.left, prefix + "0")
        if n.right:
            walk(n.right, prefix + "1")

    walk(node, "")
    return codes


def _tree_lines(node: Optional[_Node]) -> List[str]:
    lines: List[str] = []

    def render(n: Optional[_Node], prefix: str, label: str):
        if n is None:
            return
        if n.char is not None:
            shown = repr(n.char) if n.char.strip() else repr(n.char)
            lines.append(f"{prefix}{label}leaf {shown}  (freq {n.freq})")
        else:
            lines.append(f"{prefix}{label}node (freq {n.freq})")
            render(n.left, prefix + "   ", "0> ")
            render(n.right, prefix + "   ", "1> ")

    render(node, "", "")
    return lines


def huffman_compress(text: str) -> HuffmanResult:
    freqs = _count_frequencies(text)
    root = _build_tree(freqs)
    codes = _assign_codes(root)
    encoded = "".join(codes[ch] for ch in text) if codes else ""
    original_bits = len(text.encode("utf-8")) * 8
    return HuffmanResult(
        codes=codes,
        encoded_bits=encoded,
        original_bits=original_bits,
        compressed_bits=len(encoded),
        tree_lines=_tree_lines(root),
    )


def pack_to_bytes(encoded_bits: str) -> bytes:
    """Pack the bit-string into real bytes (so we can write a binary archive).
    Legacy payload-only packer kept for backward compatibility; the
    self-describing archive (compress/decompress below) is preferred."""
    pad = (8 - len(encoded_bits) % 8) % 8
    bits = encoded_bits + "0" * pad
    out = bytearray()
    out.append(pad)                                  # store padding in the header
    for i in range(0, len(bits), 8):
        out.append(int(bits[i:i + 8], 2))
    return bytes(out)


# ---------------------------------------------------------------------------
# Improvement 3 & 4 — self-contained .huff archive (serialise codes + metadata)
# and exact, bit-perfect decompression.
#
# Archive layout (all big-endian):
#   magic    : 4 bytes  b"HUF1"
#   hdr_len  : 4 bytes  unsigned int  -> length of the JSON header
#   header   : hdr_len bytes  UTF-8 JSON {"codes", "padding", "original_length"}
#   payload  : remaining bytes  -> the packed Huffman bit-stream
# The header carries the full code table, so the archive is fully self-describing
# and can be decoded with nothing but the file itself.
# ---------------------------------------------------------------------------

_MAGIC = b"HUF1"


def compress(text: str) -> bytes:
    """Compress UTF-8 `text` into a complete, self-contained .huff archive
    (magic + JSON header with the code table + packed payload bytes)."""
    freqs = _count_frequencies(text)
    root = _build_tree(freqs)
    codes = _assign_codes(root)
    encoded = "".join(codes[ch] for ch in text) if codes else ""

    pad = (8 - len(encoded) % 8) % 8 if encoded else 0
    bits = encoded + "0" * pad
    payload = bytearray()
    for i in range(0, len(bits), 8):
        payload.append(int(bits[i:i + 8], 2))

    header = json.dumps(
        {"codes": codes, "padding": pad, "original_length": len(text)},
        ensure_ascii=False,
    ).encode("utf-8")

    out = bytearray()
    out += _MAGIC
    out += struct.pack(">I", len(header))
    out += header
    out += bytes(payload)
    return bytes(out)


def decompress(archive: bytes) -> str:
    """Decode a .huff archive produced by `compress` back to the exact original
    UTF-8 text. Bit-perfect; raises ValueError on a malformed archive."""
    if archive[:4] != _MAGIC:
        raise ValueError("Not a HUF1 archive (bad magic bytes).")
    (hdr_len,) = struct.unpack(">I", archive[4:8])
    header = json.loads(archive[8:8 + hdr_len].decode("utf-8"))
    codes = header["codes"]
    padding = header["padding"]
    original_length = header["original_length"]

    payload = archive[8 + hdr_len:]
    if not codes or original_length == 0:
        return ""

    # Unpack bytes -> bit-string, dropping the trailing padding bits.
    bits = "".join(f"{byte:08b}" for byte in payload)
    if padding:
        bits = bits[:-padding] if padding <= len(bits) else bits

    # Invert the code table for prefix-free decoding.
    decoder = {code: ch for ch, code in codes.items()}

    # Single-symbol edge case: code table maps one char to "0"; emit it N times.
    if len(decoder) == 1:
        only_char = next(iter(codes))
        return only_char * original_length

    out = []
    buf = ""
    for bit in bits:
        buf += bit
        ch = decoder.get(buf)
        if ch is not None:
            out.append(ch)
            buf = ""
            if len(out) == original_length:
                break
    return "".join(out)


def verify_roundtrip(text: str) -> bool:
    """Compress then decompress `text` and confirm bit-perfect recovery."""
    return decompress(compress(text)) == text


if __name__ == "__main__":
    r = huffman_compress("the quick brown fox the the the")
    print("ratio: %.1f%%" % r.ratio, "bits", r.compressed_bits, "/", r.original_bits)
    print("\n".join(r.tree_lines))

    print("\n--- archive round-trip self-test ---")
    for sample in [
        "the quick brown fox the the the",
        "aaaaaaa",                 # single-symbol edge case
        "",                        # empty
        "héllo wörld — UTF-8 ✓",   # non-ASCII
    ]:
        arc = compress(sample)
        ok = verify_roundtrip(sample)
        print(f"  ok={ok}  archive={len(arc):>3}B  src={len(sample.encode('utf-8')):>3}B  {sample!r}")
