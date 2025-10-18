"""
Byte-range coder (LZMA-style) replacement for Stage 1 order-0 baseline.

This implementation replaces the previous ad-hoc arithmetic coder with a
byte-range coder that avoids E3 underflow handling by keeping `range`
above a renormalization threshold (1<<24) and emitting top bytes of `low`.
It uses a simple header format (same as before) so archive layout is unchanged.

Header format (same as previous):
- uint32 symbols_count
- uint16 precision
- uint16 alphabet
- (alphabet+1) uint32 little-endian cumulative counts

API (keeps previous names):
- build_cdf_from_counts(counts: List[int], precision: int = 16) -> List[int]
- encode_with_cdf(symbols: List[int], cdf: List[int], precision: int = 16) -> bytes
- decode_with_cdf(bytestream: bytes) -> List[int]

Notes:
- This coder is written for correctness and clarity. Performance is acceptable
  for modest-scale tests; we can optimize later (numpy/buffered, cython).
- The CDF builder is identical in behavior to the previous file (deterministic
  scaling into 2^precision, non-zero frequency for observed symbols).
"""
from typing import List, Tuple
import struct
import math
import io
import bisect

# parameters for byte-range coder
_PRECISION_DEFAULT = 16
_RENORM_BITS = 24
_RENORM_THRESHOLD = 1 << _RENORM_BITS
_MASK32 = (1 << 32) - 1


def build_cdf_from_counts(counts: List[int], precision: int = _PRECISION_DEFAULT) -> List[int]:
    """
    Same deterministic CDF builder: scale counts to fit into 2^precision and
    ensure non-zero entries for observed symbols.
    """
    total = sum(counts)
    if total == 0:
        alphabet = len(counts)
        cdf = [0]
        for _ in range(alphabet):
            cdf.append(cdf[-1] + 1)
        return cdf

    scale = (1 << precision)
    raw = [c * scale / total for c in counts]
    # assign at least 1 to non-zero counts, 0 to zero counts
    ints = [max(1, int(math.floor(raw[i]))) if counts[i] > 0 else 0 for i in range(len(counts))]

    s = sum(ints)
    # distribute residue deterministically by fractional parts
    if s < scale:
        rem = scale - s
        fracs = [raw[i] - math.floor(raw[i]) for i in range(len(raw))]
        order = sorted(range(len(raw)), key=lambda k: (-fracs[k], k))
        for idx in order:
            if rem <= 0:
                break
            if counts[idx] == 0:
                continue
            ints[idx] += 1
            rem -= 1
        # if still remainder (many zero counts), distribute cyclically
        idx = 0
        while rem > 0:
            if counts[idx] > 0:
                ints[idx] += 1
                rem -= 1
            idx = (idx + 1) % len(ints)

    # Build monotone cumulative
    cdf = [0]
    for v in ints:
        cdf.append(cdf[-1] + v)

    # As a last resort, if total still not equal scale, rescale linearly (shouldn't happen)
    if cdf[-1] != scale:
        if cdf[-1] == 0:
            # uniform
            cdf = [0]
            for _ in counts:
                cdf.append(cdf[-1] + 1)
        else:
            # scale proportionally to fill
            scale_factor = scale / cdf[-1]
            new_cdf = [0]
            acc = 0
            for i in range(len(counts)):
                acc += max(1, int(max(0, round((cdf[i+1] - cdf[i]) * scale_factor))))
                new_cdf.append(acc)
            # fix any discrepancy
            diff = scale - new_cdf[-1]
            j = 0
            while diff > 0:
                if counts[j] > 0:
                    new_cdf[j+1] += 1
                    for k in range(j+2, len(new_cdf)):
                        new_cdf[k] += 1
                    diff -= 1
                j = (j + 1) % len(counts)
            cdf = new_cdf

    return cdf


def _write_header(stream: io.BytesIO, symbols_count: int, precision: int, alphabet: int, cdf: List[int]):
    stream.write(struct.pack("<I", symbols_count))
    stream.write(struct.pack("<H", precision))
    stream.write(struct.pack("<H", alphabet))
    for v in cdf:
        stream.write(struct.pack("<I", int(v)))


def _read_header(byts: bytes, offset: int = 0) -> Tuple[int, int, int, List[int], int]:
    if offset + 8 > len(byts):
        raise ValueError("Stream too short for header")
    symbols_count = struct.unpack_from("<I", byts, offset)[0]; offset += 4
    precision = struct.unpack_from("<H", byts, offset)[0]; offset += 2
    alphabet = struct.unpack_from("<H", byts, offset)[0]; offset += 2
    cdf = []
    for _ in range(alphabet + 1):
        if offset + 4 > len(byts):
            raise ValueError("Stream too short for cdf")
        v = struct.unpack_from("<I", byts, offset)[0]
        cdf.append(int(v))
        offset += 4
    return symbols_count, precision, alphabet, cdf, offset


def encode_with_cdf(symbols: List[int], cdf: List[int], precision: int = _PRECISION_DEFAULT) -> bytes:
    """
    Byte-range encoder.
    """
    alphabet = len(cdf) - 1
    total = cdf[-1]
    if total <= 0:
        raise ValueError("CDF total must be positive")

    out = io.BytesIO()
    _write_header(out, len(symbols), precision, alphabet, cdf)

    low = 0
    range_ = 0xFFFFFFFF  # full 32-bit range, use Python int for headroom

    for sym in symbols:
        if sym < 0 or sym >= alphabet:
            raise ValueError(f"Symbol {sym} out of range")
        # compute interval
        cum_low = cdf[sym]
        cum_high = cdf[sym + 1]
        # split range according to totals
        step = range_ // total
        new_low = low + step * cum_low
        new_high = low + step * cum_high - 1
        low = new_low
        range_ = new_high - new_low + 1
        # renormalize
        while range_ <= (1 << _RENORM_BITS) - 1:
            # emit top byte of low (big-endian)
            b = (low >> 24) & 0xFF
            out.write(bytes((b,)))
            low = (low << 8) & 0xFFFFFFFFFFFFFFFF  # keep it wide
            range_ = (range_ << 8) & 0xFFFFFFFFFFFFFFFF
    # flush remaining bytes (emit 5 bytes of low big-endian to be safe)
    for shift in (32, 24, 16, 8, 0):
        out.write(bytes(((low >> shift) & 0xFF,)))
    return out.getvalue()


def decode_with_cdf(bytestream: bytes) -> List[int]:
    """
    Decode a stream produced by encode_with_cdf.
    """
    symbols_count, precision, alphabet, cdf, offset = _read_header(bytestream, 0)
    total = cdf[-1]
    if total <= 0:
        raise ValueError("CDF total must be positive")

    # read initial 5 bytes to form code (we wrote 5)
    if offset + 5 > len(bytestream):
        raise ValueError("Encoded stream too short")
    code = 0
    for i in range(5):
        code = (code << 8) | bytestream[offset + i]
    offset += 5

    low = 0
    range_ = 0xFFFFFFFF
    ptr = offset
    symbols = []

    for _ in range(symbols_count):
        step = range_ // total
        if step == 0:
            # degenerate; fallback to safe behavior
            scaled = 0
        else:
            scaled = (code - low) // step
            if scaled < 0:
                scaled = 0
        # find symbol via binary search on cdf
        # cdf is sorted; use bisect
        # scaled in [0..total-1], find s so that cdf[s] <= scaled < cdf[s+1]
        s = bisect.bisect_right(cdf, scaled) - 1
        if s < 0:
            s = 0
        if s >= alphabet:
            s = alphabet - 1

        cum_low = cdf[s]
        cum_high = cdf[s + 1]
        new_low = low + step * cum_low
        new_high = low + step * cum_high - 1
        low = new_low
        range_ = new_high - new_low + 1

        # renormalize: pull bytes while range_ small
        while range_ <= (1 << _RENORM_BITS) - 1:
            # shift in next byte
            if ptr < len(bytestream):
                next_byte = bytestream[ptr]
                ptr += 1
            else:
                next_byte = 0
            code = ((code << 8) & 0xFFFFFFFFFFFFFFFF) | next_byte
            low = (low << 8) & 0xFFFFFFFFFFFFFFFF
            range_ = (range_ << 8) & 0xFFFFFFFFFFFFFFFF
        symbols.append(s)
    return symbols
