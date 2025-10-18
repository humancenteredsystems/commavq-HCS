"""
Canonical scan order utilities.

Defines a frozen spatial permutation named "hilbert_8x16_v1" and exposes a
manifest-ready description for inclusion in archive manifests.

The Hilbert mapping uses the standard d -> (x,y) algorithm for a 2^k grid.
We compute the 16x16 (order=4) Hilbert curve, then filter to the top half
(rows y in 0..7) to produce exactly 128 spatial positions for the 8x16 token
grid. The permutation is expressed as row-major indices (y*16 + x).

Functions:
- get_spatial_permutation(name) -> list[int]
- get_scan_order_manifest() -> dict suitable for writing into manifest["scan_order"]
- permutation_sha256_bytes(permutation) -> hex digest (sha256 of LE int32 bytes)
"""
from typing import List, Dict
import struct
import hashlib

# Metadata for the canonical spatial permutation
SPATIAL_NAME = "hilbert_8x16_v1"
WIDTH = 16
HEIGHT = 8
_FULL_SIZE = 16  # 16x16 hilbert then filter top half


def _rot(n: int, x: int, y: int, rx: int, ry: int):
    """Rotate/flip a quadrant appropriately (helper for d2xy)."""
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        # swap x and y
        x, y = y, x
    return x, y


def _d2xy(n: int, d: int):
    """
    Convert Hilbert distance d to (x, y) on an n x n grid (n power of 2).
    Standard algorithm (see Wikipedia / Hilbert curve).
    """
    x = 0
    y = 0
    t = d
    s = 1
    while s < n:
        rx = (t // 2) & 1
        ry = (t ^ rx) & 1
        x, y = _rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def _build_hilbert_16x16_top_half() -> List[int]:
    """
    Build the 16x16 Hilbert order (d=0..255), map to (x,y), filter to y in [0..7],
    and return list of row-major indices (y*16 + x) length 128.
    """
    n = _FULL_SIZE
    coords = []
    for d in range(n * n):
        x, y = _d2xy(n, d)
        if 0 <= y < HEIGHT and 0 <= x < WIDTH:
            coords.append((x, y))
    # Convert to row-major index
    permutation = [y * WIDTH + x for (x, y) in coords]
    return permutation


# Precompute the frozen permutation once
_HILBERT_8x16_V1 = _build_hilbert_16x16_top_half()


def get_spatial_permutation(name: str) -> List[int]:
    """
    Return the permutation list for the given spatial permutation name.
    Currently supports "hilbert_8x16_v1".
    """
    if name == SPATIAL_NAME:
        return list(_HILBERT_8x16_V1)
    raise ValueError(f"Unknown spatial permutation name: {name}")


def permutation_sha256_bytes(permutation: List[int]) -> str:
    """
    Compute sha256 over little-endian int32 bytes of the permutation list.
    This produces a compact, stable fingerprint for the permutation.
    """
    b = b"".join(struct.pack("<i", int(x)) for x in permutation)
    return hashlib.sha256(b).hexdigest()


def get_scan_order_manifest() -> Dict:
    """
    Return the scan_order manifest dictionary to be embedded in archive manifest.json.
    """
    permutation = get_spatial_permutation(SPATIAL_NAME)
    sha = permutation_sha256_bytes(permutation)
    return {
        "traversal": "frame_major",
        "frame_order": "increasing",
        "spatial": {
            "name": SPATIAL_NAME,
            "width": WIDTH,
            "height": HEIGHT,
            "permutation_kind": "row_major_index",
            "permutation_sha256": sha,
            "permutation": permutation,
        },
    }


# Export convenience alias
SCAN_ORDER_MANIFEST = get_scan_order_manifest()
