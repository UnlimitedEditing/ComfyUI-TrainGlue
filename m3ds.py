"""M3DS container v1 (same format as Meshsmuggler, so unsmuggle.html decodes it): bytes <-> lossless RGB PNG chunks.

Header (big endian, 40 bytes) + filename + payload, laid out as a near-square RGB uint8 image per chunk.
Pure numpy/PIL so it can be tested and used outside ComfyUI.
"""
import gzip
import math
import struct
import zlib

import numpy as np

MAGIC = b"M3DS"
VERSION = 1
FLAG_GZIP = 0x01
HEADER_FMT = ">4sBBHHQIIQIH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
CHANNELS = 3


def _to_rgb(raw, width=None):
    pad = (-len(raw)) % CHANNELS
    if pad:
        raw += b"\x00" * pad
    n_pixels = max(1, len(raw) // CHANNELS)
    if not raw:
        raw = b"\x00" * CHANNELS
    w = width or max(1, math.ceil(math.sqrt(n_pixels)))
    h = math.ceil(n_pixels / w)
    raw += b"\x00" * ((w * h - n_pixels) * CHANNELS)
    return np.frombuffer(raw, dtype=np.uint8).reshape((h, w, CHANNELS)).copy()


def pack(data, filename, max_dimension=8192, compress=False):
    """bytes -> list of HxWx3 uint8 arrays (all chunks padded to the same size so they can batch)."""
    orig_crc = zlib.crc32(data) & 0xFFFFFFFF
    flags, blob = 0, data
    if compress:
        blob, flags = gzip.compress(data, compresslevel=6), FLAG_GZIP
    fname = filename.encode("utf-8")
    capacity = int(max_dimension) ** 2 * CHANNELS - HEADER_SIZE - len(fname)
    payloads = [blob[i:i + capacity] for i in range(0, len(blob), capacity)] or [b""]
    if len(payloads) > 0xFFFF:
        raise ValueError("too many chunks; raise max_dimension")
    raws = []
    for idx, p in enumerate(payloads):
        head = struct.pack(HEADER_FMT, MAGIC, VERSION, flags, idx, len(payloads), len(blob), len(p),
                           zlib.crc32(p) & 0xFFFFFFFF, len(data), orig_crc, len(fname))
        raws.append(head + fname + p)
    width = max(1, math.ceil(math.sqrt(max(len(r) for r in raws) / CHANNELS)))
    return [_to_rgb(r, width) for r in raws]


def unpack(arrays):
    """list of HxWx3 uint8 arrays -> (bytes, filename). Verifies every CRC."""
    parsed = []
    for i, arr in enumerate(arrays):
        flat = np.ascontiguousarray(arr).reshape(-1).tobytes()
        (magic, ver, flags, idx, count, blob_total, plen, pcrc, olen, ocrc, flen) = struct.unpack(
            HEADER_FMT, flat[:HEADER_SIZE])
        if magic != MAGIC or ver != VERSION:
            raise ValueError(f"image[{i}]: not an M3DS v1 image")
        pos = HEADER_SIZE
        name = flat[pos:pos + flen].decode("utf-8", "replace")
        payload = flat[pos + flen:pos + flen + plen]
        if len(payload) != plen or (zlib.crc32(payload) & 0xFFFFFFFF) != pcrc:
            raise ValueError(f"image[{i}]: payload truncated or altered (CRC mismatch)")
        parsed.append((idx, count, flags, blob_total, olen, ocrc, name, payload))
    parsed.sort(key=lambda t: t[0])
    idx0, count, flags, blob_total, olen, ocrc, name, _ = parsed[0]
    if [t[0] for t in parsed] != list(range(count)):
        raise ValueError(f"need chunks 0..{count - 1}, got {[t[0] for t in parsed]}")
    blob = b"".join(t[7] for t in parsed)
    data = gzip.decompress(blob) if flags & FLAG_GZIP else blob
    if len(data) != olen or (zlib.crc32(data) & 0xFFFFFFFF) != ocrc:
        raise ValueError("final length/CRC mismatch")
    return data, name or "recovered.bin"
