"""Tiny dependency-free PNG writer (RGB uint8)."""
import struct
import zlib

import numpy as np


def write_png(path, rgb: np.ndarray, scale: int = 1) -> None:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if scale > 1:
        rgb = rgb.repeat(scale, 0).repeat(scale, 1)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)
