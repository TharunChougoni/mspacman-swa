"""Maze path distance from Ms. Pac-Man to the nearest remaining dot, from the RGB screen (maze-agnostic).

walk mask: playfield rows [0, 172) of the 160-px screen; a pixel is walkable unless it is wall colour (the most common
non-background colour) in a LARGE component (dots/pellets are small components of the same colour = walkable).
Only the walkable component containing the most corridor pixels is kept (closed ghost-house interior excluded).
Rows whose pixel 0 and 159 are both walkable are joined (side tunnels / wraparound).
dist_field(walk, sources): 4-connected BFS in pixels, vectorised with cv2.dilate (one ring per iteration).
"""
from __future__ import annotations

import cv2
import numpy as np

H = 172
BG = (0, 28, 136)
K = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
INF = 30000


def wall_colour(rgb):
    pk = rgb[:H:4, ::4].astype(np.int32); pk = (pk[..., 0] << 16) | (pk[..., 1] << 8) | pk[..., 2]
    v, cnt = np.unique(pk[pk != (BG[0] << 16 | BG[1] << 8 | BG[2])], return_counts=True)
    c = int(v[np.argmax(cnt)]); return np.array([c >> 16, (c >> 8) & 255, c & 255], np.uint8)


def masks(rgb):
    """-> walk (H,160) bool, dots (H,160) bool, dot centres (K,2) float (x, y)."""
    rgb = rgb[:H]; col = wall_colour(rgb)
    w = (rgb == col).all(-1).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(w, connectivity=4)
    small = (st[:, cv2.CC_STAT_AREA] <= 8) & (np.arange(n) > 0)
    dots = small[lab] & (w > 0)
    walk = (w == 0) | dots
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(walk.astype(np.uint8), connectivity=4)
    if n2 > 2:
        big = 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))
        walk = lab2 == big
        dots &= walk
    return walk, dots, cen[np.flatnonzero(small)]


def dist_field(walk, sources):
    d = np.full(walk.shape, INF, np.int32)
    fr = (sources & walk).astype(np.uint8)
    if not fr.any():
        return d
    seen = fr.copy(); d[fr > 0] = 0; k = 0
    wrap = walk[:, 0] & walk[:, -1]
    w8 = walk.astype(np.uint8)
    while True:
        k += 1
        nx = cv2.dilate(fr, K)
        nx[:, 0] |= fr[:, -1] & wrap; nx[:, -1] |= fr[:, 0] & wrap
        nx &= w8; nx &= 1 - seen
        if not nx.any():
            return d
        d[nx > 0] = k; seen |= nx; fr = nx


def player_px(x, y, walk):
    """RAM x/y -> nearest walkable screen pixel (row, col)."""
    sx, sy = int(round(x - 8.3)), int(round(y + 5.8))
    sx = min(max(sx, 0), 159); sy = min(max(sy, 0), H - 1)
    if walk[sy, sx]:
        return sy, sx
    ys, xs = np.nonzero(walk[max(0, sy - 6):sy + 7, max(0, sx - 6):sx + 7])
    if not len(ys):
        return sy, sx
    j = np.argmin(np.abs(ys - min(6, sy)) + np.abs(xs - min(6, sx)))
    return max(0, sy - 6) + ys[j], max(0, sx - 6) + xs[j]


class DotDistance:
    """Per-game cache: recomputes the BFS field only when the dot set changes (an item eaten / new board / restore)."""

    def __init__(self):
        self.key = None; self.field = None; self.walk = None

    def __call__(self, ale, ram, key):
        if key != self.key:
            walk, dots, _ = masks(ale.getScreenRGB())
            self.walk, self.field, self.key = walk, dist_field(walk, dots), key
        r, c = player_px(int(ram[10]), int(ram[16]), self.walk)
        v = int(self.field[r, c])
        return None if v >= INF else float(v)
