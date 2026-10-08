"""RAM pellet bits 59–99 → maze-1 pixel/grid cells.

Only bits observed in maze1_bitmap_c3.npz / maze1_bitmap_all.npz are used.
Guessed assignments in maze1_bitmap.npz (all 328 bits) are ignored.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cleanrl_utils.maze_watch import PELLET_SLICE
from cleanrl_utils.pellet_bfs import CELL

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBS = (
    ROOT / "data" / "maze1_bitmap_c3.npz",
    ROOT / "data" / "maze1_bitmap_all.npz",
)


def load_observed_bits(paths: tuple[Path, ...] | None = None) -> dict[int, tuple[float, float]]:
    """bit index 0..327 → mean (px, py) over observations. Later files override."""
    out: dict[int, list[tuple[float, float]]] = {}
    for p in paths or DEFAULT_OBS:
        if not p.is_file():
            continue
        z = np.load(p)
        for b, x, y in zip(z["bit"], z["px"], z["py"]):
            out.setdefault(int(b), []).append((float(x), float(y)))
    return {b: (float(np.mean([t[0] for t in pts])), float(np.mean([t[1] for t in pts]))) for b, pts in out.items()}


def unpack_pellet_bits(ram: np.ndarray) -> np.ndarray:
    ram = np.asarray(ram, dtype=np.uint8)
    return np.unpackbits(ram[PELLET_SLICE])


def ram_goal_cells(
    ram: np.ndarray,
    observed: dict[int, tuple[float, float]] | None = None,
    cell: int = CELL,
) -> set[tuple[int, int]]:
    """Currently-set observed bits as (gx, gy) cells."""
    if observed is None:
        observed = load_observed_bits()
    bits = unpack_pellet_bits(ram)
    goals: set[tuple[int, int]] = set()
    n = int(bits.size)
    for b, (px, py) in observed.items():
        if 0 <= b < n and bits[b]:
            goals.add((int(px) // cell, int(py) // cell))
    return goals
