"""Verified Ms. Pac-Man (ALE 0.12, ROM ms_pacman) RAM layout.

Verified on ale-py 0.12.0 (see scripts/m0_*.py and tests/):
  player x,y        ram[10], ram[16]   screen ≈ (x - 8.3, y + 5.8), r = 0.998
  ghost x           ram[6..9]          6 = Sue (orange), 7 = Inky, 8 = Pinky, 9 = Blinky
  ghost y           ram[12..15]        same order, same screen offset, r = 1.000
  lives             ram[123]
  dot bitmap        ram[59:101]        336 bits; 242 set at start but only 150 are dots
                                       (142 in ram[59:100] + 8 bottom-row dots in ram[100], drawn as dashes;
                                       the other 92 bits never clear) -> always use dot_map()["mask"]
  dots-eaten count  ram[119]           +1 per dot or power pellet; board ends at 154 = 150 + 4, resets to 0
  level / board     LEVEL = ram[101] (+22 per board)
  power pellets     POWER ram[117] bits 2..5
  frightened        GHOST_STATE ram[1..4] bit 7 (per ghost), FRIGHT_TIMER ram[116]
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

PLAYER_X, PLAYER_Y = 10, 16
GHOST_X = (6, 7, 8, 9)
GHOST_Y = (12, 13, 14, 15)
GHOST_NAMES = ("sue", "inky", "pinky", "blinky")
LIVES = 123
DOTS_EATEN = 119
PELLET_SLICE = slice(59, 101)
N_BITS = 336
N_ITEMS = 154         # 150 dots + 4 power pellets = ram[119] at board end
SCREEN_DX, SCREEN_DY = -8.3, 5.8
DATA = Path(__file__).resolve().parent / "data"
DOTS_JSON = DATA / "maze1_dots.json"
LEVEL = 101           # board counter: 20, 42, 64 on boards 1, 2, 3 (+22 per clear), verified by m0_clear.py
POWER = 117           # power pellets: bits 2..5 set while uneaten (bits 0-1 are something else)
POWER_BITS = {2: "bottom-right", 3: "top-right", 4: "bottom-left", 5: "top-left"}
GHOST_STATE = (1, 2, 3, 4)   # per ghost (same order as GHOST_X); bit 7 set = frightened (blue/flashing), 99.8% vs screen
FRIGHT_BIT = 0x80
FRIGHT_TIMER = 116    # global countdown after a power pellet, 0 when no ghost is frightened (98.2% as a flag; misses eaten ghosts)


def bits(ram) -> np.ndarray:
    return np.unpackbits(np.asarray(ram, dtype=np.uint8)[PELLET_SLICE])


@lru_cache(maxsize=1)
def dot_map() -> dict:
    d = json.loads(DOTS_JSON.read_text())
    d["bit"] = np.asarray(d["bit"], dtype=np.int64)
    d["sx"] = np.asarray(d["sx"], dtype=np.float32)
    d["sy"] = np.asarray(d["sy"], dtype=np.float32)
    mask = np.zeros(N_BITS, dtype=bool)
    mask[d["bit"]] = True
    d["mask"] = mask
    return d


def dots_left(ram) -> int:
    return int(bits(ram)[dot_map()["mask"]].sum())


def remaining_dots(ram) -> np.ndarray:
    """Indices into dot_map() arrays of dots still on the board."""
    b = bits(ram)
    return np.nonzero(b[dot_map()["bit"]])[0]


def player_xy(ram) -> tuple[int, int]:
    return int(ram[PLAYER_X]), int(ram[PLAYER_Y])


def ghosts_xy(ram) -> np.ndarray:
    ram = np.asarray(ram)
    return np.stack([ram[list(GHOST_X)], ram[list(GHOST_Y)]], axis=1).astype(np.int64)


def to_screen(x, y) -> tuple[float, float]:
    return float(x) + SCREEN_DX, float(y) + SCREEN_DY


def lives(ram) -> int:
    return int(ram[LIVES])


def power_left(ram) -> np.ndarray:
    """Bool per power pellet, order = sorted(POWER_BITS): bits 2, 3, 4, 5."""
    return np.array([(int(ram[POWER]) >> k) & 1 for k in sorted(POWER_BITS)], dtype=bool)


def frightened(ram) -> np.ndarray:
    """Bool per ghost (sue, inky, pinky, blinky): currently frightened."""
    return np.array([(int(ram[j]) & FRIGHT_BIT) != 0 for j in GHOST_STATE])


def board(ram) -> int:
    """1-based board index from the level byte."""
    return (int(ram[LEVEL]) - 20) // 22 + 1
