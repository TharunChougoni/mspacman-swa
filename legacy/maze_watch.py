"""Detect a Ms. Pac-Man board clear from ALE RAM pellet bits.

Bytes 59..99 hold a bit-packed pellet map (full board ~234 bits).
A clear is: pellets drop low, then jump back up (next maze loaded).

eaten is THIS life only (start - min). 234-min is wrong after a death:
the maze does not refill, so a 2-step leftover life looks like 129 eaten.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym


PELLET_SLICE = slice(59, 100)
REFILL_FROM = 80
REFILL_BY = 80
EMPTY_AT = 8


def pellet_count(ale) -> int:
    ram = np.asarray(ale.getRAM(), dtype=np.uint8)
    return int(np.unpackbits(ram[PELLET_SLICE]).sum())


class MazeWatch(gym.Wrapper):
    """maze_changed / pellets / pellets_min / pellets_eaten on every info."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.min_p = 10**9
        self.start_p = 0
        self.changed = False
        self.lives = None
        self.need_rebase = False

    def _p(self) -> int:
        return pellet_count(self.env.unwrapped.ale)

    def _stamp(self, info: dict) -> dict:
        eaten = max(0, int(self.start_p) - int(self.min_p))
        info["maze_changed"] = bool(self.changed)
        info["pellets"] = int(self._p()) if self.min_p < 10**9 else int(self.start_p)
        info["pellets_min"] = int(self.min_p) if self.min_p < 10**9 else int(self.start_p)
        info["pellets_start"] = int(self.start_p)
        info["pellets_eaten"] = eaten
        return info

    def _rebase(self) -> None:
        p = self._p()
        self.start_p = p
        self.min_p = p
        self.changed = False
        self.need_rebase = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._rebase()
        try:
            self.lives = int(self.env.unwrapped.ale.lives())
        except Exception:
            self.lives = None
        return obs, self._stamp(info)

    def step(self, action):
        if self.need_rebase:
            self._rebase()
        obs, reward, terminated, truncated, info = self.env.step(action)
        p = self._p()
        if p < self.min_p:
            self.min_p = p
        if self.min_p <= EMPTY_AT:
            self.changed = True
        if self.min_p <= REFILL_FROM and p >= self.min_p + REFILL_BY:
            self.changed = True
        try:
            lives = int(self.env.unwrapped.ale.lives())
        except Exception:
            lives = self.lives
        if self.lives is not None and lives is not None and lives < self.lives:
            self.need_rebase = True
        self.lives = lives
        return obs, reward, terminated, truncated, self._stamp(info)
