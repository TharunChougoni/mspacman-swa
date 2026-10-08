"""Board-clear detector from RAM.

A clear = the drawn-dot count (DOT_MASK bits only) reaches <= 3 and on a later step
jumps back to >= 100 (new board loaded) AND the level byte ram[101] has increased since arming. The old maze_watch.py counted all 234 bits,
whose floor is 92 (non-dot bits), so its thresholds (<=8, refill from <=80) could never fire.
"""
from __future__ import annotations

from msclear import ram as R


class ClearDetector:
    def __init__(self, low: int = 3, refill: int = 100):
        self.low, self.refill = low, refill
        self.reset()

    def reset(self, ram=None):
        self.min_dots = 10**9 if ram is None else R.dots_left(ram)
        self.clears = 0
        self.armed = False
        self.level_at_arm = -1

    def update(self, ram) -> bool:
        """Call once per step with current RAM. Returns True on the step a clear completes."""
        d = R.dots_left(ram)
        self.min_dots = min(self.min_dots, d)
        if d <= self.low and not self.armed:
            self.armed = True
            self.level_at_arm = int(ram[R.LEVEL])
        if self.armed and d >= self.refill and int(ram[R.LEVEL]) > self.level_at_arm:
            self.armed = False
            self.clears += 1
            self.min_dots = d
            return True
        return False
