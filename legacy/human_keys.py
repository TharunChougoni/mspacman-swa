"""Keyboard → ALE Ms. Pac-Man action. Pure, no pygame import required for tests."""

from __future__ import annotations

# ALE/MsPacman-v5 default meanings (env still wins at runtime).
DEFAULT_MEANINGS = (
    "NOOP",
    "UP",
    "RIGHT",
    "LEFT",
    "DOWN",
    "UPRIGHT",
    "UPLEFT",
    "DOWNRIGHT",
    "DOWNLEFT",
)


def action_from_wasd(up: bool, down: bool, left: bool, right: bool, meanings=DEFAULT_MEANINGS) -> int:
    if up and down:
        up = down = False
    if left and right:
        left = right = False
    if up and right:
        name = "UPRIGHT"
    elif up and left:
        name = "UPLEFT"
    elif down and right:
        name = "DOWNRIGHT"
    elif down and left:
        name = "DOWNLEFT"
    elif up:
        name = "UP"
    elif down:
        name = "DOWN"
    elif left:
        name = "LEFT"
    elif right:
        name = "RIGHT"
    else:
        name = "NOOP"
    meanings = tuple(meanings)
    if name in meanings:
        return int(meanings.index(name))
    if name != "NOOP" and "NOOP" in meanings:
        return int(meanings.index("NOOP"))
    return 0


class Stick:
    """Held WASD from KEYDOWN/KEYUP. Wayland get_pressed() is unreliable."""

    __slots__ = ("up", "down", "left", "right")

    def __init__(self) -> None:
        self.up = False
        self.down = False
        self.left = False
        self.right = False

    def set_key(self, name: str, down: bool) -> None:
        n = name.lower()
        if n in ("w", "up"):
            self.up = down
        elif n in ("s", "down"):
            self.down = down
        elif n in ("a", "left"):
            self.left = down
        elif n in ("d", "right"):
            self.right = down

    def wasd(self) -> tuple[bool, bool, bool, bool]:
        return self.up, self.down, self.left, self.right


def keep_human_life(*, maze_changed: bool, leftover_min: int, keep_leftover: int) -> bool:
    """Keep a life if the maze flipped or leftover hit the near-miss bar."""
    if maze_changed:
        return True
    return int(leftover_min) <= int(keep_leftover)
