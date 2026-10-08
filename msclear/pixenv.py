"""M6 pixel pipeline: what the student sees.

ALE v5 Ms. Pac-Man is created with frameskip=1; the 4-frame skip happens here, once
(no ALE-side skip on top -> exactly 4 emulator frames per agent step). Per step:
the action is repeated for 4 ALE frames, the grayscale screens of frames 3 and 4 are
max-pooled (sprite flicker), resized to 84x84 with INTER_AREA and pushed onto a 4-frame
stack. This is the gymnasium AtariPreprocessing(frame_skip=4, noop_max=0) layout, written
against the raw ALE so the HRA teacher can read RAM on exactly the frames the student sees.

Sticky actions (repeat_action_probability) are applied by ALE per emulator frame.
Reset plays the same 280 NOOP intro frames as the HRA runner, so step 0 is the same
decision for teacher and student. Student actions 0..3 = UP, RIGHT, LEFT, DOWN = ALE
minimal-set indices 1..4 (HRAAgent.act returns these indices).
"""
from __future__ import annotations

import cv2
import numpy as np

from msclear import maze
from msclear import ram as R
from msclear.clear import ClearDetector
from msclear.env import INTRO_FRAMES, make_env

SKIP = 4
STACK = 4
SIZE = 84
ACTIONS = ("UP", "RIGHT", "LEFT", "DOWN")


def preprocess(g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
    """Max of the last two grayscale 210x160 screens -> 84x84 uint8."""
    return cv2.resize(np.maximum(g0, g1), (SIZE, SIZE), interpolation=cv2.INTER_AREA)


class PixelGame:
    """One Ms. Pac-Man game: pixel obs for the student, RAM bookkeeping for stats and the teacher."""

    def __init__(self, sticky: float = 0.25, max_steps: int = 6000):
        self.env = make_env(frameskip=1, sticky=sticky)
        self.ale = self.env.unwrapped.ale
        self.aset = self.env.unwrapped._action_set
        self.max_steps = max_steps
        self._g = np.zeros((2, 210, 160), np.uint8)
        self.on_frame = None             # optional callback after every emulator frame (video)

    def _act(self, ale_action, n: int) -> float:
        r = 0.0
        for i in range(n):
            r += self.ale.act(ale_action)
            if self.on_frame is not None:
                self.score_live = getattr(self, "score", 0.0) + r
                self.on_frame()
            if i >= n - 2:
                self.ale.getScreenGrayscale(self._g[i - (n - 2)])
        return r

    def reset(self, seed: int):
        self.env.reset(seed=seed)
        self.score = self._act(self.aset[0], INTRO_FRAMES)
        f = preprocess(self._g[0], self._g[1])
        self.stack = np.repeat(f[None], STACK, 0)
        self.ram = self.ale.getRAM().copy()
        self.det = ClearDetector()
        self.det.reset(self.ram)
        self.prev_lives = R.lives(self.ram)
        self.h = 2                       # heading tracker, same rule as hra.run_episode
        self.deaths = 0
        self.steps = 0
        self.max_eaten = 0
        self.cleared = False
        self.done = False
        return self.stack

    @property
    def frame(self) -> np.ndarray:
        return self.stack[-1]

    def items_left(self) -> int:
        return R.N_ITEMS - int(self.ram[R.DOTS_EATEN]) if R.board(self.ram) == 1 else 0

    def step(self, k: int):
        """Student action k in 0..3. Returns (stack, reward, done, info)."""
        r0 = self.ram
        p0 = (int(r0[R.PLAYER_X]), int(r0[R.PLAYER_Y]))
        rew = self._act(self.aset[k + 1], SKIP)
        self.score += rew
        self.stack = np.concatenate([self.stack[1:], preprocess(self._g[0], self._g[1])[None]])
        r = self.ram = self.ale.getRAM().copy()
        p1 = (int(r[R.PLAYER_X]), int(r[R.PLAYER_Y]))
        self.h = 2 if maze.teleported([p0, p1]) else maze.heading(p0, p1, self.h)
        lost = int(r[R.LIVES]) < self.prev_lives
        self.deaths += int(lost)
        self.prev_lives = int(r[R.LIVES])
        if R.board(r) == 1:
            self.max_eaten = max(self.max_eaten, int(r[R.DOTS_EATEN]))
        self.steps += 1
        if self.det.update(r):
            self.cleared = True
            self.max_eaten = R.N_ITEMS
        self.done = self.cleared or self.ale.game_over() or self.steps >= self.max_steps
        return self.stack, rew, self.done, dict(lost_life=lost, moved=p0 != p1)

    def result(self, seed: int) -> dict:
        return dict(seed=seed, cleared=self.cleared, steps=self.steps, score=self.score,
                    eaten=self.max_eaten, deaths=self.deaths)
