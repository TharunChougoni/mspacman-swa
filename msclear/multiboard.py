"""M10 multi-board eval (eval only): fasteval workers whose games do NOT end at the board-1 clear.

Board index = R.board(ram) from the level byte ram[101] (+22 per board, verified on boards 1-3 by m0_clear).
ClearDetector uses the board-1 dot mask, so it is NOT used past board 1; a clear = level byte increase.
Per board: items = max ram[119] seen on that board (resets to 0 each board), deaths, steps, score.
At the first decision of every board >= 2 a snapshot (ALE cloneState, 4-stack, PixelGame bookkeeping
= endgame_env format) is pickled to $MB_STATES/seed{seed}_board{b}.pkl.
"""
from __future__ import annotations

import os
import pickle

from msclear import ram as R
from msclear.pixenv import PixelGame


class MultiGame(PixelGame):
    def __init__(self, sticky=0.25, max_steps=None):
        super().__init__(sticky=sticky, max_steps=int(os.environ.get("MB_CAP", "20000")))
        self.states_dir = os.environ.get("MB_STATES")

    def reset(self, seed):
        out = super().reset(seed)
        self.seed = int(seed)
        self.cur = R.board(self.ram)
        self._final_counted = False
        self.boards = {self.cur: dict(items=0, deaths=0, start_step=0, start_score=0.0, lives=int(self.ram[R.LIVES]) & 7)}
        return out

    def step(self, k):
        stack, rew, _, info = super().step(k)
        r = self.ram
        b = R.board(r)
        if b != self.cur:
            self.cur = b
            self.boards[b] = dict(items=0, deaths=0, start_step=self.steps, start_score=float(self.score),
                                  lives=int(r[R.LIVES]) & 7)                 # spare lives on arrival
            if self.states_dir and b >= 2:
                from msclear.endgame_env import _book
                p = os.path.join(self.states_dir, f"seed{self.seed}_board{b}.pkl")
                with open(p, "wb") as f:
                    pickle.dump(dict(state=self.ale.cloneState(), stack=self.stack.copy(), book=_book(self),
                                     seed=self.seed, board=b, level_byte=int(r[R.LEVEL]), lives=int(r[R.LIVES])), f)
        bb = self.boards[self.cur]
        bb["items"] = max(bb["items"], int(r[R.DOTS_EATEN]))
        bb["deaths"] += int(info["lost_life"])
        self.cap_hit = self.steps >= self.max_steps and not self.ale.game_over()
        self.done = self.ale.game_over() or self.steps >= self.max_steps
        if self.ale.game_over() and not getattr(self, "_final_counted", False):
            self._final_counted = True            # the last life ends the game without decrementing ram[123]
            if not info["lost_life"]:
                bb["deaths"] += 1
                self.deaths += 1
        return stack, rew, self.done, info

    def result(self, seed):
        d = super().result(seed)
        cur = self.cur
        d.update(boards={str(k): v for k, v in self.boards.items()}, max_board=cur,
                 clears=cur - 1, cap_hit=bool(getattr(self, "cap_hit", False)),
                 total_items=sum(v["items"] for v in self.boards.values()), level_byte=int(self.ram[R.LEVEL]))
        return d


def mb_worker(remote):
    from msclear import pixenv, fasteval
    pixenv.PixelGame = MultiGame
    fasteval._worker(remote)
