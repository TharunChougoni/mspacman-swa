"""Search teacher: emulator-lookahead planner (ALE cloneState/restoreState) for any maze.

Per decision: for each of the 4 actions, R rollouts of H decisions (4 frames each): the action is held for COMMIT
decisions, then a cheap default policy follows the BFS maze-distance gradient to the nearest remaining dot
(msclear.mazedist; field recomputed when the item count changes) with eps-random turns.
Score = items eaten + CLEAR_B on clear - DEATH_P on life lost - DIST_W * pathdist(terminal); action value = mean
(ties broken by max). Items via ram[119] (DOTS_EATEN), clear = level byte ram[101] increase, death = lives drop / game over.
"""
from __future__ import annotations

import numpy as np

from msclear import mazedist as MD
from msclear import ram as R

SKIP = 4
DIRS = ((-1, 0), (0, 1), (0, -1), (1, 0))      # UP RIGHT LEFT DOWN (pixenv.ACTIONS order)


GX, GY = list(R.GHOST_X), list(R.GHOST_Y)
import os
GHOST_R = int(os.environ.get('GHOST_R', '16'))
RANDDEF = float(os.environ.get('RANDDEF', '0'))
CATCH = int(os.environ.get('CATCH', '6'))
HYST = float(os.environ.get('HYST', '3'))
MIXMAX = float(os.environ.get('MIXMAX', '0.5'))


def caught(ram, thr=4):
    """Collision with a non-frightened ghost (RAM coords). Lives/game-over only change ~20 decisions after the hit."""
    d = np.abs(ram[GX].astype(int) - int(ram[10])) + np.abs(ram[GY].astype(int) - int(ram[16]))
    return bool(((d <= thr) & ~R.frightened(ram)).any())


class DotFields:
    """Per-dot BFS fields, cached per board (walk grid from the first screen of the board). At each root the remaining
    dots are read from the screen; inside rollouts an eaten item removes the remaining dot nearest the player."""

    def __init__(self):
        self.level = None

    def root(self, ale, ram):
        walk, dots, _ = MD.masks(ale.getScreenRGB())
        if int(ram[R.LEVEL]) != self.level:
            self.level, self.walk, self.D = int(ram[R.LEVEL]), walk, {}
        n, lab, st, cen = __import__("cv2").connectedComponentsWithStats(dots.astype(np.uint8), connectivity=4)
        keys = []
        for j in range(1, n):
            ys, xs = np.nonzero(lab == j); y, x = int(ys.mean() + .5), int(xs.mean() + .5)
            k = (y, x)
            if k not in self.D:
                src = np.zeros_like(self.walk); src[ys, xs] = True
                f = MD.dist_field(self.walk, src); self.D[k] = np.minimum(f, 30000).astype(np.int16)
            keys.append(k)
        self.F = np.stack([self.D[k] for k in keys]) if keys else np.zeros((0,) + self.walk.shape, np.int16)

    def pos(self, ram):
        return MD.player_px(int(ram[10]), int(ram[16]), self.walk)


def nearest(F, alive, r, c):
    if not alive.any():
        return 0
    return int(F[alive, r, c].min())


def greedy_dir(df, alive, ram, rng, eps, avoid=-1):
    if rng.random() < eps:
        return int(rng.integers(4))
    r, c = df.pos(ram); walk = df.walk
    if avoid >= 0:
        bk = int((avoid + 1 + rng.integers(3)) % 4)
    px, py = int(ram[10]), int(ram[16])
    fr = R.frightened(ram)
    gp = np.stack([ram[GX].astype(int), ram[GY].astype(int)], 1)[~fr] if GHOST_R > 0 else None
    best = MD.INF + 1
    if avoid < 0:
        bk = int(rng.integers(4))
    for k, (dr, dc) in enumerate(DIRS):
        if k == avoid:
            continue
        v = MD.INF
        for s in (1, 2, 3):
            rr, cc = r + dr * s, (c + dc * s) % 160
            if 0 <= rr < MD.H and walk[rr, cc]:
                v = min(v, nearest(df.F, alive, rr, cc))
        if gp is not None:
            gd = int((np.abs(gp[:, 0] - (px + dc * 6)) + np.abs(gp[:, 1] - (py + dr * 6))).min()) if len(gp) else 99
            if gd < GHOST_R:
                v += (GHOST_R - gd) * 20
        if v < best or (v == best and rng.random() < .5):
            best, bk = v, k
    return bk


class Teacher:
    def __init__(self, H=40, R_=4, commit=4, eps=0.15, clear_b=500., death_p=300., dist_w=0.05, seed=0):
        self.H, self.R, self.commit, self.eps = H, R_, commit, eps
        self.clear_b, self.death_p, self.dist_w = clear_b, death_p, dist_w
        self.rng = np.random.default_rng(seed); self.df = DotFields()

    def _rollout(self, ale, aset, a0, ram0, fixed=()):
        lv0, li0, it0 = int(ram0[R.LEVEL]), int(ram0[R.LIVES]) & 7, int(ram0[R.DOTS_EATEN])
        it = it0; ram = ram0; df = self.df; alive = np.ones(len(df.F), bool); avoid = -1; k = a0
        self.seq = seq = []; self.pos = []
        for t in range(self.H):
            p0 = (int(ram[10]), int(ram[16]))
            if t < len(fixed):
                k = fixed[t]
            elif t >= self.commit and RANDDEF:
                if avoid >= 0 or self.rng.random() < RANDDEF:
                    k = int(self.rng.integers(4))
            elif t >= self.commit:
                k = greedy_dir(df, alive, ram, self.rng, self.eps, avoid)
            hit = False; seq.append(k)
            for _ in range(SKIP):
                ale.act(aset[k + 1])
                hit = hit or caught(ale.getRAM(), CATCH)
            ram = ale.getRAM(); self.pos.append((int(ram[10]), int(ram[16])))
            avoid = k if (int(ram[10]), int(ram[16])) == p0 else -1
            if int(ram[R.LEVEL]) > lv0:
                return (it - it0) + self.clear_b + (self.H - t)     # sooner clear is better
            if ale.game_over() or (int(ram[R.LIVES]) & 7) < li0 or hit:
                return (it - it0) - self.death_p
            e = int(ram[R.DOTS_EATEN])
            if e > it:
                r, c = df.pos(ram)
                for _ in range(e - it):
                    if alive.any():
                        ii = np.flatnonzero(alive); alive[ii[np.argmin(df.F[ii, r, c])]] = False
                it = e
        r, c = df.pos(ram)
        d = min(nearest(df.F, alive, r, c), 300)
        return (it - it0) - self.dist_w * d

    @staticmethod
    def _reset_sticky(ale, root, g, last):
        """restoreState does not restore ALE's sticky 'previous action'; rollouts left it at their last action, so the
        real step (and the next rollout) would repeat a wrong action w.p. 0.25 on its first frame. Re-seed it with the
        last real action (3 acts: P(wrong) ~ 1.5%), then restore again."""
        ale.restoreState(root)
        if last is not None:
            for _ in range(3):
                ale.act(g.aset[last + 1])
            ale.restoreState(root)

    def act(self, g, prev=None, shift=1):
        """Deterministic given the cloned ALE RNG (sticky draws included), so max over rollouts is achievable:
        the best rollout's action sequence is kept as a plan and re-evaluated (shifted) next decision."""
        ale = g.ale; root = ale.cloneState(include_rng=True); ram0 = ale.getRAM().copy(); self.df.root(ale, ram0)
        q = np.full((4, self.R + 1), np.nan); best = (-1e9, None)
        last = getattr(self, "last", None)
        plan = getattr(self, "plan", None)
        plan = plan[shift:] if plan is not None and len(plan) > shift else None
        for a in range(4):
            for j in range(self.R + (1 if plan is not None and plan[0] == a else 0)):
                self._reset_sticky(ale, root, g, last)
                v = self._rollout(ale, g.aset, a, ram0, fixed=plan if j == self.R else ())
                q[a, j] = v
                if v > best[0]:
                    best = (v, list(self.seq)); self.best_pos = list(self.pos)
        self.plan = best[1]
        self._reset_sticky(ale, root, g, last)
        q = np.where(np.isnan(q), np.nanmin(q), q)
        m = (1 - MIXMAX) * q.mean(1) + MIXMAX * q.max(1)
        if prev is not None:
            m[prev] += HYST
        return int(np.argmax(m)), q
