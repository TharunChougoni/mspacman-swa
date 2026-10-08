"""M1: maze graph from observed transitions + per-target GVF tables.

Decision step = 4 frames (frameskip 4, sticky 0), actions UP/RIGHT/LEFT/DOWN (env actions 1..4; NOOP is
excluded because the game keeps the last joystick direction, which would make NOOP depend on hidden state).
State s = (player RAM x, y, heading). Ghosts are pinned inside the pen by RAM poke during collection, so the
graph only reflects walls.

Targets:
  * dot i (150): pseudo-reward 1 when Ms. Pac-Man passes a position in dot i's eat zone during the step.
    Eat zone = {p : (p - dot_screen_pos) in EAT_OFFSETS}, offsets learned from observed bit flips.
  * power pellet k (4): pseudo-reward 1 when the step passes a position where pellet k was observed eaten
    (ram[117] bit 2+k cleared).
  * cell c = (x//4, y//4): pseudo-reward 1 when the step passes through cell c (used later for ghosts).
GVF  G_t(s,a) = E[r_t + (1 - r_t) * gamma * max_a' G_t(s',a')], solved on the empirical transition model by
synchronous sweeps (= tabular TD with alpha=1 on the model) until max change < tol.
"""
from __future__ import annotations

import collections
import multiprocessing as mp
import pickle
import time

import numpy as np

from msclear import ram as R
from msclear.env import INTRO_FRAMES, make_env

ACTIONS = (1, 2, 3, 4)                       # UP RIGHT LEFT DOWN (env action ids)
HEAD = {(0, -1): 0, (1, 0): 1, (-1, 0): 2, (0, 1): 3}   # heading index == action index


def teleported(seq) -> bool:
    """True if consecutive positions jump (death/board reset), ignoring the left-right tunnel wrap."""
    for p, q in zip(seq[:-1], seq[1:]):
        dx, dy = abs(p[0] - q[0]), abs(p[1] - q[1])
        if dy > 3 or (3 < dx < 100):
            return True
    return False


def heading(p0, p1, h):
    dx, dy = int(p1[0]) - int(p0[0]), int(p1[1]) - int(p0[1])
    if abs(dx) > 40:                          # tunnel wrap
        dx = -np.sign(dx)
    if dx == 0 and dy == 0:
        return h
    if abs(dx) >= abs(dy):
        return HEAD[(int(np.sign(dx)), 0)]
    return HEAD[(0, int(np.sign(dy)))]


class PinnedGame:
    """Frame-level ALE wrapper: ghosts held in the pen, lives topped up, dots refilled when low."""

    def __init__(self, seed=0, refill_below=10, sticky=0.0):
        self.env = make_env(frameskip=1, sticky=sticky)
        self.seed = seed
        self.refill_below = refill_below
        self.new_game()

    def new_game(self):
        self.env.reset(seed=self.seed); self.seed += 7919
        self.ale = self.env.unwrapped.ale
        self.aset = self.env.unwrapped._action_set
        for _ in range(INTRO_FRAMES):
            self.ale.act(self.aset[0])
        r = self.ale.getRAM()
        self.pen = (int(r[R.GHOST_X[1]]), int(r[R.GHOST_Y[1]]))
        self.full = r[R.PELLET_SLICE].copy()
        self.h = 2                           # starts moving left

    pin_on = True

    def pin(self):
        if not self.pin_on:
            return
        for gi in range(4):
            self.ale.setRAM(R.GHOST_X[gi], self.pen[0]); self.ale.setRAM(R.GHOST_Y[gi], self.pen[1])

    def pos(self):
        r = self.ale.getRAM(); return int(r[R.PLAYER_X]), int(r[R.PLAYER_Y])

    def step(self, a, frames=4):
        """Returns (visited positions, flipped bits list per frame, reward)."""
        vis, flips, rew = [], [], 0
        r0 = self.ale.getRAM(); b0 = R.bits(r0); pw0 = int(r0[R.POWER])
        p0 = self.pos()
        for _ in range(frames):
            rew += self.ale.act(self.aset[a]); self.pin()
            r = self.ale.getRAM(); b1 = R.bits(r)
            p = (int(r[R.PLAYER_X]), int(r[R.PLAYER_Y])); vis.append(p)
            fl = np.flatnonzero(b0 & ~b1)
            for i in fl:
                flips.append((int(i), p))
            pw1 = int(r[R.POWER])
            for k in range(4):
                if (pw0 >> (2 + k)) & 1 and not (pw1 >> (2 + k)) & 1:
                    flips.append((1000 + k, p))
            b0, pw0 = b1, pw1
        self.h = heading(p0, vis[-1], self.h)
        return vis, flips, rew

    def upkeep(self):
        r = self.ale.getRAM()
        if r[R.LIVES] < 2:
            self.ale.setRAM(R.LIVES, 3)
        if self.refill_below is not None and R.dots_left(r) < self.refill_below:
            for k, v in enumerate(self.full):
                self.ale.setRAM(R.PELLET_SLICE.start + k, int(v))
            self.ale.setRAM(R.POWER, int(r[R.POWER]) | 0x3C)


def _collect(seed, n_dec, q, p_switch=0.2, counter=None):
    g = PinnedGame(seed=seed); rng = np.random.default_rng(seed)
    trans = collections.Counter(); flips = collections.Counter(); a = 3
    stats = dict(teleports=0, lives=[])
    s = (*g.pos(), g.h)
    for it in range(n_dec):
        if counter is not None and it % 5000 == 4999:
            with counter.get_lock():
                counter.value += 5000
        if rng.random() < p_switch:
            a = int(rng.integers(1, 5))
        vis, fl, _ = g.step(a)
        s2 = (*vis[-1], g.h)
        if teleported([s[:2]] + vis):
            stats["teleports"] += 1; stats["lives"].append(int(g.ale.getRAM()[R.LIVES]))
        else:
            trans[(s, a, s2, tuple(vis))] += 1
        for i, p in fl:
            flips[(i, p)] += 1
        g.upkeep(); s = s2
        if g.ale.game_over():                 # 108k-frame episode cap
            g.new_game(); s = (*g.pos(), g.h)
    q.put((trans, flips, stats))


def collect(workers=12, n_dec=200_000, seed0=0, progress=None, every=30.0):
    """progress: callable(str) for a status line every `every` seconds (decisions done, rate, ETA)."""
    import queue as _queue
    q = mp.Queue(); counter = mp.Value("q", 0); t0 = time.time()
    ps = [mp.Process(target=_collect, args=(seed0 + w, n_dec, q, 0.2, counter)) for w in range(workers)]
    [p.start() for p in ps]
    trans, flips = collections.Counter(), collections.Counter(); tel = 0; got = 0; last = t0
    total = workers * n_dec
    while got < len(ps):
        try:
            t, f, st = q.get(timeout=2.0)
        except _queue.Empty:
            t = None
        if t is not None:
            trans.update(t); flips.update(f); tel += st["teleports"]; got += 1
        if progress and time.time() - last >= every:
            last = time.time(); done = counter.value; el = last - t0; rate = done / max(el, 1e-9)
            progress(f"  collect: {done:,}/{total:,} decisions ({100 * done / total:.0f}%), {rate:,.0f}/s, "
                     f"elapsed {el / 60:.1f} min, ETA {(total - done) / max(rate, 1e-9) / 60:.1f} min")
    collect.teleports = tel
    [p.join() for p in ps]
    return trans, flips


def eat_offsets(flips):
    m = R.dot_map(); sx = {int(b): int(np.floor(x)) for b, x in zip(m["bit"], m["sx"])}
    sy = {int(b): int(np.floor(y)) for b, y in zip(m["bit"], m["sy"])}
    offs = collections.Counter()
    for (i, p), c in flips.items():
        if i in sx:
            offs[(p[0] - sx[i], p[1] - sy[i])] += c
    return offs


class Model:
    """Empirical transition model + GVF tables."""

    def __init__(self, trans, flips, gamma=0.99):
        self.gamma = gamma
        states = sorted({k[0] for k in trans} | {k[2] for k in trans})
        self.states = states
        self.sid = {s: i for i, s in enumerate(states)}
        self.nS, self.nA = len(states), 4
        pos = sorted({(s[0], s[1]) for s in states} | {p for k in trans for p in k[3]})
        self.positions = pos
        self.offsets = eat_offsets(flips)
        m = R.dot_map()
        self.dot_bits = [int(b) for b in m["bit"]]
        offs = set(self.offsets)
        posset = set(pos)
        self.zones = []
        for b, x, y in zip(m["bit"], m["sx"], m["sy"]):
            fx, fy = int(np.floor(x)), int(np.floor(y))
            self.zones.append({(fx + dx, fy + dy) for dx, dy in offs} & posset)
        for k in range(4):                    # power pellets: observed eat positions
            self.zones.append({p for (i, p) in flips if i == 1000 + k})
        self.n_dots = len(self.dot_bits)
        self.cells = sorted({(p[0] // 4, p[1] // 4) for p in pos})
        self.cid = {c: i for i, c in enumerate(self.cells)}
        self.nT = len(self.zones) + len(self.cells)
        # outcomes
        sa, nxt, cnt, rew = [], [], [], []
        zone_of = collections.defaultdict(list)
        for t, z in enumerate(self.zones):
            for p in z:
                zone_of[p].append(t)
        nd = len(self.zones)
        for (s, a, s2, vis), c in trans.items():
            sa.append(self.sid[s] * 4 + (a - 1)); nxt.append(self.sid[s2]); cnt.append(c)
            hit = set()
            for p in vis:
                hit.update(zone_of.get(p, ()))
                hit.add(nd + self.cid[(p[0] // 4, p[1] // 4)])
            rew.append(sorted(hit))
        order = np.argsort(sa, kind="stable")
        self.o_sa = np.asarray(sa)[order]; self.o_next = np.asarray(nxt)[order]
        c = np.asarray(cnt, np.float64)[order]
        tot = np.bincount(self.o_sa, weights=c, minlength=self.nS * 4)
        self.o_p = (c / tot[self.o_sa]).astype(np.float32)
        self.sa_count = tot.reshape(self.nS, 4)
        self.o_rew = [rew[i] for i in order]
        self.starts = np.flatnonzero(np.r_[True, self.o_sa[1:] != self.o_sa[:-1]])
        self.sa_seen = self.o_sa[self.starts]

    def solve(self, targets=None, tol=1e-3, max_iter=2000, log=None):
        t_start = time.time()
        """Value iteration for the given target indices (default: all). Returns G [nS, 4, nT'] and history."""
        T = np.arange(self.nT) if targets is None else np.asarray(targets)
        col = {t: j for j, t in enumerate(T)}
        R_ = np.zeros((len(self.o_sa), len(T)), np.float32)
        for k, hits in enumerate(self.o_rew):
            for t in hits:
                j = col.get(t)
                if j is not None:
                    R_[k, j] = 1.0
        notR = (1.0 - R_) * self.gamma
        G = np.zeros((self.nS * 4, len(T)), np.float32)
        hist = []
        for it in range(max_iter):
            V = G.reshape(self.nS, 4, -1).max(1)
            tgt = (R_ + notR * V[self.o_next]) * self.o_p[:, None]
            Gn = np.zeros_like(G)
            Gn[self.sa_seen] = np.add.reduceat(tgt, self.starts, axis=0)
            delta = float(np.abs(Gn - G).max()); G = Gn; hist.append(delta)
            if log and it % 25 == 0:
                log(f"  sweep {it}: max change {delta:.2e} (target 1e-3), elapsed {time.time() - t_start:.0f}s")
            if delta < tol:
                break
        return G.reshape(self.nS, 4, -1), hist

    def nearest_state(self, x, y, h):
        s = (x, y, h)
        if s in self.sid:
            return self.sid[s]
        for hh in range(4):
            if (x, y, hh) in self.sid:
                return self.sid[(x, y, hh)]
        best = min(self.states, key=lambda q: abs(q[0] - x) + abs(q[1] - y) + (q[2] != h))
        return self.sid[best]

    def save(self, path, **extra):
        with open(path, "wb") as f:
            pickle.dump(dict(model=self, **extra), f)
