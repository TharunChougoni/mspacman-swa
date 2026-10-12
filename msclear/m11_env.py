"""M11 multi-board PPO envs. PixelGame workers; games play THROUGH board clears (episode = game over or max_steps
decisions from the start). Start mix: p_b1 fresh reset, p_b2 board-2 snapshot, rest board-3+ snapshot (base files or
harvested pool). Maze-agnostic item counting: ram[119] (+1 per dot/pellet, resets to 0 on a new board; 154 per board).

Reward per decision, rw = (item, score, clear, death, k_end):
  rw_item * sum over eaten items (1 + k_end * (1 - rem/30) if rem < 30)  + rw_score * score delta
  + rw_clear * [board index increased] - rw_death * [life lost or game over]  - stall penalty
Stall (items, n, pen, cap): with <= items left on the CURRENT board (154 - ram[119]) and no item for > n decisions:
  - min(cap, pen + pen * (since - n) / n).
Harvest: on boards >= 3 (any start), a snapshot at the board start and at the first step with <= 120/90/60/30/15 left;
kept in a per-worker FIFO (cap pool_cap) and mirrored to disk (harvest_dir/w{wid}_{slot}.pkl, slot rotates = FIFO).
step() also returns per env the board of the NEW observation (for the maze-1 KL mask).
"""
from __future__ import annotations

import collections
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np

from msclear.pixenv import PixelGame as _PG  # original class (snap_worker patches pixenv.PixelGame)

SEED_BASE = 4_000_000
H_THRESH = (120, 90, 60, 30, 15)


def _book(g):
    return dict(score=g.score, det=dict(vars(g.det)), prev_lives=g.prev_lives, h=g.h, deaths=g.deaths,
                steps=g.steps, max_eaten=g.max_eaten)


def restore(g, s):
    b = s["book"]
    g.ale.restoreState(s["state"])
    g.score, g.prev_lives, g.h, g.deaths, g.steps, g.max_eaten = (b["score"], b["prev_lives"], b["h"], b["deaths"],
                                                                 b["steps"], b["max_eaten"])
    g.det.__dict__.update(b["det"]); g.ram = g.ale.getRAM().copy(); g.stack = s["stack"].copy()
    g.cleared = False; g.done = False


def _worker(remote, n, wid, cfg):
    from msclear import ram as R
    from msclear.pixenv import PixelGame
    rng = np.random.default_rng(cfg["seed_off"] * 7919 + wid)
    b2 = [pickle.load(open(f, "rb")) for f in cfg["b2_files"]]
    b3 = [pickle.load(open(f, "rb")) for f in cfg["b3_files"]]
    pool = collections.deque(maxlen=cfg["pool_cap"]); slot = [0]
    hdir = cfg.get("harvest_dir")
    games = [PixelGame(sticky=0.25, max_steps=10**9) for _ in range(n)]
    meta = [None] * n; epn = [0]; rw = cfg["rw"]; stall = cfg["stall"]

    def snap(g):
        return dict(state=g.ale.cloneState(), stack=g.stack.copy(), book=_book(g), board=R.board(g.ram),
                    left=154 - int(g.ram[R.DOTS_EATEN]), lives=int(g.ram[R.LIVES]))

    def harvest(g):
        s = snap(g); pool.append(s)
        if hdir:
            p = os.path.join(hdir, f"w{wid:02d}_{slot[0]:03d}.pkl"); slot[0] = (slot[0] + 1) % cfg["pool_cap"]
            with open(p + ".tmp", "wb") as f:
                pickle.dump(s, f)
            os.replace(p + ".tmp", p)

    def new(i):
        g = games[i]
        seed = SEED_BASE + cfg["seed_off"] * 1000 + wid * 100_000 + epn[0]; epn[0] += 1
        u = rng.random(); start = "b1"
        if u < cfg["p_b1"] or not hasattr(g, "det"):
            g.reset(seed)                     # 280 intro frames; snapshot starts skip it (restoreState is enough)
        if u >= cfg["p_b1"]:
            if u < cfg["p_b1"] + cfg["p_b2"] and b2:
                restore(g, b2[int(rng.integers(len(b2)))]); start = "b2"
            elif pool and (not b3 or rng.random() < cfg["p_harvest"]):
                restore(g, pool[int(rng.integers(len(pool)))]); start = "b3h"
            elif b3:
                restore(g, b3[int(rng.integers(len(b3)))]); start = "b3"
        b = R.board(g.ram)
        meta[i] = dict(start=start, b0=b, cur=b, steps0=g.steps, score0=g.score, since=0,
                       thr=set(), e=int(g.ram[R.DOTS_EATEN]), items=0, deaths=0)
        return g.stack.copy()

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                st = np.stack([new(i) for i in range(n)])
                remote.send((st, np.array([meta[i]["cur"] for i in range(n)], np.int8)))
            elif cmd == "pool":
                remote.send(len(pool))
            elif cmd == "step":
                frames = np.empty((n, 84, 84), np.uint8); rew = np.zeros(n, np.float32); done = np.zeros(n, np.uint8)
                boards = np.zeros(n, np.int8); comp = np.zeros((n, 6), np.float32); results = []; starts = {}
                for i, (g, a) in enumerate(zip(games, data)):
                    m = meta[i]; sc0 = g.score
                    _, _, _, info = g.step(int(a))
                    r = g.ram; b = R.board(r); e = int(r[R.DOTS_EATEN])
                    clear = 0.0
                    if b != m["cur"]:
                        clear = float(b > m["cur"]); items = 154 - m["e"] if b > m["cur"] else 0
                        rem0 = 154 - m["e"]; m["cur"] = b; m["thr"] = set()
                        if b >= 3:
                            harvest(g)
                    else:
                        items = max(0, e - m["e"]); rem0 = 154 - m["e"]
                    m["e"] = e
                    bonus = sum(rw[4] * (1 - (rem0 - j) / 30) for j in range(items) if rem0 - j < 30)
                    over = bool(g.ale.game_over()); trunc = g.steps - m["steps0"] >= cfg["max_steps"]
                    death = float(info["lost_life"] or over)
                    left = 154 - e
                    m["since"] = 0 if items else m["since"] + 1
                    sp = 0.0
                    if stall and left <= stall[0] and m["since"] > stall[1]:
                        sp = min(stall[3], stall[2] + stall[2] * (m["since"] - stall[1]) / stall[1])
                    rew[i] = rw[0] * (items + bonus) + rw[1] * (g.score - sc0) + rw[2] * clear - rw[3] * death - sp
                    comp[i] = (items, g.score - sc0, clear, death, bonus, sp)
                    m["items"] += items; m["deaths"] += int(death)
                    if b >= 3 and not (over or trunc):
                        for th in H_THRESH:
                            if left <= th and th not in m["thr"]:
                                m["thr"].update(t for t in H_THRESH if t >= th); harvest(g); break
                    if over or trunc:
                        results.append(dict(start=m["start"], b0=m["b0"], board=b, clears=b - m["b0"], items=m["items"],
                                            score=g.score - m["score0"], steps=g.steps - m["steps0"], deaths=m["deaths"],
                                            truncated=bool(trunc and not over), left_end=left))
                        starts[i] = new(i); frames[i] = starts[i][-1]; done[i] = 1; boards[i] = meta[i]["cur"]
                    else:
                        frames[i] = g.frame; boards[i] = b
                remote.send((frames, rew, done, comp, boards, results, starts))
            elif cmd == "close":
                break
    except (KeyboardInterrupt, EOFError):
        pass


class VecGames11:
    """Workers in G groups (for env/GPU overlap): step_send(g, actions) / step_recv(g)."""

    def __init__(self, n_workers, per_worker, groups=2, **cfg):
        ctx = mp.get_context("spawn"); self.per = per_worker; self.nw = n_workers; self.n = n_workers * per_worker
        main = sys.modules["__main__"]
        saved = {k: main.__dict__.pop(k) for k in ("__file__", "__spec__") if k in main.__dict__}
        main.__spec__ = None
        self.remotes, self.procs = [], []
        try:
            for w in range(n_workers):
                a, b = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(b, per_worker, w, cfg), daemon=True)
                p.start(); b.close(); self.remotes.append(a); self.procs.append(p)
        finally:
            del main.__spec__; main.__dict__.update(saved)
        self.gw = [list(range(g, n_workers, groups)) for g in range(groups)]             # workers per group
        self.genv = [np.concatenate([np.arange(w * per_worker, (w + 1) * per_worker) for w in ws]) for ws in self.gw]

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        out = [r.recv() for r in self.remotes]
        return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])

    def step_send(self, g, actions):
        """actions for the envs of group g, in self.genv[g] order."""
        k = self.per
        for j, w in enumerate(self.gw[g]):
            self.remotes[w].send(("step", actions[j * k:(j + 1) * k]))

    def step_recv(self, g):
        k = self.per; out = [self.remotes[w].recv() for w in self.gw[g]]
        starts = {j * k + i: s for j, o in enumerate(out) for i, s in o[6].items()}       # index within the group
        cat = lambda q: np.concatenate([o[q] for o in out])  # noqa: E731
        return cat(0), cat(1), cat(2), cat(3), cat(4), [x for o in out for x in o[5]], starts

    def pool_size(self):
        for r in self.remotes:
            r.send(("pool", None))
        return sum(r.recv() for r in self.remotes)

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


class SnapGame:
    """fasteval-protocol game that starts from snapshot file M11_SNAPS[seed % len] (eval c). Plays until game over
    or MB_CAP. result: start board, board reached, items on the start board, cleared_start_board, deaths."""

    def __init__(self, sticky=0.25, max_steps=None):
        self.g = _PG(sticky=sticky, max_steps=10**9); self.env = self.g.env
        self.files = os.environ["M11_SNAPS"].split(","); self.cap = int(os.environ.get("MB_CAP", "8000"))

    def reset(self, seed):
        from msclear import ram as R
        g = self.g; g.reset(seed); s = pickle.load(open(self.files[seed % len(self.files)], "rb")); restore(g, s)
        self.b0 = R.board(g.ram); self.e0 = int(g.ram[R.DOTS_EATEN]); self.items = 0; self.n = 0; self.d = 0
        self.score0 = g.score; self.done = False; self.cur = self.b0; self.file = os.path.basename(self.files[seed % len(self.files)])
        return g.stack

    @property
    def frame(self):
        return self.g.frame

    def step(self, a):
        from msclear import ram as R
        g = self.g; _, _, _, info = g.step(a); self.n += 1; self.d += int(info["lost_life"])
        b = R.board(g.ram)
        if b == self.b0:
            self.items = max(self.items, int(g.ram[R.DOTS_EATEN]) - self.e0)
        elif self.cur == self.b0:
            self.items = 154 - self.e0
        self.cur = max(self.cur, b)
        over = g.ale.game_over()
        self.d += int(over and not info["lost_life"])
        self.done = over or self.n >= self.cap

    def result(self, seed):
        return dict(seed=seed, file=self.file, b0=self.b0, board=self.cur, cleared=self.cur > self.b0, items=self.items,
                    left0=154 - self.e0, steps=self.n, deaths=self.d, score=self.g.score - self.score0, eaten=self.items,
                    cap_hit=bool(self.n >= self.cap and not self.g.ale.game_over()))


def snap_worker(remote):
    from msclear import fasteval, pixenv
    pixenv.PixelGame = SnapGame
    fasteval._worker(remote)
