"""M8 endgame PPO environments: M7's PixelGame workers + a per-worker pool of late-board start states.

Obs pipeline = M6/M7 exactly (PixelGame: 4 emulator frames per decision, sticky 0.25 per frame, max-pool,
84x84). With prob P_POOL a new episode starts from a random pool snapshot instead of a normal reset.

Snapshot = ALE cloneState() (emulator state incl. the sticky "last action"; the ALE RNG is NOT included,
so sticky-action randomness differs on every restore -> no replayed outcomes) + the 4-stack the agent
saw at that step + PixelGame bookkeeping (score, clear detector, lives, heading, deaths, steps, items).
On restore the 4-stack is the stored one (real motion), and the main process restarts the SWA token
window at the restored step (same clamping as at a normal reset).

Pool (FIFO, POOL_CAP per worker, ~36 KB/snapshot -> 14 x 200 x 36 KB ~ 100 MB total), filled from
current-policy play, so it refreshes continuously:
  * threshold snapshots: first step a NORMAL-start game has <= 30 / 20 / 12 / 6 items left on board 1
  * pre-death snapshots: a ring of every 5th step (13 deep) per game; when a life is lost with
    <= 40 items left, the ring entry 40-60 steps before the death is added (normal or pool starts)

Stall penalty (M9; default stall=None = M8 behaviour): stall=(items, n, pen, cap). On board 1 with <= `items` left and
no item eaten for more than `n` decisions: reward -= min(cap, pen + pen * (since - n) / n) per decision
(M8 failure mode: 23% of games ran into the 6000-step cap with a few pellets left).

Reward per decision (weights rw = item, score, clear, death, k_end):
  item term  rw_item * sum over eaten items of (1 + k_end * (1 - rem/30)) for rem < 30 (else 1)
  + rw_score * score delta + rw_clear * clear - rw_death * death   (death incl. the final, non-decrementing one)
"""
from __future__ import annotations

import collections
import multiprocessing as mp
import sys

import numpy as np

SEED_BASE = 2_000_000          # M8 training seeds (M7 used 1,000,000+; evals 0..99 and 50000+)
THRESH = (30, 20, 12, 6)


def _book(g):
    return dict(score=g.score, det=dict(vars(g.det)), prev_lives=g.prev_lives, h=g.h, deaths=g.deaths,
                steps=g.steps, max_eaten=g.max_eaten)


def _worker(remote, gids, sticky, max_steps, rw, seed_off, p_pool, pool_cap, wid, stall=None):
    from msclear.pixenv import PixelGame
    from msclear import ram as R
    rng = np.random.default_rng(seed_off * 7919 + wid)
    games = [PixelGame(sticky=sticky, max_steps=max_steps) for _ in gids]
    pool = collections.deque(maxlen=pool_cap)
    epn = [0] * len(gids)
    meta = [None] * len(gids)                     # per game: dict(start, seed, items0, thr_done, ring)

    def snap(g):
        return (g.ale.cloneState(), g.stack.copy(), _book(g))

    def new(i):
        g = games[i]
        if pool and rng.random() < p_pool:
            st, stack, b = pool[int(rng.integers(len(pool)))]
            g.ale.restoreState(st)
            g.score, g.prev_lives, g.h, g.deaths = b["score"], b["prev_lives"], b["h"], b["deaths"]
            g.steps, g.max_eaten = b["steps"], b["max_eaten"]
            g.det.__dict__.update(b["det"])
            g.ram = g.ale.getRAM().copy(); g.stack = stack.copy(); g.cleared = False; g.done = False
            meta[i] = dict(start="pool", seed=-1, items0=g.items_left(), thr=set(THRESH), ring=collections.deque(maxlen=13), since=0)
        else:
            seed = SEED_BASE + gids[i] * 100_000 + seed_off + epn[i]; epn[i] += 1
            g.reset(seed)
            meta[i] = dict(start="normal", seed=seed, items0=154, thr=set(), ring=collections.deque(maxlen=13), since=0)
        return g.stack.copy()

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send(np.stack([new(i) for i in range(len(games))]))
            elif cmd == "step":
                n = len(games)
                frames = np.empty((n, 84, 84), np.uint8)
                rew = np.zeros(n, np.float32); done = np.zeros(n, np.uint8)
                comp = np.zeros((n, 6 if stall else 5), np.float32)   # items, score, clear, death, endgame-bonus[, stall penalty]
                results, starts = [], {}
                for i, (g, a) in enumerate(zip(games, data)):
                    m = meta[i]
                    if g.steps % 5 == 0:
                        m["ring"].append((g.steps, snap(g)))
                    b1 = R.board(g.ram) == 1
                    rem0 = g.items_left() if b1 else 154
                    e0 = int(g.ram[R.DOTS_EATEN])
                    _, sc, d, info = g.step(int(a))
                    items = max(0, int(g.ram[R.DOTS_EATEN]) - e0) if b1 else 0
                    bonus = sum(rw[4] * (1 - (rem0 - j) / 30) for j in range(items) if rem0 - j < 30)
                    over = d and g.ale.game_over() and not g.cleared
                    death = float(info["lost_life"] or over)
                    clear = float(d and g.cleared)
                    left = g.items_left() if R.board(g.ram) == 1 else 154
                    m["since"] = 0 if items else m["since"] + 1
                    stall_pen = 0.0
                    if stall is not None and left <= stall[0] and m["since"] > stall[1]:
                        stall_pen = min(stall[3], stall[2] * (m["since"] - stall[1]) / stall[1] + stall[2])
                    comp[i] = (items, sc, clear, death, bonus, stall_pen)[:comp.shape[1]]
                    rew[i] = rw[0] * (items + bonus) + rw[1] * sc + rw[2] * clear - rw[3] * death - stall_pen
                    if not d:
                        if m["start"] == "normal" and R.board(g.ram) == 1:
                            for th in THRESH:
                                if left <= th and th not in m["thr"]:
                                    m["thr"].add(th); pool.append(snap(g)); break
                        if info["lost_life"] and left <= 40:
                            cand = [s for t, s in m["ring"] if 40 <= g.steps - t <= 60]
                            if cand:
                                pool.append(cand[0])
                    if d:
                        res = g.result(m["seed"]); res.update(start=m["start"], items0=m["items0"],
                                                              truncated=bool(not g.cleared and not g.ale.game_over()))
                        results.append(res)
                        starts[i] = new(i); frames[i] = starts[i][-1]; done[i] = 1
                    else:
                        frames[i] = g.frame
                remote.send((frames, rew, done, comp, results, starts, len(pool)))
            elif cmd == "close":
                break
    except (KeyboardInterrupt, EOFError):
        pass


class VecGames:
    def __init__(self, n_workers, per_worker, sticky=0.25, max_steps=6000, rw=(0.1, 0.0005, 10.0, 5.0, 2.0),
                 seed_off=0, p_pool=0.6, pool_cap=200, stall=None):
        ctx = mp.get_context("spawn")
        self.n = n_workers * per_worker; self.per = per_worker
        # spawn children re-import the parent's __main__ (the trainer -> torch, ~400 MB per worker).
        # The workers only need msclear, so hide __main__'s file while starting them (~5 GB RAM saved).
        main = sys.modules["__main__"]
        saved = {k: main.__dict__.pop(k) for k in ("__file__", "__spec__") if k in main.__dict__}
        main.__spec__ = None
        self.remotes, self.procs = [], []
        self.pool_size = 0
        for w in range(n_workers):
            a, b = ctx.Pipe()
            gids = list(range(w * per_worker, (w + 1) * per_worker))
            p = ctx.Process(target=_worker, args=(b, gids, sticky, max_steps, tuple(rw), seed_off, p_pool, pool_cap, w,
                                                                tuple(stall) if stall else None),
                            daemon=True)
            p.start(); b.close()
            self.remotes.append(a); self.procs.append(p)
        del main.__spec__
        main.__dict__.update(saved)

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return np.concatenate([r.recv() for r in self.remotes])          # (N, 4, 84, 84)

    def step(self, actions):
        k = self.per
        for j, r in enumerate(self.remotes):
            r.send(("step", actions[j * k:(j + 1) * k]))
        out = [r.recv() for r in self.remotes]
        starts = {j * k + i: s for j, o in enumerate(out) for i, s in o[5].items()}
        self.pool_size = sum(o[6] for o in out)
        return (np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out]),
                np.concatenate([o[2] for o in out]), np.concatenate([o[3] for o in out]),
                [x for o in out for x in o[4]], starts)

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
