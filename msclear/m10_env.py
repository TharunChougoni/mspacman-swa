"""M10 distillation envs: PixelGame workers whose games play THROUGH board clears (until game over or MAX_STEPS),
starting from a normal reset (seed SEED_BASE + ...) or, with prob p_snap, from a stored board-2 snapshot
(runs/multiboard/states/*_board2.pkl: ALE state + 4-stack + bookkeeping). Rewards are not needed (distillation).
step(actions) -> frames (N,84,84) of the next obs, done (N,), results (finished games), starts {i: start 4-stack}."""
from __future__ import annotations

import multiprocessing as mp
import pickle
import sys

import numpy as np

SEED_BASE = 3_000_000


def _worker(remote, n, sticky, max_steps, p_snap, snap_files, wid, seed_off):
    from msclear import ram as R
    from msclear.pixenv import PixelGame
    rng = np.random.default_rng(seed_off * 7919 + wid)
    snaps = [pickle.load(open(f, "rb")) for f in snap_files]
    games = [PixelGame(sticky=sticky, max_steps=max_steps) for _ in range(n)]
    meta = [None] * n; epn = [0]

    def new(i):
        g = games[i]
        seed = SEED_BASE + seed_off * 1000 + wid * 100_000 + epn[0]; epn[0] += 1
        g.reset(seed)
        if snaps and rng.random() < p_snap:
            s = snaps[int(rng.integers(len(snaps)))]; b = s["book"]
            g.ale.restoreState(s["state"])
            g.score, g.prev_lives, g.h, g.deaths, g.steps, g.max_eaten = (b["score"], b["prev_lives"], b["h"],
                                                                         b["deaths"], b["steps"], b["max_eaten"])
            g.det.__dict__.update(b["det"]); g.ram = g.ale.getRAM().copy(); g.stack = s["stack"].copy()
            g.cleared = False; g.done = False
            meta[i] = dict(start="board2", seed=s["seed"], b0=2)
        else:
            meta[i] = dict(start="normal", seed=seed, b0=1)
        meta[i]["steps0"] = g.steps
        return g.stack.copy()

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send(np.stack([new(i) for i in range(n)]))
            elif cmd == "step":
                frames = np.empty((n, 84, 84), np.uint8); done = np.zeros(n, np.uint8); results = []; starts = {}
                for i, (g, a) in enumerate(zip(games, data)):
                    g.step(int(a))
                    over = g.ale.game_over() or g.steps - meta[i]["steps0"] >= max_steps
                    if over:
                        m = meta[i]
                        results.append(dict(start=m["start"], seed=m["seed"], board=R.board(g.ram),
                                            clears=R.board(g.ram) - m["b0"], score=g.score, steps=g.steps - m["steps0"]))
                        starts[i] = new(i); frames[i] = starts[i][-1]; done[i] = 1
                    else:
                        frames[i] = g.frame
                remote.send((frames, done, results, starts))
            elif cmd == "close":
                break
    except (KeyboardInterrupt, EOFError):
        pass


class VecGames10:
    def __init__(self, n_workers, per_worker, sticky=0.25, max_steps=8000, p_snap=0.3, snap_files=(), seed_off=0):
        ctx = mp.get_context("spawn"); self.per = per_worker; self.n = n_workers * per_worker
        main = sys.modules["__main__"]
        saved = {k: main.__dict__.pop(k) for k in ("__file__", "__spec__") if k in main.__dict__}
        main.__spec__ = None
        self.remotes, self.procs = [], []
        try:
            for w in range(n_workers):
                a, b = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(b, per_worker, sticky, max_steps, p_snap, list(snap_files), w, seed_off),
                                daemon=True)
                p.start(); b.close(); self.remotes.append(a); self.procs.append(p)
        finally:
            del main.__spec__; main.__dict__.update(saved)

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return np.concatenate([r.recv() for r in self.remotes])

    def step(self, actions):
        k = self.per
        for j, r in enumerate(self.remotes):
            r.send(("step", actions[j * k:(j + 1) * k]))
        out = [r.recv() for r in self.remotes]
        starts = {j * k + i: s for j, o in enumerate(out) for i, s in o[3].items()}
        return (np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out]),
                [x for o in out for x in o[2]], starts)

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
