"""M23 colour observations + a batched rollout engine shared by collection, DAgger and eval.

Obs: max of the last two RGB screens of each 4-frame decision, playfield crop rows [0:172) (score bar dropped),
INTER_AREA to 128x128 RGB (HWC uint8). The 84x84 gray 4-stack of PixelGame is kept alongside for the M22 teacher.
A game restored from a snapshot gets its colour frame from the current screen (no previous frame to max with).

Game = msclear.m13_env.SnapGame semantics (the multiseed protocol): start from a snapshot file (or a fresh reset),
play until the start board is cleared, the game is over or `cap` decisions; deaths = lives lost (+1 if game over
without a life-loss event). clean = cleared and deaths == 0.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import sys

import numpy as np

CROP, RES = 172, 128


def colour_pp(c0, c1):
    import cv2
    return cv2.resize(np.maximum(c0, c1)[:CROP], (RES, RES), interpolation=cv2.INTER_AREA)


def _make_game_cls():
    from msclear.pixenv import PixelGame

    class ColourGame(PixelGame):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._c = np.zeros((2, 210, 160, 3), np.uint8)

        def _act(self, ale_action, n):
            r = 0.0
            for i in range(n):
                r += self.ale.act(ale_action)
                if i >= n - 2:
                    self.ale.getScreenGrayscale(self._g[i - (n - 2)])
                    self.ale.getScreenRGB(self._c[i - (n - 2)])
            return r

        def reset(self, seed):
            s = super().reset(seed); self.colour = colour_pp(self._c[0], self._c[1]); return s

        def step(self, k):
            out = super().step(k); self.colour = colour_pp(self._c[0], self._c[1]); return out

    return ColourGame


class Game:
    def __init__(self, sticky, cap):
        self.g = _make_game_cls()(sticky=sticky, max_steps=10**9); self.cap = cap

    def reset(self, seed, file):
        from msclear import ram as R
        from msclear.m13_env import restore
        g = self.g; g.reset(int(seed))
        if file:
            restore(g, pickle.load(open(file, "rb"))); c = g.ale.getScreenRGB(); g.colour = colour_pp(c, c)
        self.seed, self.file = int(seed), os.path.basename(file) if file else "fresh"
        self.b0 = R.board(g.ram); self.e0 = int(g.ram[R.DOTS_EATEN]); self.items = 0; self.n = 0; self.d = 0
        self.score0 = g.score; self.cur = self.b0; self.done = False
        return g.stack, g.colour

    def step(self, a):
        from msclear import ram as R
        g = self.g; _, _, _, info = g.step(int(a)); self.n += 1; self.d += int(info["lost_life"])
        b = R.board(g.ram)
        if b == self.b0:
            self.items = max(self.items, int(g.ram[R.DOTS_EATEN]) - self.e0)
        elif self.cur == self.b0:
            self.items = R.items_total(self.b0) - self.e0
        self.cur = max(self.cur, b)
        over = g.ale.game_over()
        self.d += int(over and not info["lost_life"])
        self.done = over or self.n >= self.cap or self.cur > self.b0

    def result(self):
        from msclear import ram as R
        return dict(seed=self.seed, file=self.file, b0=self.b0, cleared=self.cur > self.b0, deaths=self.d,
                    clean=bool(self.cur > self.b0 and self.d == 0), items=self.items, left0=R.items_total(self.b0) - self.e0,
                    steps=self.n, score=self.g.score - self.score0, cap_hit=bool(self.n >= self.cap and not self.g.ale.game_over()))


def _worker(remote, encode):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    import cv2
    cv2.setNumThreads(1)
    enc = (lambda im: cv2.imencode(".png", im, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()) if encode else None
    games = []
    while True:
        try:
            cmd, data = remote.recv()
        except (EOFError, KeyboardInterrupt):
            break
        try:
            if cmd == "reset":
                for g in games:
                    g.g.env.close()
                specs, sticky, cap = data; games = []; st, co = [], []
                for seed, file in specs:
                    g = Game(sticky, cap); s, c = g.reset(seed, file); games.append(g); st.append(s); co.append(c)
                remote.send((np.stack(st) if st else np.zeros((0, 4, 84, 84), np.uint8),
                             np.stack(co) if co else np.zeros((0, RES, RES, 3), np.uint8)))
            elif cmd == "step":
                done = np.zeros(len(games), np.uint8); fr, co, res, keep = [], [], [], []
                for i, (g, a) in enumerate(zip(games, data)):
                    g.step(a)
                    if g.done:
                        done[i] = 1; res.append(g.result()); g.g.env.close()
                    else:
                        keep.append(g); fr.append(g.g.frame); co.append(g.g.colour)
                games = keep
                fr = np.stack(fr) if fr else np.zeros((0, 84, 84), np.uint8)
                co = np.stack(co) if co else np.zeros((0, RES, RES, 3), np.uint8)
                remote.send((done, fr, co, res))
            elif cmd == "encode":                       # PNG-encode (colour, gray) pairs for the shard writer
                remote.send([(enc(c), enc(g)) for c, g in data])
            elif cmd == "close":
                break
        except Exception:
            import traceback
            remote.send(("__error__", traceback.format_exc())); break


def _recv(r):
    m = r.recv()
    if isinstance(m, tuple) and len(m) == 2 and isinstance(m[0], str) and m[0] == "__error__":
        raise RuntimeError(m[1])
    return m


class Pool:
    def __init__(self, workers, encode=False):
        ctx = mp.get_context("spawn"); main = sys.modules["__main__"]
        saved = {k: main.__dict__.pop(k) for k in ("__file__", "__spec__") if k in main.__dict__}; main.__spec__ = None
        self.remotes, self.procs = [], []
        try:
            for _ in range(workers):
                a, b = ctx.Pipe(); p = ctx.Process(target=_worker, args=(b, encode), daemon=True); p.start(); b.close()
                self.remotes.append(a); self.procs.append(p)
        finally:
            del main.__spec__; main.__dict__.update(saved)

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5); p.is_alive() and p.terminate()


def load_teacher(path, device):
    import torch
    import torch.nn as nn
    from msclear.m10_net import M10Net

    class ValueHead(nn.Module):                      # = scripts/m18_ppo.ValueHead
        def __init__(self, d):
            super().__init__(); self.f = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 1))

        def forward(self, h):
            return self.f(h)[:, 0]

    ck = torch.load(path, map_location="cpu", weights_only=False)
    pi = M10Net(**ck["arch"]); pi.load_state_dict(ck["model"])
    vnet = M10Net(**ck["arch"]); vnet.load_state_dict(ck["vnet"]); vh = ValueHead(ck["arch"]["d"]); vh.load_state_dict(ck["vh"])
    return [m.eval().to(device) for m in (pi, vnet, vh)]


TAU = 0.25          # M22 policy logits = adv / TAU (scripts/m18_ppo.py)


def rollout(pool, specs, teacher, student=None, actor="teacher", eps=0.0, sticky=0.25, cap=8000, rng=None,
            on_step=None, device="cuda", amp=False, progress=None):
    """Play all `specs` [(seed, file|None)] at once (no autoreset). Teacher (M22) always runs: its logits/values label
    every step and give agreement. actor: 'teacher' or 'student' (greedy, eps-random). on_step(gid, colour, gray,
    tlogits, tvalue, actions) gets numpy arrays for the alive games. Returns per-game results (+ agree, n_agree)."""
    import torch
    pi, vnet, vh = teacher; G = len(specs); W = len(pool.remotes)
    groups = [list(range(w, G, W)) for w in range(W)]
    for w, r in enumerate(pool.remotes):
        r.send(("reset", ([specs[i] for i in groups[w]], float(sticky), int(cap))))
    stacks = np.empty((G, 4, 84, 84), np.uint8); colour = np.empty((G, RES, RES, 3), np.uint8)
    for w, r in enumerate(pool.remotes):
        s, c = _recv(r)
        if groups[w]:
            stacks[groups[w]] = s; colour[groups[w]] = c
    alive_w = [list(g) for g in groups]; alive = np.ones(G, bool); results = [None] * G
    agree = np.zeros(G, np.int64); nstep = np.zeros(G, np.int64)
    tc = vc = sc = None; t = 0; rng = rng or np.random.default_rng(0)
    ac = torch.autocast("cuda", torch.bfloat16) if amp else torch.autocast("cuda", enabled=False)
    with torch.no_grad():
        while alive.any():
            idx = np.flatnonzero(alive)
            x = torch.from_numpy(stacks[idx]).to(device)
            z = pi.embed(x); zv = vnet.embed(x)
            if tc is None:
                m = torch.ones(G, dtype=torch.bool, device=device)
                tc = pi.new_cache(G, device); pi.prime(tc, m, z); vc = vnet.new_cache(G, device); vnet.prime(vc, m, zv)
            tl = (pi.adv(pi.step(tc, z)) / TAU).float(); tv = vh(vnet.step(vc, zv)).float()
            ta = tl.argmax(-1)
            if student is not None:
                with ac:
                    so, sc = student.act(torch.from_numpy(colour[idx]).to(device), sc, t)
                sa = so.float().argmax(-1); agree[idx] += (sa == ta).cpu().numpy(); nstep[idx] += 1
            a = (sa if actor == "student" else ta).cpu().numpy()
            if eps > 0:
                rnd = rng.random(len(a)) < eps; a[rnd] = rng.integers(0, 4, int(rnd.sum()))
            if on_step is not None:
                on_step(idx, colour[idx], stacks[idx, -1], tl.cpu().numpy(), tv.cpu().numpy(), a)
            act_at = dict(zip(idx.tolist(), a.tolist()))
            for w, r in enumerate(pool.remotes):
                if alive_w[w]:
                    r.send(("step", [act_at[i] for i in alive_w[w]]))
            for w, r in enumerate(pool.remotes):
                ids = alive_w[w]
                if not ids:
                    continue
                done, fr, co, res = _recv(r); ids = np.asarray(ids)
                for gi, rr in zip(ids[done == 1].tolist(), res):
                    rr["agree"] = float(agree[gi] / max(1, nstep[gi])) if student is not None else None
                    results[gi] = rr; alive[gi] = False
                live = ids[done == 0]
                if len(live):
                    stacks[live, :-1] = stacks[live, 1:]; stacks[live, -1] = fr; colour[live] = co
                alive_w[w] = live.tolist()
            still = alive[idx]
            if not still.all():
                keep = torch.from_numpy(np.flatnonzero(still)).to(device)
                tc = pi.select(tc, keep); vc = vnet.select(vc, keep)
                if sc is not None:
                    sc = student.select(sc, keep)
            t += 1
            if progress is not None and t % 200 == 0:
                progress(t, int(alive.sum()))
    return results
