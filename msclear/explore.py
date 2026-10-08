"""RAM-cell Go-Explore on deterministic ALE (frameskip 4, sticky 0).

Cell = (dots left [drawn dots only], which of 12 bitmap regions still hold dots, player tile, lives).
Returns the first action sequence that completes a board (ClearDetector), from a given start state.
"""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from msclear import ram as R
from msclear.clear import ClearDetector
from msclear.env import make_env

GROUPS = np.array_split(np.arange(R.PELLET_SLICE.stop - R.PELLET_SLICE.start), 12)


def _cell(r):
    b = np.asarray(r)[R.PELLET_SLICE]
    return (R.dots_left(r), tuple(int(b[g].any()) for g in GROUPS), int(r[R.PLAYER_X]) // 12, int(r[R.PLAYER_Y]) // 12, int(r[R.LIVES]))


def _worker(seed, budget, start_acts, q, burst=60, p_switch=0.15):
    env = make_env(frameskip=4, sticky=0.0); env.reset(seed=0); ale = env.unwrapped.ale
    for a in start_acts:
        env.step(a)
    r = ale.getRAM().copy()
    arch = {_cell(r): dict(state=ale.cloneState(), acts=[], visits=0)}
    rng = np.random.default_rng(seed); t0 = time.time(); steps = 0
    while time.time() - t0 < budget:
        keys = list(arch)
        dl = np.array([k[0] for k in keys], float); lv = np.array([k[4] for k in keys], float)
        v = np.array([arch[k]["visits"] for k in keys], float)
        w = np.exp(-(dl - dl.min()) / 3.0) * (1.5 ** lv) / np.sqrt(v + 1)
        c = arch[keys[rng.choice(len(keys), p=w / w.sum())]]; c["visits"] += 1
        ale.restoreState(c["state"]); acts = list(c["acts"])
        det = ClearDetector(); det.reset(ale.getRAM()); det.armed = R.dots_left(ale.getRAM()) <= det.low
        a = int(rng.integers(1, 5))
        for _ in range(burst):
            if rng.random() < p_switch:
                a = int(rng.integers(1, 5))
            _, _, te, tr, _ = env.step(a); acts.append(a); steps += 1
            r = ale.getRAM().copy()
            if te or tr:
                break
            if det.update(r):
                q.put(dict(seed=seed, acts=acts, t=time.time() - t0, steps=steps, cells=len(arch))); return
            k = _cell(r)
            if k not in arch or len(acts) < len(arch[k]["acts"]):
                arch[k] = dict(state=ale.cloneState(), acts=list(acts), visits=0)
    q.put(dict(seed=seed, acts=None, t=time.time() - t0, steps=steps, cells=len(arch)))


def find_clear(start_acts, workers=8, budget=900.0):
    """Run `workers` independent explorers; return the first clearing action list (relative to start_acts) or None."""
    q = mp.Queue()
    ps = [mp.Process(target=_worker, args=(s, budget, list(start_acts), q)) for s in range(workers)]
    for p in ps:
        p.start()
    res = None
    for _ in ps:
        r = q.get()
        if r["acts"] is not None:
            res = r; break
    for p in ps:
        p.terminate()
    return res
