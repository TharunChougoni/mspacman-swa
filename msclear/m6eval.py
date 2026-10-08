"""M6 pixel-only evaluation: the student plays from 84x84 stacks, CPU workers (1 thread each).

The HRA teacher is NOT consulted; RAM is read only for the score/items/deaths/clear bookkeeping.
"""
from __future__ import annotations

import collections
import multiprocessing as mp

import numpy as np

_C = {}


def _init():
    import torch
    torch.set_num_threads(1)


def _net(path):
    if _C.get("path") != path:
        import torch
        from msclear.swa_net import SWAv2
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = SWAv2(**ck["arch"]); net.load_state_dict(ck["model"]); net.eval()
        _C.update(path=path, net=net)
    return _C["net"]


def play(args):
    path, seed, sticky = args
    from msclear.demos import Student
    from msclear.pixenv import PixelGame
    st = Student(_net(path))
    g = PixelGame(sticky=sticky)
    stack = g.reset(seed); st.reset()
    while not g.done:
        stack, *_ = g.step(st.act(stack))
    r = g.result(seed)
    r["hist"] = [st.hist.get(a, 0) for a in range(4)]
    return r


def make_pool(workers):
    return mp.get_context("spawn").Pool(workers, initializer=_init)


def summarize(res):
    from msclear.hra import summarize as s
    out = s(res)
    h = np.sum([r["hist"] for r in res], 0)
    out["action_frac"] = (h / max(h.sum(), 1)).round(3).tolist()
    out["steps"] = float(np.mean([r["steps"] for r in res]))
    return out


def evaluate(path, seeds, workers=8, sticky=0.25, pool=None):
    own = pool is None
    pool = pool or make_pool(workers)
    res = sorted(pool.map(play, [(path, s, sticky) for s in seeds]), key=lambda r: r["seed"])
    if own:
        pool.close(); pool.join()
    return summarize(res), res
