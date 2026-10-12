"""Batched GPU fp32 eval for M10Net with the per-env KV cache (one CNN pass + one query per layer per step).
Same worker pool / protocol / results as msclear.fasteval.FastEvaluator (subclass; only the net step differs)."""
from __future__ import annotations

import contextlib
import os

import numpy as np

from msclear import fasteval as FE


def load_m10(src, device):
    import torch
    from msclear.m10_net import M10Net
    if isinstance(src, (str, os.PathLike)):
        ck = torch.load(os.fspath(src), map_location="cpu", weights_only=False)
        net = M10Net(**ck["arch"]); net.load_state_dict(ck["model"])
    else:
        net = src
    return net.eval().to(device)


class M10Evaluator(FE.FastEvaluator):
    def evaluate(self, net_or_ckpt_path, seeds, sticky=0.25, progress=None):
        import torch
        from msclear.m6eval import summarize
        seeds = [int(s) for s in seeds]; G = len(seeds)
        groups = [[] for _ in range(self.workers)]; gindex = [[] for _ in range(self.workers)]
        for i, s in enumerate(seeds):
            groups[i % self.workers].append(s); gindex[i % self.workers].append(i)
        for w, r in enumerate(self.remotes):
            r.send(("reset", (groups[w], float(sticky))))
        net = load_m10(net_or_ckpt_path, self.device)
        got = [FE._recv(r) for r in self.remotes]
        stacks = np.empty((G, 4, 84, 84), np.uint8)
        for w, st in enumerate(got):
            if gindex[w]:
                stacks[np.asarray(gindex[w])] = st
        alive = np.ones(G, bool); hist = np.zeros((G, 4), np.int64)
        worker_alive = [list(x) for x in gindex]; results = [None] * G; finished = []
        cache = None; step = 0; reported = -1
        with torch.no_grad(), FE._strict_fp32():
            while True:
                idx = np.flatnonzero(alive)
                if len(idx) == 0:
                    break
                if step >= FE._MAX_STEPS:
                    raise RuntimeError("games still running at step cap")
                x = torch.from_numpy(np.ascontiguousarray(stacks[idx])).to(self.device)
                z = net.embed(x)
                if cache is None:
                    cache = net.new_cache(G, self.device)
                    net.prime(cache, torch.ones(G, dtype=torch.bool, device=self.device), z)
                actions = net.adv(net.step(cache, z)).argmax(-1).cpu().numpy()   # argmax Q == argmax adv
                for j, gi in enumerate(idx.tolist()):
                    hist[gi, int(actions[j])] += 1
                act_at = {gi: int(a) for gi, a in zip(idx.tolist(), actions)}
                for w, r in enumerate(self.remotes):
                    if worker_alive[w]:
                        r.send(("step", [act_at[i] for i in worker_alive[w]]))
                new_alive = []
                for w, r in enumerate(self.remotes):
                    ids = worker_alive[w]
                    if not ids:
                        new_alive.append([]); continue
                    done, frames, dres = FE._recv(r)
                    ids_arr = np.asarray(ids); dead = np.flatnonzero(done); live = np.flatnonzero(done == 0)
                    for li, res in zip(dead.tolist(), dres):
                        gi = int(ids_arr[li]); res["hist"] = hist[gi].tolist(); results[gi] = res
                        finished.append(res); alive[gi] = False
                    if len(live):
                        gis = ids_arr[live]
                        stacks[gis, :-1] = stacks[gis, 1:].copy(); stacks[gis, -1] = frames
                    new_alive.append(ids_arr[live].tolist())
                worker_alive = new_alive
                still = alive[idx]
                if not still.all():
                    keep = torch.from_numpy(np.flatnonzero(still)).to(self.device)
                    cache = net.select(cache, keep)
                step += 1
                if progress is not None and (step % 50 == 0 or len(finished) != reported):
                    reported = len(finished); progress(dict(done=reported, n=G, step=step, results=finished))
        out = sorted(results, key=lambda r: r["seed"])
        return summarize(out), out
