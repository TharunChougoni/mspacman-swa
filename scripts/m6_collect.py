"""M6 collection: HRA demos (MODE=demo) or DAgger rounds (MODE=dagger).

Shard i holds GPS games with seeds SEED0 + i*GPS + j; it is skipped if already on disk,
so a rerun after a power cut only plays the missing shards.
  demo:   executed = HRA action (or uniform random with prob eps = 0.05 in every 4th game);
          sticky cycles over STICKY (default 0.25,0.25,0.25,0.1,0.4) per shard.
  dagger: executed = student (CPU, greedy) with prob 1-BETA, HRA otherwise; label = HRA always.
Env: OUT N_SHARDS GPS SEED0 WORKERS MODE STUDENT BETA STICKY
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from msclear.demos import rollout, write_shard  # noqa: E402
from msclear.pixenv import PixelGame  # noqa: E402

OUT = Path(os.environ.get("OUT", "runs/m6/demos"))
N_SHARDS = int(os.environ.get("N_SHARDS", "150"))
GPS = int(os.environ.get("GPS", "8"))
SEED0 = int(os.environ.get("SEED0", "100000"))
WORKERS = int(os.environ.get("WORKERS", "16"))
MODE = os.environ.get("MODE", "demo")
STUDENT = os.environ.get("STUDENT", "")
BETA = float(os.environ.get("BETA", "0.0"))
STICKY = [float(s) for s in os.environ.get("STICKY", "0.25,0.25,0.25,0.1,0.4").split(",")]
CFG = os.environ.get("CFG", "configs/hra_best.json")

_W = {}


def _init():
    from msclear.hra import HRAConfig, Tables
    kw = json.load(open(CFG)); kw["gammas_g"] = tuple(kw["gammas_g"])
    _W["T"], _W["cfg"] = Tables(), HRAConfig(**kw)
    if MODE == "dagger":
        import torch
        torch.set_num_threads(1)
        from msclear.demos import Student
        from msclear.swa_net import SWAv2
        ck = torch.load(STUDENT, map_location="cpu", weights_only=False)
        net = SWAv2(**ck["arch"]); net.load_state_dict(ck["model"])
        _W["student"] = Student(net)


class Mixed:
    """DAgger actor: student with prob 1-beta, else the teacher's label (set by the caller)."""

    def __init__(self, student, beta, rng):
        self.s, self.beta, self.rng = student, beta, rng
        self.teacher_a = None

    def reset(self):
        self.s.reset()

    def act(self, stack):
        a = self.s.act(stack)                      # always advance the student's token cache
        if self.beta and self.rng.random() < self.beta:
            return self.teacher_a
        return a


def _shard(i):
    from msclear.hra import HRAAgent
    path = OUT / f"shard_{i:05d}.npz"
    if path.exists():
        return i, None
    sticky = STICKY[i % len(STICKY)] if MODE == "demo" else 0.25
    eps_l, rows_l = [], []
    for j in range(GPS):
        seed = SEED0 + i * GPS + j
        rng = np.random.default_rng(seed)
        game = PixelGame(sticky=sticky, max_steps=_W["cfg"].max_steps)
        teacher = HRAAgent(_W["T"], _W["cfg"])
        if MODE == "demo":
            eps = 0.05 if j % 4 == 3 else 0.0
            res, rows = rollout(game, seed, teacher=teacher, eps=eps, rng=rng)
        else:
            eps = 0.0
            actor = Mixed(_W["student"], BETA, rng)

            class T2:                              # teacher wrapper that hands its label to the mixed actor
                def reset(self):
                    teacher.reset()

                def act(self, ram, h):
                    a = teacher.act(ram, h)
                    actor.teacher_a = a - 1
                    self.last_q = teacher.last_q
                    return a
            res, rows = rollout(game, seed, teacher=T2(), student=actor, rng=rng)
        eps_l.append(dict(res, sticky=sticky, eps=eps, mode=MODE, beta=BETA))
        rows_l.append(rows)
    n = write_shard(path, eps_l, rows_l)
    return i, dict(n=n, games=eps_l)


def summarize():
    from msclear.demos import read_shard  # noqa: F401
    eps = []
    nbytes = 0
    for p in sorted(OUT.glob("shard_*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        nbytes += p.stat().st_size
        with np.load(p) as z:
            eps += json.loads(bytes(z["episodes"]).decode())
    n = np.array([e["len"] for e in eps]); cl = np.array([e["cleared"] for e in eps])
    agree = None
    s = dict(games=len(eps), transitions=int(n.sum()), clear_rate=float(cl.mean()),
             transitions_from_clears=int(n[cl].sum()), frac_from_clears=float(n[cl].sum() / n.sum()),
             items=float(np.mean([e["eaten"] for e in eps])), score=float(np.mean([e["score"] for e in eps])),
             deaths=float(np.mean([e["deaths"] for e in eps])), disk_gb=nbytes / 1e9,
             by_sticky={str(k): dict(games=int(sum(e["sticky"] == k for e in eps)),
                                     clear_rate=float(np.mean([e["cleared"] for e in eps if e["sticky"] == k])))
                        for k in sorted({e["sticky"] for e in eps})},
             by_eps={str(k): float(np.mean([e["cleared"] for e in eps if e["eps"] == k]))
                     for k in sorted({e["eps"] for e in eps})}, agree=agree)
    (OUT / "summary.json").write_text(json.dumps(s, indent=1))
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for p in OUT.glob("*.tmp.npz"):
        p.unlink()                                  # half-written shard from a cut
    todo = [i for i in range(N_SHARDS) if not (OUT / f"shard_{i:05d}.npz").exists()]
    print(f"[{MODE}] {OUT}: {N_SHARDS - len(todo)}/{N_SHARDS} shards on disk, playing {len(todo)} x {GPS} games "
          f"with {WORKERS} workers", flush=True)
    t0 = last = time.time(); done = 0; trans = 0; games = 0; clears = 0
    if todo:
        ctx = mp.get_context("spawn" if MODE == "dagger" else "fork")
        with ctx.Pool(WORKERS, initializer=_init) as pool:
            for i, r in pool.imap_unordered(_shard, todo):
                done += 1
                if r:
                    trans += r["n"]; games += len(r["games"]); clears += sum(g["cleared"] for g in r["games"])
                now = time.time()
                if now - last >= 30 or done == len(todo):
                    last = now; el = now - t0
                    print(f"  shards {done}/{len(todo)}  games {games}  clears {clears} ({clears / max(games, 1):.0%})  "
                          f"transitions {trans:,}  elapsed {el:.0f}s  ETA {el / done * (len(todo) - done):.0f}s", flush=True)
    s = summarize()
    print(json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
