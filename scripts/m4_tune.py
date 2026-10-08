"""M4: staged tuning of the HRA aggregator on screening seeds (10000+), then confirm top configs.

Stage 1: w_g x gamma_g grid.  Stage 2: blue-ghost head (w_b, gamma_b, fright_margin).
Stage 3: pellet head (mix, pp_weight) x count term (kappa).  Each stage keeps the best config so far.
Confirm: top 3 configs on 300 fresh screening seeds; best -> runs/m4/best.json.
Final 100-episode number on seeds 0..99: scripts/m3_eval.py with CFG=$(cat runs/m4/best.json) TAG=m4.
Env: WORKERS (12), SCREEN (episodes per config, 60), BUDGET_S (time box in seconds, 7200).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msclear.hra import HRAConfig, _init, evaluate  # noqa: E402

W = int(os.environ.get("WORKERS", "12")); SCREEN = int(os.environ.get("SCREEN", "60"))
BUDGET = float(os.environ.get("BUDGET_S", "7200"))
OUT = Path("runs/m4"); OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time(); log = open(OUT / "trials.jsonl", "a")


def score(s):
    return s["clear_rate"] + 1e-4 * s["eaten"]


def trial(pool, cfg, seeds):
    s, _ = evaluate(cfg, seeds, pool=pool)
    log.write(json.dumps(dict(cfg=asdict(cfg), seeds=[seeds[0], seeds[-1]], **s)) + "\n"); log.flush()
    done = sum(1 for _ in open(OUT / "trials.jsonl"))
    print(f"[{time.time() - t0:5.0f}s, trial {done}] clear {s['clear_rate']:.3f}±{s['clear_se']:.3f} eaten {s['eaten']:.1f} score {s['score']:.0f}  {cfg.name()}", flush=True)
    return s


if __name__ == "__main__":
    pool = mp.Pool(W, initializer=_init, initargs=("runs/m1/model.pkl",))
    seeds = list(range(10000, 10000 + SCREEN))
    best = HRAConfig(); results = []
    stages = [
        [dict(w_g=w, gammas_g=g) for w in (0.5, 1.0, 2.0, 4.0) for g in ((0.7,), (0.8,), (0.85,), (0.9,), (0.7, 0.9), (0.8, 0.95))],
        [dict(w_b=wb, gamma_b=gb, fright_margin=fm) for wb in (0.0, 0.5, 2.0) for gb in (0.8, 0.9) for fm in (0.7, 1.0, 1.5)],
        [dict(mix=mx, pp_weight=pw, kappa=k) for mx in (0.0, 0.1, 0.3) for pw in (0.3, 1.0) for k in (0.003, 0.01, 0.03)],
    ]
    for si, stage in enumerate(stages):
        cand = []
        for kw in stage:
            if time.time() - t0 > BUDGET:
                print("time box reached", flush=True); break
            cfg = replace(best, **kw); s = trial(pool, cfg, seeds); cand.append((score(s), cfg, s)); results.append((score(s), cfg, s))
        if cand:
            best = max(cand, key=lambda z: z[0])[1]
        print(f"stage {si + 1} best: {best.name()}", flush=True)
    top = sorted(results, key=lambda z: -z[0])[:3]
    conf = list(range(20000, 20300)); fin = []
    for _, cfg, _ in top:
        s = trial(pool, cfg, conf); fin.append((score(s), cfg, s))
    b = max(fin, key=lambda z: z[0])
    json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(b[1]).items()}, open(OUT / "best.json", "w"))
    print(f"BEST on 300 confirm seeds: clear {b[2]['clear_rate']:.3f}±{b[2]['clear_se']:.3f} eaten {b[2]['eaten']:.1f} score {b[2]['score']:.0f}  {b[1].name()}", flush=True)
    pool.close()
