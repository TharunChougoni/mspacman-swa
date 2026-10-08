"""M4 refine: stage 1 of m4_tune.py picked w_g at the low edge of its grid (0.5). Extend the grid downward
around the current best (runs/m4/best.json), screen on seeds 10000+, confirm the top 3 plus the current
best on 300 confirm seeds (20000-20299), and overwrite runs/m4/best.json only if a candidate beats it.
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
from msclear.hra import HRAConfig, _init  # noqa: E402
import scripts.m4_tune as T  # noqa: E402

if __name__ == "__main__":
    kw = json.loads(Path("runs/m4/best.json").read_text()); kw["gammas_g"] = tuple(kw["gammas_g"])
    cur = HRAConfig(**kw)
    pool = mp.Pool(T.W, initializer=_init, initargs=("runs/m1/model.pkl",))
    seeds = list(range(10000, 10000 + T.SCREEN)); res = []
    grid = [dict(w_g=w, gammas_g=g) for w in (0.15, 0.25, 0.35, 0.5) for g in ((0.7, 0.9), (0.6, 0.9), (0.7, 0.85), (0.7, 0.95), (0.8,), (0.85,))]
    grid += [dict(kappa=k) for k in (0.003, 0.03)] + [dict(mix=m) for m in (0.0, 0.3)]
    for g in grid:
        cfg = replace(cur, **g); s = T.trial(pool, cfg, seeds); res.append((T.score(s), cfg, s))
    top = [c for _, c, _ in sorted(res, key=lambda z: -z[0])[:3]]
    conf = list(range(20000, 20000 + int(os.environ.get("CONFIRM", "300")))); fin = []
    for cfg in [cur] + [c for c in top if c != cur]:
        s = T.trial(pool, cfg, conf); fin.append((T.score(s), cfg, s))
    b = max(fin, key=lambda z: z[0])
    if b[1] != cur:
        json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(b[1]).items()}, open("runs/m4/best.json", "w"))
    print(f"REFINED BEST on 300 confirm seeds: clear {b[2]['clear_rate']:.3f}±{b[2]['clear_se']:.3f} eaten {b[2]['eaten']:.1f} "
          f"score {b[2]['score']:.0f}  {b[1].name()}  (changed={b[1] != cur})", flush=True)
    pool.close()
