"""M6 obs/action alignment checks.

1. Replay: re-play stored demo episodes with their executed actions (same seed -> same sticky RNG) and
   check that stored frame t / RAM t are exactly the live frame / RAM BEFORE action t, and that label t is
   what HRA picks from that RAM (so the label belongs to obs t, not t-1 or t+1).
2. Sync: Ms. Pac-Man's sprite centroid on the raw screen vs her RAM position. A linear fit should have the
   smallest residual at lag 0 (screen and RAM from the same step).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msclear import ram as R  # noqa: E402
from msclear.demos import read_shard  # noqa: E402
from msclear.hra import HRAAgent, HRAConfig, Tables  # noqa: E402
from msclear.pixenv import PixelGame  # noqa: E402

kw = json.load(open("configs/hra_best.json")); kw["gammas_g"] = tuple(kw["gammas_g"])
T = Tables(); cfg = HRAConfig(**kw)

# 1. replay
bad = 0; checked = 0
for shard in ("runs/m6/demos/shard_00000.npz", "runs/m6/demos/shard_00003.npz"):
    eps, a = read_shard(Path(shard))
    for e in eps[:3]:
        s0, n = e["start"], e["len"]
        g = PixelGame(sticky=e["sticky"]); g.reset(e["seed"]); hra = HRAAgent(T, cfg)
        for t in range(n):
            i = s0 + t
            ok_f = np.array_equal(g.frame, a["frames"][i]); ok_r = np.array_equal(g.ram, a["ram"][i])
            lab = hra.act(g.ram, g.h) - 1
            ok_l = lab == a["label"][i]
            if not (ok_f and ok_r and ok_l):
                bad += 1
                if bad < 5:
                    print(f"  mismatch seed {e['seed']} t {t}: frame {ok_f} ram {ok_r} label {ok_l}")
            g.step(int(a["act"][i])); checked += 1
print(f"replay: {checked} steps from 6 episodes, mismatches {bad}")

# 2. sync between RAM and screen
g = PixelGame(sticky=0.0); g.reset(7); rng = np.random.default_rng(0)
hra = HRAAgent(T, cfg); pos, cen = [], []
for t in range(600):
    rgb = g.ale.getScreenRGB()[:172]
    m = np.all(rgb == (210, 164, 74), axis=2)
    if m.sum() > 10:
        ys, xs = np.nonzero(m); cen.append((xs.mean(), ys.mean())); pos.append(R.player_xy(g.ram))
    else:
        cen.append((np.nan, np.nan)); pos.append(R.player_xy(g.ram))
    g.step(hra.act(g.ram, g.h) - 1)
    if g.done:
        break
pos, cen = np.array(pos, float), np.array(cen, float)
for lag in (-2, -1, 0, 1, 2):
    p = pos[max(0, lag):len(pos) + min(0, lag)]; c = cen[max(0, -lag):len(cen) - max(0, lag)]
    ok = ~np.isnan(c).any(1) & (np.abs(np.diff(np.r_[p[:1], p], axis=0)).sum(1) < 8)   # skip teleports
    res = []
    for k in range(2):
        A = np.c_[p[ok, k], np.ones(ok.sum())]; coef, *_ = np.linalg.lstsq(A, c[ok, k], rcond=None)
        res.append(np.sqrt(np.mean((A @ coef - c[ok, k]) ** 2)))
    print(f"sync lag {lag:+d} (screen t vs RAM t{lag:+d}): residual x {res[0]:.2f}px  y {res[1]:.2f}px  (n={ok.sum()})")
