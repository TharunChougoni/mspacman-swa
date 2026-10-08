"""M0: verify the frightened-ghost RAM signal against blue ghosts on screen.

Replays the deterministic clear trajectory (msclear/data/clear_traj.json) at frameskip 1 (each action x4,
identical to frameskip 4 with sticky 0). For each frame and ghost, crops the sprite at its RAM position.
Label: ghost drawn in its normal colour -> normal; drawn in a non-normal colour -> frightened.
Then scores ram[116] and an exhaustive search over every RAM byte/bit and threshold rule.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msclear import ram as R  # noqa: E402
from msclear.env import make_env  # noqa: E402

OUT = Path("runs/m0")
NORMAL = {"sue": (180, 122, 48), "inky": (84, 184, 153), "pinky": (198, 89, 179), "blinky": (200, 72, 72)}
IGNORE = {(0, 28, 136), (228, 111, 111), (210, 164, 74), (0, 0, 0)}   # background, wall/dots, Ms. Pac-Man, black

d = json.loads((R.DATA / "clear_traj.json").read_text())
env = make_env(frameskip=1, sticky=0.0, obs_type="rgb"); env.reset(seed=0); ale = env.unwrapped.ale
rows = []                 # (ram, ghost_idx, label 0/1, colour)
colours = collections.Counter()
for a in d["pre"] + d["board1"] + d["board2"]:
    for _ in range(4):
        o, *_ = env.step(a); r = ale.getRAM().copy()
        for gi, name in enumerate(R.GHOST_NAMES):
            sx, sy = R.to_screen(r[R.GHOST_X[gi]], r[R.GHOST_Y[gi]])
            x0, y0 = int(round(sx)), int(round(sy))
            if not (4 <= x0 < 156 and 6 <= y0 < 166):
                continue
            crop = o[y0 - 6:y0 + 7, x0 - 4:x0 + 5].reshape(-1, 3)
            cs = collections.Counter(map(tuple, crop))
            spr = {c: n for c, n in cs.items() if c not in IGNORE}
            if not spr:
                continue
            c, n = max(spr.items(), key=lambda kv: kv[1])
            if n < 8:
                continue
            if c == NORMAL[name]:
                rows.append((r, gi, 0, c))
            elif c not in NORMAL.values():
                rows.append((r, gi, 1, c)); colours[c] += 1
print("ghost sightings:", len(rows), " frightened-labelled:", sum(x[2] for x in rows), " frightened colours:", colours.most_common(4))
Rm = np.array([x[0] for x in rows], dtype=np.int64); y = np.array([x[2] for x in rows]); g = np.array([x[1] for x in rows])


def acc(pred):
    return float((pred == y).mean())


print("candidate ram[116]: value ranges  fright:", np.unique(Rm[y == 1, 116])[:12], " normal:", np.unique(Rm[y == 0, 116])[:12])
best = []
for j in range(128):
    v = Rm[:, j]
    for t in np.unique(v):
        best.append((acc((v <= t).astype(int)), f"ram[{j}] <= {t}")); best.append((acc((v > t).astype(int)), f"ram[{j}] > {t}"))
    for k in range(8):
        bit = (v >> k) & 1
        best.append((acc(bit), f"ram[{j}] bit{k}")); best.append((acc(1 - bit), f"not ram[{j}] bit{k}"))
# per-ghost byte hypothesis: ram[j + ghost_index]
idx = np.arange(len(g))
for j in range(125):
    v = Rm[idx, j + g]
    for k in range(8):
        bit = (v >> k) & 1
        best.append((acc(bit), f"ram[{j}+ghost] bit{k}")); best.append((acc(1 - bit), f"not ram[{j}+ghost] bit{k}"))
# per-ghost bit hypothesis: one byte with one bit per ghost
for j in range(128):
    for base in range(5):
        p = (Rm[:, j] >> (base + g)) & 1
        best.append((acc(p), f"ram[{j}] bit(base{base}+ghost)")); best.append((acc(1 - p), f"not ram[{j}] bit(base{base}+ghost)"))
best.sort(reverse=True)
print("majority-class baseline:", round(max(y.mean(), 1 - y.mean()), 4))
for a_, rule in best[:8]:
    print(f"  {a_:.4f}  {rule}")
from msclear.ram import frightened  # noqa: E402
pf = np.array([frightened(r)[gi] for r, gi in zip(Rm, g)]).astype(int)
print(f"msclear.ram.frightened(): accuracy {acc(pf):.4f}  (fright recall {pf[y==1].mean():.4f}, normal recall {1-pf[y==0].mean():.4f})")
print(f"ram[116] > 0 as global flag: accuracy {acc((Rm[:,116]>0).astype(int)):.4f}")
json.dump(dict(frightened_acc=acc(pf), timer_acc=acc((Rm[:,116]>0).astype(int)), n=len(rows), n_fright=int(y.sum()), top=[(float(a_), r_) for a_, r_ in best[:8]], colours=[(list(map(int, c)), n) for c, n in colours.most_common(4)]),
          open(OUT / "m0_fright.json", "w"))
