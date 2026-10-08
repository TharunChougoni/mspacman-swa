"""M0: map every dot bit in ram[59:101] to its screen position.

Method A (exact): restore a fixed state, write a single bit into the bitmap, render one frame,
diff against an empty-bitmap frame. Bits that draw nothing are not dots.
Method B (cross-check, online): play with random held actions; whenever a dot bit clears,
record the player's screen position. Both must agree within a few pixels.
Writes msclear/data/maze1_dots.json and runs/m0/pellet_overlay.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msclear import ram as R  # noqa: E402
from msclear.env import make_env  # noqa: E402
from msclear.pngio import write_png  # noqa: E402

OUT = Path("runs/m0"); OUT.mkdir(parents=True, exist_ok=True)
DOT_RGB = np.array([228, 111, 111])

env = make_env(frameskip=1, sticky=0.0, obs_type="rgb"); env.reset(seed=0); ale = env.unwrapped.ale
for _ in range(250): env.step(0)
S = ale.cloneState(); full = ale.getRAM()[R.PELLET_SLICE].copy()
NB = len(full); P0 = R.PELLET_SLICE.start


def frame(bm):
    ale.restoreState(S)
    for i in range(NB): ale.setRAM(P0 + i, int(bm[i]))
    return env.step(0)[0]


F0 = frame(np.zeros(NB, np.uint8)).astype(int)
Ffull = frame(full)
rec = {"bit": [], "sx": [], "sy": [], "npix": []}
set_bits = np.nonzero(np.unpackbits(full))[0]
for b in set_bits:
    by = np.zeros(NB, np.uint8); by[b // 8] = 1 << (7 - b % 8)
    m = np.abs(frame(by).astype(int) - F0).sum(-1) > 0
    if m.any():
        ys, xs = np.nonzero(m)
        rec["bit"].append(int(b)); rec["sx"].append(float(xs.mean())); rec["sy"].append(float(ys.mean())); rec["npix"].append(int(m.sum()))
n_dots = len(rec["bit"])
print(f"bits set at start: {len(set_bits)}   bits that draw a dot: {n_dots}   non-dot bits: {len(set_bits) - n_dots}")
(R.DATA).mkdir(parents=True, exist_ok=True)
R.DOTS_JSON.write_text(json.dumps({"rom": "ms_pacman", "maze": 1, "note": "screen coords of each dot bit (RAM 59..100, MSB-first)", **rec}))
R.dot_map.cache_clear()

# every mapped dot must sit on a dot-coloured pixel of the full-board frame
ok = sum(bool((Ffull[int(round(y)) - 1:int(round(y)) + 2, int(round(x)) - 2:int(round(x)) + 2] == DOT_RGB).all(-1).any()) for x, y in zip(rec["sx"], rec["sy"]))
dot_px = (Ffull[:172] == DOT_RGB).all(-1).sum()
print(f"mapped dots on a dot-coloured pixel: {ok}/{n_dots};  dot pixels on full frame {dot_px} vs mapped {sum(rec['npix'])}")

# Method B: online mapping by play (frameskip 4, sticky 0, random held actions, lives poked so play continues)
env4 = make_env(frameskip=4, sticky=0.0, obs_type="ram"); env4.reset(seed=1); a4 = env4.unwrapped.ale
rng = np.random.default_rng(0); online = {}; a = 1; prev = a4.getRAM().copy()
d = R.dot_map(); unexplained = counted = 0
for t in range(60000):
    if rng.random() < 0.08: a = int(rng.integers(1, 5))
    _, _, te, tr, _ = env4.step(a)
    r = a4.getRAM().copy()
    if te or tr: env4.reset(); prev = a4.getRAM().copy(); continue
    a4.setRAM(R.LIVES, max(int(r[R.LIVES]), 2))
    flipped = np.nonzero(R.bits(prev).astype(int) - R.bits(r).astype(int) == 1)[0]
    if int(r[R.LEVEL]) == int(prev[R.LEVEL]) and r[R.DOTS_EATEN] >= prev[R.DOTS_EATEN]:   # completeness: counter vs mapped items
        inc = int(r[R.DOTS_EATEN]) - int(prev[R.DOTS_EATEN])
        ate = int(d["mask"][flipped].sum()) + int((R.power_left(prev) & ~R.power_left(r)).sum())
        counted += inc; unexplained += abs(inc - ate)
    for b in flipped:
        online.setdefault(int(b), []).append(R.to_screen(*R.player_xy(r)))
    if R.dots_left(r) == 0 or R.dots_left(r) > R.dots_left(prev) + 50:
        pass
    prev = r
idx = {b: i for i, b in enumerate(d["bit"])}
print(f"completeness: {counted} dots-eaten counter increments, {unexplained} not explained by a mapped dot or power pellet")
errs = []
for b, pts in online.items():
    if b in idx:
        p = np.mean(pts, 0); errs.append(np.hypot(p[0] - d["sx"][idx[b]], p[1] - d["sy"][idx[b]]))
non_dot_flips = [b for b in online if b not in idx]
print(f"online: {len(online)} distinct bits seen clearing; {sum(b in idx for b in online)}/{n_dots} dots covered; "
      f"non-dot bits that ever cleared: {len(non_dot_flips)}; pos error px median {np.median(errs):.1f} p95 {np.percentile(errs, 95):.1f} max {np.max(errs):.1f}")

ov = Ffull.copy()
for x, y in zip(rec["sx"], rec["sy"]):
    xi, yi = int(round(x)), int(round(y))
    ov[max(0, yi - 3):yi + 4, xi] = (0, 255, 0); ov[yi, max(0, xi - 3):xi + 4] = (0, 255, 0)
for b, pts in online.items():
    p = np.mean(pts, 0); xi, yi = int(round(p[0])), int(round(p[1]))
    if 0 <= yi < ov.shape[0] and 0 <= xi < ov.shape[1]: ov[yi, xi] = (0, 255, 255)
write_png(OUT / "pellet_overlay.png", ov, scale=4)
print("wrote", OUT / "pellet_overlay.png")
