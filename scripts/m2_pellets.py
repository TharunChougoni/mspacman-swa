"""M2: pellets-only navigation from the M1 GVFs, ghosts pinned in the pen (harmless), lives topped up.

Targets = remaining dots + power pellets (a board only ends when the 4 power pellets are eaten too).
Policies over remaining targets i:  sum: sum_i G_i(s,a);  nearest: max_i G_i(s,a);  mix: max + 0.1 * mean.
All policies subtract KAPPA * n(s,a) (visits of that state-action in this episode), HRA's count term, which
breaks the loops caused by the ~4 % of transitions that are not deterministic given (x, y, heading).
Runs deterministic (sticky 0) and sticky 0.25 episodes until a board clear (ClearDetector) or MAX_STEPS.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msclear import ram as R  # noqa: E402
from msclear import maze  # noqa: E402
from msclear.clear import ClearDetector  # noqa: E402

OUT = Path("runs/m2"); OUT.mkdir(parents=True, exist_ok=True)
MAX_STEPS = int(os.environ.get("MAX_STEPS", "3000"))
KAPPA = float(os.environ.get("KAPPA", "0.01"))
D = pickle.load(open("runs/m1/model.pkl", "rb")); M = D["model"]; Gd = D["Gd"]
BITS = np.array(M.dot_bits)


def policy_q(kind, g):
    if g.shape[1] == 0:
        return np.zeros(4)
    if kind == "sum":
        return g.sum(1)
    if kind == "nearest":
        return g.max(1)
    return g.max(1) + 0.1 * g.mean(1)


def run(kind, sticky, seed):
    game = maze.PinnedGame(seed=seed, refill_below=None, sticky=sticky)
    det = ClearDetector(); det.reset(game.ale.getRAM()); n = {}; deaths = 0; lv = int(game.ale.getRAM()[R.LIVES])
    for t in range(MAX_STEPS):
        r = game.ale.getRAM()
        left = np.r_[R.bits(r)[BITS].astype(bool), R.power_left(r)]
        x, y = game.pos(); si = M.nearest_state(x, y, game.h)
        if not left.any():
            game.pin_on = False                # let the board-end sequence run normally
        q = policy_q(kind, Gd[si][:, left]) - KAPPA * np.array([n.get((si, b), 0) for b in range(4)])
        a = int(np.argmax(q)) + 1; n[(si, a - 1)] = n.get((si, a - 1), 0) + 1
        game.step(a)
        r = game.ale.getRAM()
        if int(r[R.LIVES]) < lv and game.pin_on:
            deaths += 1
        lv = int(r[R.LIVES])
        if game.pin_on and r[R.LIVES] < 2:
            game.ale.setRAM(R.LIVES, 3); lv = 3
        if det.update(r):
            return dict(kind=kind, sticky=sticky, seed=seed, cleared=True, steps=t + 1, left=0, deaths=deaths)
    return dict(kind=kind, sticky=sticky, seed=seed, cleared=False, steps=MAX_STEPS, left=R.dots_left(game.ale.getRAM()), power_left=int(R.power_left(game.ale.getRAM()).sum()), deaths=deaths)


if __name__ == "__main__":
    import multiprocessing as mp
    jobs = [(k, 0.0, 0) for k in ("sum", "nearest", "mix")] + [(k, 0.25, s) for k in ("sum", "nearest", "mix") for s in range(10)]
    with mp.Pool(int(os.environ.get("WORKERS", "12"))) as pool:
        res = pool.starmap(run, jobs)
    for kind in ("sum", "nearest", "mix"):
        d = [r for r in res if r["kind"] == kind and r["sticky"] == 0][0]
        st = [r for r in res if r["kind"] == kind and r["sticky"] > 0]
        print(f"{kind:8s} deterministic: cleared={d['cleared']} steps={d['steps']} left={d['left']}+{d.get('power_left', 0)}pp | sticky 0.25 x{len(st)}: "
              f"clears {sum(r['cleared'] for r in st)}/{len(st)}, mean steps {np.mean([r['steps'] for r in st]):.0f}, "
              f"dots+power left {[r['left'] + r.get('power_left', 0) for r in st]}, deaths while pinned {sum(r['deaths'] for r in st) + d['deaths']}", flush=True)
    json.dump(res, open(OUT / "m2.json", "w"), indent=1)
