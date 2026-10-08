"""M0: get a real board clear, verify the detector on it, find the level byte, check false positives.

1. Go-Explore (msclear/explore.py) from the start state -> clear of board 1, then from there -> clear of board 2.
2. Replay both deterministically, ClearDetector must fire exactly twice, at the right steps.
3. Level byte = RAM byte constant during board 1, constant during board 2, constant during board 3, values differ.
4. False positives: N random episodes (sticky 0.25) -> detector fires 0 times.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msclear import ram as R  # noqa: E402
from msclear.clear import ClearDetector  # noqa: E402
from msclear.env import INTRO_FRAMES, make_env  # noqa: E402
from msclear.explore import find_clear  # noqa: E402

OUT = Path("runs/m0"); OUT.mkdir(parents=True, exist_ok=True)
W = int(os.environ.get("WORKERS", "10"))
TRAJ = R.DATA / "clear_traj.json"
PRE = [0] * (INTRO_FRAMES // 4)


def get_traj():
    if TRAJ.exists():
        return json.loads(TRAJ.read_text())
    t0 = time.time(); r1 = find_clear(PRE, workers=W, budget=1800)
    assert r1, "no board-1 clear found"
    print(f"board 1 clear: {len(r1['acts'])} steps, found in {r1['t']:.0f}s by worker {r1['seed']}", flush=True)
    r2 = find_clear(PRE + r1["acts"], workers=W, budget=1800)
    assert r2, "no board-2 clear found"
    print(f"board 2 clear: {len(r2['acts'])} steps, found in {r2['t']:.0f}s", flush=True)
    d = dict(pre=PRE, board1=r1["acts"], board2=r2["acts"], search_seconds=time.time() - t0)
    TRAJ.write_text(json.dumps(d)); return d


def replay(d, tail=400):
    env = make_env(frameskip=4, sticky=0.0); env.reset(seed=0); ale = env.unwrapped.ale
    acts = d["pre"] + d["board1"] + d["board2"]
    det = ClearDetector(); det.reset(ale.getRAM()); rams, fires = [], []
    rng = np.random.default_rng(0)
    for i, a in enumerate(acts + [int(x) for x in rng.integers(1, 5, size=tail)]):
        _, _, te, tr, _ = env.step(a); r = ale.getRAM().copy(); rams.append(r)
        if det.update(r):
            fires.append(i)
        if te or tr:
            break
    return np.array(rams), fires, len(d["pre"]) + len(d["board1"]) - 1, len(acts) - 1


def level_byte(rams, e1, e2):
    s = len(PRE) + 40
    seg = [rams[s:e1 - 5], rams[e1 + 60:e2 - 5], rams[e2 + 60:]]
    cands = []
    for j in range(128):
        if R.PELLET_SLICE.start <= j < R.PELLET_SLICE.stop:
            continue
        u = [np.unique(x[:, j]) for x in seg]
        if all(len(x) == 1 for x in u) and len({int(x[0]) for x in u}) == 3:
            cands.append((j, [int(x[0]) for x in u]))
    return cands


def _fp_worker(seed, n, q):
    env = make_env(frameskip=4, sticky=0.25); rng = np.random.default_rng(seed); fires = 0; steps = 0; mind = []
    for ep in range(n):
        env.reset(seed=seed * 100000 + ep); ale = env.unwrapped.ale; det = ClearDetector(); det.reset(ale.getRAM()); a = 1
        while True:
            if rng.random() < 0.1:
                a = int(rng.integers(0, 9))
            _, _, te, tr, _ = env.step(a); steps += 1
            fires += det.update(ale.getRAM())
            if te or tr:
                break
        mind.append(det.min_dots)
    q.put((fires, steps, mind))


if __name__ == "__main__":
    d = get_traj()
    rams, fires, e1, e2 = replay(d)
    print(f"replay: detector fired at steps {fires}; expected board ends at {e1} and {e2}", flush=True)
    lv = level_byte(rams, e1, e2)
    print("level-byte candidates (byte, [board1, board2, board3]):", lv, flush=True)
    n_ep = int(os.environ.get("FP_EPISODES", "1000")); q = mp.Queue(); per = n_ep // W
    ps = [mp.Process(target=_fp_worker, args=(s, per, q)) for s in range(W)]
    [p.start() for p in ps]; res = [q.get() for _ in ps]; [p.join() for p in ps]
    f = sum(r[0] for r in res); st = sum(r[1] for r in res); md = np.concatenate([r[2] for r in res])
    print(f"false-positive test: {per * W} random episodes (sticky 0.25), {st} steps, detector fires {f}; "
          f"min dots left per episode: median {np.median(md):.0f}, min {md.min()}", flush=True)
    (OUT / "m0_clear.json").write_text(json.dumps(dict(fires=fires, e1=e1, e2=e2, level_candidates=lv, fp_episodes=per * W,
                                                       fp_fires=int(f), fp_steps=int(st), board1_len=len(d["board1"]), board2_len=len(d["board2"]))))
