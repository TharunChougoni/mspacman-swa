import json

import numpy as np

from msclear import ram as R
from msclear.clear import ClearDetector
from msclear.env import make_env


def _fake(dots, level):
    r = np.zeros(128, np.uint8); r[R.LEVEL] = level
    m = R.dot_map()["mask"]; b = np.zeros(R.N_BITS, bool); b[np.flatnonzero(m)[:dots]] = True
    r[R.PELLET_SLICE] = np.packbits(b); return r


def test_synthetic():
    d = ClearDetector(); d.reset(_fake(150, 20))
    seq = [(100, 20), (3, 20), (1, 20), (150, 42)]
    assert [d.update(_fake(*s)) for s in seq] == [False, False, False, True]
    # game over -> new game: dots refill but level does not increase -> no clear
    d.reset(_fake(150, 20))
    assert [d.update(_fake(*s)) for s in [(2, 20), (150, 20)]] == [False, False]


def test_real_clear_replay():
    t = json.loads((R.DATA / "clear_traj.json").read_text())
    env = make_env(frameskip=4, sticky=0.0); env.reset(seed=0); ale = env.unwrapped.ale
    det = ClearDetector(); det.reset(ale.getRAM()); fires = []
    acts = t["pre"] + t["board1"]
    for i, a in enumerate(acts):
        env.step(a)
        if det.update(ale.getRAM()):
            fires.append(i)
    assert fires == [len(acts) - 1]
    r = ale.getRAM(); assert R.board(r) == 2 and R.dots_left(r) == 150 and R.power_left(r).all()
