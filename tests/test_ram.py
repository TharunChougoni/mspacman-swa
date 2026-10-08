import numpy as np

from msclear import ram as R
from msclear.env import make_env, skip_intro


def test_dot_map_shape():
    m = R.dot_map()
    assert len(m["bit"]) == 150 and len(set(m["bit"])) == 150
    assert m["mask"].sum() == 150 and m["mask"].shape == (R.N_BITS,)


def test_start_state():
    env = make_env(frameskip=4, sticky=0.0); env.reset(seed=0)
    r = env.unwrapped.ale.getRAM()
    assert R.dots_left(r) == 150
    assert R.bits(r).sum() == 242          # 150 dots + 92 non-dot bits
    assert R.power_left(r).all() and r[R.DOTS_EATEN] == 0
    skip_intro(env); r = env.unwrapped.ale.getRAM()   # Ms. Pac-Man starts moving left ~step 66 even on NOOP
    assert 146 <= R.dots_left(r) <= 150
    assert R.board(r) == 1 and R.lives(r) >= 2
    assert not R.frightened(r).any() and r[R.FRIGHT_TIMER] == 0
    x, y = R.player_xy(r); assert 0 < x < 160 and 0 < y < 200
    assert R.ghosts_xy(r).shape == (4, 2)


def test_screen_mapping_hits_pacman():
    env = make_env(frameskip=4, sticky=0.0, obs_type="rgb"); env.reset(seed=0); skip_intro(env)
    for a in [3] * 10:
        o, *_ = env.step(a)
    r = env.unwrapped.ale.getRAM()
    sx, sy = R.to_screen(*R.player_xy(r))
    crop = o[int(sy) - 6:int(sy) + 7, int(sx) - 5:int(sx) + 6].reshape(-1, 3)
    assert (crop == np.array([210, 164, 74])).all(1).sum() >= 10
