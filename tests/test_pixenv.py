import gymnasium as gym
import numpy as np

from msclear.env import INTRO_FRAMES
from msclear.pixenv import SKIP, PixelGame


def test_one_step_is_four_ale_frames():
    g = PixelGame(sticky=0.25)
    obs = g.reset(seed=3)
    assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8
    assert g.ale.getEpisodeFrameNumber() == INTRO_FRAMES
    for t in range(1, 40):
        prev = g.stack.copy()
        g.step(t % 4)
        assert g.ale.getEpisodeFrameNumber() == INTRO_FRAMES + SKIP * t
        assert np.array_equal(g.stack[:-1], prev[1:])          # stack shifts by exactly one frame


def test_matches_gymnasium_atari_preprocessing():
    """Same frames as AtariPreprocessing(frame_skip=4, max-pool last 2, 84x84 gray) over a frameskip=1 v5 env."""
    env = gym.make("ALE/MsPacman-v5", frameskip=1, repeat_action_probability=0.0)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=0, frame_skip=4, screen_size=84, grayscale_obs=True)
    env.reset(seed=5)
    for _ in range(INTRO_FRAMES // 4):
        o, *_ = env.step(0)
    g = PixelGame(sticky=0.0)
    g.reset(seed=5)
    assert np.array_equal(g.frame, o)
    rng = np.random.default_rng(0)
    for _ in range(60):
        k = int(rng.integers(4))
        o, *_ = env.step(k + 1)
        g.step(k)
        assert np.array_equal(g.frame, o)
