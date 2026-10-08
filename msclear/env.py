from __future__ import annotations

import ale_py  # noqa: F401  (registers ALE/ envs)
import gymnasium as gym

INTRO_FRAMES = 280   # Ms. Pac-Man does not move before ~frame 260


def make_env(frameskip: int = 4, sticky: float = 0.25, obs_type: str = "ram", render: bool = False, **kw):
    """ALE v5 Ms. Pac-Man with ONE frameskip (no extra MaxAndSkip -> 4 frames/decision)."""
    return gym.make(
        "ALE/MsPacman-v5",
        frameskip=frameskip,
        repeat_action_probability=sticky,
        obs_type=obs_type,
        render_mode="rgb_array" if render else None,
        **kw,
    )


def skip_intro(env, frames: int = INTRO_FRAMES) -> None:
    fs = env.unwrapped._frameskip
    for _ in range(max(1, frames // fs)):
        env.step(0)
