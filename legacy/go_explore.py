"""Go-Explore archive on leftover + player cell. No BC, no GPU.

Cell = (gx, gy, leftover). Return via ALE cloneSystemState.
Random *held* cardinals from a sampled cell; deaths restore, they do not reset progress.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np

from cleanrl_utils.maze_watch import pellet_count
from cleanrl_utils.pellet_bfs import extract, player_pos, seek_action

BIN = 8  # px per spatial bin (160x210 -> 20x27)
A_UP, A_RIGHT, A_LEFT, A_DOWN = 1, 2, 3, 4
CARDINALS = (A_UP, A_RIGHT, A_LEFT, A_DOWN)


def cell_key(px, py, leftover: int, bin_size: int = BIN) -> tuple[int, int, int]:
    leftover = int(leftover)
    if px is None or py is None:
        return (0, 0, leftover)
    return (int(px) // bin_size, int(py) // bin_size, leftover)


def sample_index(
    leftover: np.ndarray,
    visits: np.ndarray,
    rng: np.random.Generator,
    frontier: int | None = 2,
    keys: list | None = None,
    spatial_boost: bool = False,
) -> int:
    """Prefer low leftover. If frontier is set, only sample leftover <= min+frontier.

    spatial_boost (5e corridor-escape, 2026-09-02): within the frontier window,
    down-weight cells whose spatial bin (gx,gy) already holds many archive cells,
    biasing toward under-explored maze regions so the sampler stops hammering one
    corridor (best cells had all collapsed to gx 9-10, gy 13-16).
    """
    leftover = np.asarray(leftover)
    visits = np.asarray(visits, dtype=np.float64)
    idx = np.arange(len(leftover))
    if frontier is not None and leftover.size:
        best = int(leftover.min())
        mask = leftover <= best + int(frontier)
        if np.any(mask):
            idx = np.flatnonzero(mask)
    L = leftover[idx].astype(np.float64)
    V = visits[idx]
    w = (234.0 - L + 1.0) ** 2 / (V + 1.0)
    if spatial_boost and keys is not None:
        from collections import Counter

        bins = Counter((int(keys[i][0]), int(keys[i][1])) for i in idx)
        w = w / (np.asarray([bins[(int(keys[i][0]), int(keys[i][1]))] for i in idx]) + 1.0)
    w = np.maximum(w, 1e-12)
    w /= w.sum()
    return int(idx[int(rng.choice(len(idx), p=w))])


def snapshot(ale) -> bytes:
    return pickle.dumps(ale.cloneSystemState(), protocol=pickle.HIGHEST_PROTOCOL)


def restore(ale, blob: bytes) -> None:
    ale.restoreSystemState(pickle.loads(blob))


def player_xy(rgb: np.ndarray, prev):
    _wall, player, _ghost = extract(rgb)
    pp, _ = player_pos(player, prev)
    return pp


@dataclass
class Archive:
    keys: list[tuple[int, int, int]] = field(default_factory=list)
    leftover: list[int] = field(default_factory=list)
    visits: list[int] = field(default_factory=list)
    states: list[bytes] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    index: dict[tuple[int, int, int], int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def min_leftover(self) -> int:
        return min(self.leftover) if self.leftover else 234

    def add(self, key: tuple[int, int, int], blob: bytes, leftover: int, actions: np.ndarray) -> bool:
        """True if this is a new cell or a shorter path to the same leftover."""
        leftover = int(leftover)
        actions = np.asarray(actions, dtype=np.uint8)
        i = self.index.get(key)
        if i is None:
            self.index[key] = len(self.keys)
            self.keys.append(key)
            self.leftover.append(leftover)
            self.visits.append(0)
            self.states.append(blob)
            self.actions.append(actions)
            return True
        old_l = self.leftover[i]
        if leftover < old_l or (leftover == old_l and actions.size < self.actions[i].size):
            self.leftover[i] = leftover
            self.states[i] = blob
            self.actions[i] = actions
            return True
        return False

    def sample(
        self,
        rng: np.random.Generator,
        frontier: int | None = 2,
        spatial_boost: bool = False,
    ) -> int:
        i = sample_index(
            np.asarray(self.leftover),
            np.asarray(self.visits, dtype=np.int32),
            rng,
            frontier=frontier,
            keys=self.keys,
            spatial_boost=spatial_boost,
        )
        self.visits[i] += 1
        return i


def run_burst(
    ale,
    blob: bytes,
    start_actions,
    hold: int,
    horizon_holds: int,
    rng: np.random.Generator,
    policy: str = "random",
    seek_mode: str = "nearest",
    seek_epsilon: float = 0.12,
):
    """Restore + explore. Returns (found, dead).

    found: list of (key, state_bytes, leftover, actions_uint8) unique to this burst.
    policy=random: held cardinals. policy=seek: flee ghosts, BFS to remaining dots.
    """
    restore(ale, blob)
    actions = [int(a) for a in np.asarray(start_actions, dtype=np.uint8).tolist()]
    lives0 = int(ale.lives())
    xy = None
    local: dict[tuple[int, int, int], tuple[bytes, int, np.ndarray]] = {}
    dead = False
    last_left = -1
    nsteps = int(hold) * int(horizon_holds)
    a = int(CARDINALS[0])
    for k in range(nsteps):
        if ale.game_over():
            dead = True
            break
        # Play THROUGH deaths (only game_over stops the burst). The stored best
        # cells are multi-life states; panic-breaking on the first life loss cut
        # bursts to ~223-615 frames that ate ~0 pellets (verified: 242k visits
        # produced nothing). Stepping lives0 down keeps `dead` accurate while
        # letting the burst traverse 2-3 lives to reach the pellet-eating arc.
        if int(ale.lives()) < lives0:
            dead = True
            lives0 = int(ale.lives())
        if policy == "seek":
            if k % 4 == 0:
                rgb = np.asarray(ale.getScreenRGB())
                a, xy = seek_action(rgb, xy, rng, epsilon=seek_epsilon, mode=seek_mode)
        else:
            if k % hold == 0:
                a = int(CARDINALS[int(rng.integers(0, len(CARDINALS)))])
        ale.act(int(a))
        actions.append(int(a))
        leftover = pellet_count(ale)
        snap = leftover != last_left or (k + 1) % hold == 0 or k == nsteps - 1
        if not snap:
            continue
        last_left = leftover
        if policy != "seek":
            rgb = np.asarray(ale.getScreenRGB())
            xy = player_xy(rgb, xy)
        elif xy is None:
            rgb = np.asarray(ale.getScreenRGB())
            xy = player_xy(rgb, xy)
        key = cell_key(None if xy is None else xy[0], None if xy is None else xy[1], leftover)
        arr = np.asarray(actions, dtype=np.uint8)
        prev = local.get(key)
        if prev is None or leftover < prev[1] or (leftover == prev[1] and arr.size < prev[2].size):
            local[key] = (snapshot(ale), leftover, arr)
    found = [(k, v[0], v[1], v[2]) for k, v in local.items()]
    return found, dead


def worker_loop(
    wid: int,
    job_q,
    res_q,
    env_id: str,
    hold: int,
    horizon_holds: int,
    policy: str = "random",
    seek_mode: str = "nearest",
    seek_epsilon: float = 0.12,
) -> None:
    """Spawn-safe ALE worker. Jobs: (state_bytes, actions, seed) or None to stop."""
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import ale_py  # noqa: F401
    import gymnasium as gym

    env = gym.make(env_id, frameskip=1, repeat_action_probability=0.0)
    env.reset(seed=10_000 + int(wid))
    ale = env.unwrapped.ale
    try:
        while True:
            job = job_q.get()
            if job is None:
                break
            blob, acts, seed = job
            rng = np.random.default_rng(int(seed))
            found, dead = run_burst(
                ale, blob, acts, hold, horizon_holds, rng,
                policy=policy, seek_mode=seek_mode, seek_epsilon=seek_epsilon,
            )
            res_q.put((found, dead))
    finally:
        env.close()


def consider(archive: Archive, ale, actions: list[int], prev_xy) -> tuple[object, bool, int]:
    """Snapshot current ALE into the archive. Returns (xy, added, leftover)."""
    rgb = np.asarray(ale.getScreenRGB())
    xy = player_xy(rgb, prev_xy)
    leftover = pellet_count(ale)
    key = cell_key(None if xy is None else xy[0], None if xy is None else xy[1], leftover)
    i = archive.index.get(key)
    acts = np.asarray(actions, dtype=np.uint8)
    need = i is None or leftover < archive.leftover[i] or (
        leftover == archive.leftover[i] and acts.size < archive.actions[i].size
    )
    added = False
    if need:
        added = archive.add(key, snapshot(ale), leftover, acts)
    return xy, added, leftover
