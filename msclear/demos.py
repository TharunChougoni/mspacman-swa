"""M6 rollouts and demo shards.

`rollout` plays one game in PixelGame. The HRA teacher reads RAM every step and gives the
label; besides the HRA action, `equiv` marks every action with the same next state in
the M1 model (in a corridor a perpendicular input does the same as going on), so the label
is a set, not an arbitrary tie-break. The executed action is the teacher's (demos), a random action with prob eps
(noise injection), or the student's (DAgger / eval). Every step records the newest 84x84
frame the student saw BEFORE acting, so obs t = frames[t-3..t] (clamped to the episode start).

A shard is one compressed .npz with all steps of a handful of games, written atomically
(tmp + os.replace): it either exists complete or not at all.
"""
from __future__ import annotations

import collections
import json
import os
from pathlib import Path

import numpy as np

from msclear.pixenv import PixelGame

STEP_FIELDS = dict(frames=np.uint8, label=np.int8, act=np.int8, reward=np.float32, done=np.uint8,
                   items=np.int16, lives=np.int8, lost=np.uint8, moved=np.uint8, q=np.float16, equiv=np.uint8,
                   ram=np.uint8, h=np.int8, ep=np.int32)


def rollout(game: PixelGame, seed: int, teacher=None, student=None, eps: float = 0.0, rng=None, record=True):
    stack = game.reset(seed)
    if teacher is not None:
        teacher.reset()
    if student is not None:
        student.reset()
    rows = collections.defaultdict(list)
    rng = rng or np.random.default_rng(seed)
    while not game.done:
        a_t, q, eq = -1, np.zeros(4, np.float32), np.ones(4, np.uint8)
        ram_t, h_t = game.ram.copy(), game.h
        if teacher is not None:
            a_t = teacher.act(game.ram, game.h) - 1
            q, eq = teacher.last_q, teacher.last_equiv
        if student is not None:
            a = student.act(stack)
        elif eps and rng.random() < eps:
            a = int(rng.integers(4))
        else:
            a = a_t
        if record:
            rows["frames"].append(game.frame.copy()); rows["label"].append(a_t); rows["act"].append(a)
            rows["items"].append(game.items_left()); rows["lives"].append(int(game.prev_lives)); rows["q"].append(q)
            rows["equiv"].append(np.asarray(eq, np.uint8)); rows["ram"].append(ram_t); rows["h"].append(h_t)
        stack, r, done, info = game.step(a)
        if record:
            rows["reward"].append(r); rows["done"].append(done)
            rows["lost"].append(info["lost_life"]); rows["moved"].append(info["moved"])
    return game.result(seed), rows


def write_shard(path: Path, episodes: list[dict], rows_list: list[dict]) -> int:
    out = {k: [] for k in STEP_FIELDS}
    start = 0
    for i, (ep, rows) in enumerate(zip(episodes, rows_list)):
        n = len(rows["label"])
        ep.update(start=start, len=n)
        start += n
        for k in STEP_FIELDS:
            if k == "ep":
                out[k].append(np.full(n, i, np.int32))
            else:
                out[k].append(np.asarray(rows[k], dtype=STEP_FIELDS[k]).reshape(n, *np.shape(rows[k][0])))
    arrays = {k: np.concatenate(v) for k, v in out.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, episodes=np.frombuffer(json.dumps(episodes).encode(), np.uint8), **arrays)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return start


def read_shard(path: Path):
    z = np.load(path)
    eps = json.loads(bytes(z["episodes"]).decode())
    return eps, {k: z[k] for k in STEP_FIELDS}


class Student:
    """Greedy pixel policy with a token cache: one CNN pass per step (CPU or GPU)."""

    def __init__(self, net, device="cpu"):
        import torch
        self.torch, self.net, self.device = torch, net.eval(), device
        self.window = net.window
        self.hist = collections.Counter()

    def reset(self):
        self.z = None

    def act(self, stack: np.ndarray) -> int:
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(stack))[None].to(self.device)
            z = self.net.embed(x)[0]
            if self.z is None:
                self.z = collections.deque([z] * self.window, maxlen=self.window)
            else:
                self.z.append(z)
            q = self.net.head(torch.stack(list(self.z))[None])[0]
        a = int(q.argmax())
        self.hist[a] += 1
        return a
