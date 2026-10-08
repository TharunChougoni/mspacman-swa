from pathlib import Path

import numpy as np
import pytest

from msclear.demos import read_shard, rollout, write_shard
from msclear.pixenv import PixelGame

MODEL = Path("runs/m1/model.pkl")


@pytest.mark.skipif(not MODEL.exists(), reason="needs the M1 model")
def test_rollout_and_shard_roundtrip(tmp_path):
    import json
    from msclear.hra import HRAAgent, HRAConfig, Tables
    kw = json.load(open("configs/hra_best.json")); kw["gammas_g"] = tuple(kw["gammas_g"]); kw["max_steps"] = 60
    T = Tables(str(MODEL))
    eps, rows = [], []
    for seed in (1, 2):
        g = PixelGame(sticky=0.25, max_steps=60)
        res, r = rollout(g, seed, teacher=HRAAgent(T, HRAConfig(**kw)))
        eps.append(dict(res, sticky=0.25, eps=0.0)); rows.append(r)
        assert len(r["label"]) == 60 and all(a == b for a, b in zip(r["label"], r["act"]))
    n = write_shard(tmp_path / "s.npz", eps, rows)
    e2, a = read_shard(tmp_path / "s.npz")
    assert n == 120 and a["frames"].shape == (120, 84, 84) and e2[1]["start"] == 60
    assert np.array_equal(a["frames"][60], rows[1]["frames"][0]) and a["q"].shape == (120, 4)
