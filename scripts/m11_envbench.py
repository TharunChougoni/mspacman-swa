"""M11: env-only throughput of msclear.m10_env.VecGames10 for worker/envs-per-worker configs (random actions)."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msclear.m10_env import VecGames10
out = {}
for nw, per in ((14, 3), (18, 3), (18, 6), (18, 10), (19, 12)):
    v = VecGames10(nw, per, p_snap=0.0); v.reset(); rng = np.random.default_rng(0)
    for _ in range(20): v.step(rng.integers(0, 4, v.n))
    t = time.perf_counter(); n = 150
    for _ in range(n): v.step(rng.integers(0, 4, v.n))
    dt = time.perf_counter() - t; v.close()
    out[f"{nw}x{per}"] = dict(envs=nw * per, ms_per_step=round(dt / n * 1e3, 2), env_steps_per_s=round(n * nw * per / dt))
    print(f"{nw}x{per}", out[f"{nw}x{per}"], flush=True)
Path("runs/m11").mkdir(parents=True, exist_ok=True); Path("runs/m11/envbench.json").write_text(json.dumps(out, indent=1))
