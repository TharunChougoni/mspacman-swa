"""M3: evaluate the HRA agent (real ghosts, sticky 0.25) on fixed seeds 0..N-1.

Env: N (default 100), WORKERS (default 12), MODEL (default runs/m1/model.pkl), SEED0 (default 0), CFG (optional JSON of HRAConfig fields, e.g. from runs/m4/best.json).
An episode ends at the first board clear or game over; score = game score up to then.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msclear.hra import HRAConfig, evaluate  # noqa: E402

N = int(os.environ.get("N", "100")); W = int(os.environ.get("WORKERS", "12"))
kw = json.loads(os.environ["CFG"]) if os.environ.get("CFG") else {}
if "gammas_g" in kw:
    kw["gammas_g"] = tuple(kw["gammas_g"])
cfg = HRAConfig(**kw)
tag = os.environ.get("TAG", "m3")
out = Path("runs") / tag; out.mkdir(parents=True, exist_ok=True)
MODEL = os.environ.get("MODEL", "runs/m1/model.pkl"); S0 = int(os.environ.get("SEED0", "0"))
print(f"start: {N} episodes, {W} workers, config {cfg.name()}", flush=True)
t0 = time.time(); s, res = evaluate(cfg, range(S0, S0 + N), workers=W, path=MODEL, progress=lambda m: print(m, flush=True), every=float(os.environ.get("EVERY", "30")))
print(f"config: {cfg.name()}   model: {MODEL}")
print(f"{N} episodes, sticky 0.25, seeds {S0}..{S0 + N - 1}: clear rate {s['clear_rate']:.3f} ± {s['clear_se']:.3f}, "
      f"items eaten on board 1 {s['eaten']:.1f} ± {s['eaten_se']:.1f} / 154, score {s['score']:.0f} ± {s['score_se']:.0f}, "
      f"deaths {s['deaths']:.2f}; {time.time() - t0:.0f}s", flush=True)
json.dump(dict(config=cfg.__dict__, summary=s, episodes=res), open(out / f"{tag}_eval.json", "w"), indent=1, default=list)
