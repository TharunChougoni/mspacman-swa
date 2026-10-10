"""Batched GPU eval of a student checkpoint.

Env: CKPT, SEED0 (0), N (100), STICKY (0.25), WORKERS (14), BF16 (0), OUT (json).
Same summary and ``{"...summary", "games": [...]}`` payload as scripts/m6_eval.py.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msclear.fasteval import FastEvaluator  # noqa: E402


def main():
    ck = os.environ.get("CKPT", "runs/m6/best.pt")
    s0, n = int(os.environ.get("SEED0", "0")), int(os.environ.get("N", "100"))
    sticky = float(os.environ.get("STICKY", "0.25"))
    workers = int(os.environ.get("WORKERS", "14"))
    bf16 = bool(int(os.environ.get("BF16", "0")))
    out = Path(os.environ.get("OUT", "runs/m9/fasteval.json"))
    seeds = list(range(s0, s0 + n))
    print(f"eval {ck}: seeds {s0}..{s0 + n - 1}, sticky {sticky}, workers {workers}, bf16 {int(bf16)}",
          flush=True)
    t0 = last = time.time()

    def progress(info):
        nonlocal last
        now = time.time()
        done, total = info["done"], info["n"]
        if now - last < 30 and done != total:
            return
        last = now
        el = now - t0
        res = info["results"]
        if res:
            items = sum(x["eaten"] for x in res) / len(res)
            eta = el / len(res) * (total - len(res))
            extra = f"  items {items:.1f}  ETA {eta:.0f}s"
        else:
            extra = ""
        clears = sum(x["cleared"] for x in res)
        print(f"  games {done}/{total}  clears {clears}{extra}  step {info['step']}  elapsed {el:.0f}s",
              flush=True)

    with FastEvaluator(workers=workers, device="cuda", bf16=bf16) as ev:
        s, res = ev.evaluate(ck, seeds, sticky=sticky, progress=progress)
    s.update(ckpt=ck, seed0=s0, sticky=sticky)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(s, games=res)))
    print(json.dumps(s), flush=True)
    print(f"wall {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
