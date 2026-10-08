"""Pixel-only eval of a student checkpoint. Env: CKPT, SEED0 (0), N (100), STICKY (0.25), WORKERS (16), OUT (json)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msclear import m6eval  # noqa: E402


def main():
    ck = os.environ.get("CKPT", "runs/m6/best.pt")
    s0, n = int(os.environ.get("SEED0", "0")), int(os.environ.get("N", "100"))
    sticky = float(os.environ.get("STICKY", "0.25"))
    out = Path(os.environ.get("OUT", "runs/m6/eval.json"))
    pool = m6eval.make_pool(int(os.environ.get("WORKERS", "16")))
    res = []; t0 = last = time.time()
    print(f"eval {ck}: seeds {s0}..{s0 + n - 1}, sticky {sticky}, pixels only", flush=True)
    for r in pool.imap_unordered(m6eval.play, [(ck, s, sticky) for s in range(s0, s0 + n)]):
        res.append(r)
        if time.time() - last >= 30 or len(res) == n:
            last = time.time(); el = last - t0
            print(f"  games {len(res)}/{n}  clears {sum(x['cleared'] for x in res)}  "
                  f"items {sum(x['eaten'] for x in res) / len(res):.1f}  elapsed {el:.0f}s  ETA {el / len(res) * (n - len(res)):.0f}s",
                  flush=True)
    pool.close(); pool.join()
    res.sort(key=lambda r: r["seed"])
    s = m6eval.summarize(res); s.update(ckpt=ck, seed0=s0, sticky=sticky)
    out.write_text(json.dumps(dict(s, games=res)))
    print(json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
