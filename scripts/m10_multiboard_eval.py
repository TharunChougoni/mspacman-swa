"""M10: how far does a checkpoint get past board 1? Eval only, greedy fp32, sticky 0.25 (= M9 confirm settings).

Reuses msclear.fasteval (batched GPU, strict fp32) with msclear.multiboard workers: games continue after
board clears until game over or CAP decisions. Env: CKPT, SEED0 (50000), N (100), CAP (20000), WORKERS (14),
OUTDIR (runs/multiboard). Writes OUTDIR/multiboard.json and snapshots OUTDIR/states/seed*_board*.pkl.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msclear import fasteval, multiboard  # noqa: E402


def main():
    ck = os.environ.get("CKPT", "runs/m9run/best.pt")
    s0, n = int(os.environ.get("SEED0", "50000")), int(os.environ.get("N", "100"))
    cap = int(os.environ.get("CAP", "20000"))
    workers = int(os.environ.get("WORKERS", "14"))
    out = Path(os.environ.get("OUTDIR", "runs/multiboard"))
    (out / "states").mkdir(parents=True, exist_ok=True)
    os.environ["MB_CAP"] = str(cap)
    os.environ["MB_STATES"] = str((out / "states").resolve())
    fasteval._MAX_STEPS = cap + 1
    fasteval._worker = multiboard.mb_worker           # spawn target -> MultiGame games
    seeds = list(range(s0, s0 + n))
    print(f"multiboard eval {ck}: seeds {s0}..{s0 + n - 1}, cap {cap}, fp32 greedy, sticky 0.25", flush=True)
    t0 = last = time.time()

    def progress(info):
        nonlocal last
        if time.time() - last < 30 and info["done"] != info["n"]:
            return
        last = time.time()
        res = info["results"]
        mb = collections.Counter(r["max_board"] for r in res)
        print(f"  games {info['done']}/{info['n']} step {info['step']} max-board {dict(sorted(mb.items()))} "
              f"elapsed {time.time() - t0:.0f}s", flush=True)

    with fasteval.FastEvaluator(workers=workers, device="cuda", bf16=False) as ev:
        _, res = ev.evaluate(ck, seeds, sticky=0.25, progress=progress)
    N = len(res)
    maxb = max(r["max_board"] for r in res)
    reach = {b: sum(r["max_board"] >= b for r in res) / N for b in range(1, maxb + 1)}
    clears = dict(sorted(collections.Counter(r["clears"] for r in res).items()))
    deaths_by_board = collections.Counter()
    items_by_board = collections.defaultdict(list)
    for r in res:
        for b, v in r["boards"].items():
            deaths_by_board[int(b)] += v["deaths"]
            items_by_board[int(b)].append(v["items"])
    td = sum(deaths_by_board.values())
    s = dict(ckpt=ck, seed0=s0, n=N, cap=cap, sticky=0.25, fp32=True, greedy=True,
             reach_frac=reach, clears_dist=clears,
             mean_clears=sum(r["clears"] for r in res) / N,
             mean_score=sum(r["score"] for r in res) / N,
             mean_total_items=sum(r["total_items"] for r in res) / N,
             deaths_by_board={b: deaths_by_board[b] for b in sorted(deaths_by_board)},
             death_frac_by_board={b: deaths_by_board[b] / max(td, 1) for b in sorted(deaths_by_board)},
             mean_items_by_board_reached={b: sum(v) / len(v) for b, v in sorted(items_by_board.items())},
             n_reached_by_board={b: len(v) for b, v in sorted(items_by_board.items())},
             cap_hits=sum(r["cap_hit"] for r in res),
             mean_steps=sum(r["steps"] for r in res) / N,
             states=len(list((out / "states").glob("*.pkl"))))
    best = max(res, key=lambda r: (r["max_board"], r["score"]))
    s["best_game"] = dict(seed=best["seed"], max_board=best["max_board"], score=best["score"], steps=best["steps"])
    (out / "multiboard.json").write_text(json.dumps(dict(s, games=res), indent=1))
    print(json.dumps(s, indent=1), flush=True)
    print(f"wall {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
