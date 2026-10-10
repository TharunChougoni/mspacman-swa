#!/usr/bin/env python3
"""Live TensorBoard for the mspacman-clear runs: converts runs/*/metrics.jsonl -> runs/tb/<run>/ event files.

Run with the separate TensorBoard venv (never the project .venv):
    ~/HermesWorkspace/.tbvenv/bin/python scripts/tb_sync.py            # loop, every 60 s
    ~/HermesWorkspace/.tbvenv/bin/python scripts/tb_sync.py --once     # one pass
    ... --rebuild                                                      # wipe runs/tb/<run>/ and convert from scratch
Incremental: the byte offset of the last complete line read is kept in runs/tb/<run>/.offset; a metrics.jsonl that
shrank (rewritten) is reconverted from scratch. x-axis (step) = env_steps; progress/update maps steps -> PPO update.
Tags: train/* (sps, t_roll, t_train, kl, ent, clipfrac, expl_var, beta, stall_pen, ...), recent/* (sampled-policy
normal-start games: clear, items, deaths), pool/* (endgame pool: clear, size, items0), eval/* (clear_rate, clear_se,
eaten, deaths, steps, score, ...). Nested dicts are flattened, None / non-numeric values skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

from tensorboardX import SummaryWriter

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
TB = RUNS / "tb"
SECTION = {"train_recent": "recent", "pool": "pool"}          # nested dicts in train rows -> their own section
SKIP = {"type", "update", "env_steps"}


def flat(d, prefix):
    for k, v in d.items():
        if k in SKIP and "/" not in prefix:
            continue
        if isinstance(v, dict):
            yield from flat(v, f"{prefix}/{k}")
        elif isinstance(v, bool):
            yield f"{prefix}/{k}", float(v)
        elif isinstance(v, (int, float)):
            if math.isfinite(v):
                yield f"{prefix}/{k}", float(v)
        elif isinstance(v, list) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            for i, x in enumerate(v):
                yield f"{prefix}/{k}_{i}", float(x)


def rows_to_scalars(row):
    t = row.get("type")
    if t == "train":
        out = {}
        for k, v in row.items():
            if isinstance(v, dict) and k in SECTION:
                out.update(flat(v, SECTION[k]))
            elif not isinstance(v, dict):
                out.update(flat({k: v}, "train"))
        return out
    if t == "eval":
        return dict(flat(row, "eval"))
    return {}


class RunSync:
    def __init__(self, name, src):
        self.name, self.src, self.dir = name, src, TB / name
        self.off_f = self.dir / ".offset"
        self.w = None

    def offset(self):
        try:
            return int(self.off_f.read_text())
        except (OSError, ValueError):
            return 0

    def reset(self):
        if self.w is not None:
            self.w.close(); self.w = None
        shutil.rmtree(self.dir, ignore_errors=True)

    def sync(self):
        size = self.src.stat().st_size
        off = self.offset()
        if size < off:                                        # file rewritten / truncated -> start over
            print(f"[{self.name}] metrics.jsonl shrank ({size} < {off}); rebuilding", flush=True)
            self.reset(); off = 0
        if size == off:
            return 0
        with open(self.src, "rb") as f:
            f.seek(off); data = f.read(size - off)
        end = data.rfind(b"\n")
        if end < 0:
            return 0                                          # only a partial line so far
        if self.w is None:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.w = SummaryWriter(str(self.dir), flush_secs=30)
        n = 0
        for line in data[:end + 1].splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            step = row.get("env_steps")
            if not isinstance(step, (int, float)):
                continue
            step = int(step)
            sc = rows_to_scalars(row)
            if not sc:
                continue
            if row.get("type") == "train" and isinstance(row.get("update"), (int, float)):
                sc["progress/update"] = float(row["update"])
            for tag, v in sc.items():
                self.w.add_scalar(tag, v, step)
            n += 1
        self.w.flush()
        self.off_f.write_text(str(off + end + 1))
        return n


def discover():
    return {p.parent.name: p for p in sorted(RUNS.glob("*/metrics.jsonl")) if p.parent.name != "tb"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--every", type=float, default=60.0)
    a = ap.parse_args()
    TB.mkdir(parents=True, exist_ok=True)
    syncs = {}
    if a.rebuild:
        for name in discover():
            shutil.rmtree(TB / name, ignore_errors=True)
    while True:
        for name, src in discover().items():
            s = syncs.setdefault(name, RunSync(name, src))
            try:
                n = s.sync()
                if n:
                    print(f"{time.strftime('%H:%M:%S')} [{name}] +{n} rows", flush=True)
            except Exception as e:                            # keep the loop alive (e.g. a file mid-rewrite)
                print(f"{time.strftime('%H:%M:%S')} [{name}] error: {e!r}", file=sys.stderr, flush=True)
        if a.once:
            break
        time.sleep(a.every)
    for s in syncs.values():
        if s.w is not None:
            s.w.close()


if __name__ == "__main__":
    main()
