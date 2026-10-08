"""Record a board-clear video with the best config: find the first seed (from SEED0) that clears board 1
(fast, no rendering), replay it with rendering, keep playing ~EXTRA frames into board 2, write MP4.
Env: CFG (JSON, default runs/m4/best.json), MODEL, SEED0 (0), EXTRA (360 frames), OUT dir (runs/videos).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msclear import maze  # noqa: E402
from msclear import ram as R  # noqa: E402
from msclear.clear import ClearDetector  # noqa: E402
from msclear.env import INTRO_FRAMES, make_env  # noqa: E402
from msclear.hra import HRAAgent, HRAConfig, Tables, run_episode  # noqa: E402

kw = json.loads(os.environ.get("CFG") or Path("runs/m4/best.json").read_text()); kw["gammas_g"] = tuple(kw["gammas_g"])
cfg = HRAConfig(**kw); T = Tables(os.environ.get("MODEL", "runs/m1/model.pkl"))
OUT = Path(os.environ.get("OUT", "runs/videos")); OUT.mkdir(parents=True, exist_ok=True)
EXTRA = int(os.environ.get("EXTRA", "360")); t0 = time.time()
seed = int(os.environ.get("SEED0", "0"))
while True:
    r = run_episode(T, cfg, seed)
    print(f"seed {seed}: cleared={r['cleared']} items={r['eaten']} score={r['score']:.0f} ({time.time() - t0:.0f}s)", flush=True)
    if r["cleared"]:
        break
    seed += 1

# replay with rendering (same seed -> same sticky-action RNG -> same game)
from PIL import Image, ImageDraw  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

env = make_env(frameskip=1, sticky=0.25); env.reset(seed=seed); ale = env.unwrapped.ale; aset = env.unwrapped._action_set
path = OUT / f"board_clear{os.environ.get('TAG', '')}_seed{seed}.mp4"
wr = imageio.get_writer(path, fps=30, codec="libx264", quality=7, macro_block_size=1)
score = 0.0; fcount = 0; S = 3


def frame():
    global fcount
    fcount += 1
    if fcount % 2:
        return
    rr = ale.getRAM(); img = Image.fromarray(ale.getScreenRGB()).resize((160 * S, 210 * S), Image.NEAREST)
    canvas = Image.new("RGB", (160 * S, 210 * S + 36), (0, 0, 0)); canvas.paste(img, (0, 36))
    left = R.dots_left(rr) + int(R.power_left(rr).sum())
    ImageDraw.Draw(canvas).text((8, 10), f"score {score:5.0f}   items left {left:3d}/154   lives {int(rr[R.LIVES])}   board {R.board(rr)}   seed {seed}", fill=(255, 255, 255))
    wr.append_data(np.asarray(canvas))


for _ in range(INTRO_FRAMES):
    score += ale.act(aset[0]); frame()
ag = HRAAgent(T, cfg); det = ClearDetector(); det.reset(ale.getRAM()); h = 2; cleared_at = None
for t in range(cfg.max_steps):
    rr = ale.getRAM(); a = ag.act(rr, h) if cleared_at is None or True else 0
    p0 = (int(rr[R.PLAYER_X]), int(rr[R.PLAYER_Y]))
    for _ in range(4):
        score += ale.act(aset[a]); frame()
    rr = ale.getRAM(); p1 = (int(rr[R.PLAYER_X]), int(rr[R.PLAYER_Y]))
    h = 2 if maze.teleported([p0, p1]) else maze.heading(p0, p1, h)
    if cleared_at is None and det.update(rr):
        cleared_at = fcount; print(f"board 1 cleared at decision {t + 1}, frame {fcount}, score {score:.0f}", flush=True)
        ag.reset()
    if cleared_at is not None and fcount - cleared_at >= EXTRA or ale.game_over():
        break
    if t % 250 == 0:
        print(f"  rendering: decision {t}, items left {R.dots_left(rr) + int(R.power_left(rr).sum())}, score {score:.0f}, {time.time() - t0:.0f}s", flush=True)
wr.close()
print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {fcount // 2} frames @30fps) in {time.time() - t0:.0f}s", flush=True)
