"""Board-clear video of the PIXEL student: find the first seed (from SEED0) it clears from pixels,
replay it with rendering (same seed + deterministic greedy student = same game), ~EXTRA frames into board 2.
The 84x84 frame the student sees is shown on the right. Env: CKPT, SEED0 (0), MAX_SEEDS (200), EXTRA (360), OUT.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msclear import m6eval  # noqa: E402
from msclear import ram as R  # noqa: E402
from msclear.demos import Student  # noqa: E402
from msclear.pixenv import PixelGame  # noqa: E402

CK = os.environ.get("CKPT", "runs/m6/best.pt")
OUT = Path(os.environ.get("OUT", "runs/videos")); OUT.mkdir(parents=True, exist_ok=True)
EXTRA = int(os.environ.get("EXTRA", "360"))
S0, MAXS = int(os.environ.get("SEED0", "0")), int(os.environ.get("MAX_SEEDS", "200"))


def main():
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    m6eval._init()
    seed = None
    if os.environ.get("FORCE_SEED"):               # render a given seed without searching (testing)
        seed, ref = int(os.environ["FORCE_SEED"]), None
    pool = m6eval.make_pool(16) if seed is None else None
    for r in (pool.imap(m6eval.play, [(CK, s, 0.25) for s in range(S0, S0 + MAXS)]) if pool else []):
        print(f"seed {r['seed']}: cleared={r['cleared']} items={r['eaten']} score={r['score']:.0f} ({time.time() - t0:.0f}s)", flush=True)
        if r["cleared"]:
            seed, ref = r["seed"], r
            break
    if pool:
        pool.terminate()
    if seed is None:
        print("no clear found"); sys.exit(1)
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio
    st = Student(m6eval._net(CK))
    g = PixelGame(sticky=0.25)
    prefix = os.environ.get("PREFIX", "board_clear_pixel_m6")
    path = OUT / f"{prefix}{('_' + os.environ['TAG']) if os.environ.get('TAG') else ''}_seed{seed}.mp4"
    wr = imageio.get_writer(path, fps=30, codec="libx264", quality=7, macro_block_size=1)
    S = 3; fc = [0]

    def frame():
        fc[0] += 1
        if fc[0] % 2:
            return
        rr = g.ale.getRAM()
        img = Image.fromarray(g.ale.getScreenRGB()).resize((160 * S, 210 * S), Image.NEAREST)
        side = 84 * 3
        canvas = Image.new("RGB", (160 * S + side + 24, 210 * S + 36), (0, 0, 0)); canvas.paste(img, (0, 36))
        if hasattr(g, "stack"):
            canvas.paste(Image.fromarray(g.stack[-1]).resize((side, side), Image.NEAREST).convert("RGB"), (160 * S + 12, 60))
        d = ImageDraw.Draw(canvas)
        left = R.dots_left(rr) + int(R.power_left(rr).sum())
        d.text((8, 10), f"PIXEL student (SWA-v2, no RAM)  score {getattr(g, 'score_live', 0):5.0f}  items left {left:3d}/154  "
                        f"lives {int(rr[R.LIVES])}  board {R.board(rr)}  seed {seed}", fill=(255, 255, 255))
        d.text((160 * S + 12, 40), "student input (84x84, newest)", fill=(200, 200, 200))
        wr.append_data(np.asarray(canvas))

    g.on_frame = frame
    stack = g.reset(seed); st.reset(); cleared_at = None
    while True:
        stack, _, done, _ = g.step(st.act(stack))
        if g.cleared and cleared_at is None:
            cleared_at = fc[0]
            print(f"board 1 cleared at decision {g.steps}, score {g.score:.0f}", flush=True)
        if (cleared_at is not None and fc[0] - cleared_at >= EXTRA) or g.ale.game_over() or g.steps > 8000:
            break
        if cleared_at is None and done:
            break
    wr.close()
    info = dict(path=str(path), seed=seed, cleared=bool(cleared_at), score=g.score, steps=g.steps, deaths=g.deaths,
                mb=path.stat().st_size / 1e6, eval_run=ref)
    Path(os.environ.get("VIDEO_JSON", "runs/m6/video.json")).write_text(json.dumps(info))
    print(f"wrote {path} ({info['mb']:.1f} MB) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
