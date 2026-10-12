"""Video of one M10Net (kind m10) game played through board clears until game over (or CAP decisions).
Greedy, strict fp32 GPU, same incremental KV-cache path as M10Evaluator -> same game as the multiboard eval.
Env: CKPT, SEED (default: furthest game in MB_JSON), MB_JSON, OUT (runs/videos), PREFIX, VIDEO_JSON, CAP (20000)."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msclear import ram as R  # noqa: E402
from msclear.fasteval import _strict_fp32  # noqa: E402
from msclear.m10_fasteval import load_m10  # noqa: E402
from msclear.pixenv import PixelGame  # noqa: E402


def main():
    ck = os.environ.get("CKPT", "runs/m11/best.pt"); cap = int(os.environ.get("CAP", "20000"))
    seed = os.environ.get("SEED")
    if seed is None:
        g = json.load(open(os.environ.get("MB_JSON", "runs/multiboard_m11/multiboard.json")))["games"]
        seed = max(g, key=lambda r: (r["max_board"], r["score"]))["seed"]
    seed = int(seed); out = Path(os.environ.get("OUT", "runs/videos")); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{os.environ.get('PREFIX', 'multiboard_m11')}_seed{seed}.mp4"
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio
    net = load_m10(ck, "cuda"); g = PixelGame(sticky=0.25, max_steps=10**9)
    wr = imageio.get_writer(path, fps=60, codec="libx264", quality=7, macro_block_size=1); S = 3; fc = [0]

    def frame():
        fc[0] += 1
        if fc[0] % 2:
            return
        rr = g.ale.getRAM()
        img = Image.fromarray(g.ale.getScreenRGB()).resize((160 * S, 210 * S), Image.NEAREST)
        cv = Image.new("RGB", (160 * S, 210 * S + 30)); cv.paste(img, (0, 30))
        ImageDraw.Draw(cv).text((8, 8), f"M11 pixel agent  board {R.board(rr)}  items {int(rr[R.DOTS_EATEN])}/154  lives {int(rr[R.LIVES])}  "
                                         f"score {getattr(g, 'score_live', 0):.0f}  seed {seed}", fill=(255, 255, 255))
        wr.append_data(np.asarray(cv))

    g.on_frame = frame; t0 = time.time()
    stack = g.reset(seed); cache = net.new_cache(1, "cuda")
    with torch.no_grad(), _strict_fp32():
        first = True
        while True:
            z = net.embed(torch.from_numpy(stack[None].copy()).cuda())
            if first:
                net.prime(cache, torch.ones(1, dtype=torch.bool, device="cuda"), z); first = False
            a = int(net.adv(net.step(cache, z)).argmax(-1))
            stack, _, _, _ = g.step(a)
            if g.ale.game_over() or g.steps >= cap:
                break
    wr.close()
    info = dict(path=str(path), seed=seed, board=R.board(g.ram), score=g.score, steps=g.steps, mb=path.stat().st_size / 1e6)
    Path(os.environ.get("VIDEO_JSON", "runs/m11/video.json")).write_text(json.dumps(info))
    print(f"wrote {path}: board {info['board']} score {g.score:.0f} steps {g.steps} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
