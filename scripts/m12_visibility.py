"""Are maze-2 dots visible after the 84x84 preprocessing? For a maze-1 and a maze-2 (60-left) state: grayscale screen,
the same screen with dot pixels painted background (dot-free), both through pixenv.preprocess. Signal = per-dot
max |processed - processed_dotfree|. Writes runs/m12/diag/visibility.png + visibility.json."""
import json, pickle, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2  # noqa: E402
from msclear.m11_env import restore  # noqa: E402
from msclear.pixenv import PixelGame, preprocess  # noqa: E402

def dotmask(rgb, col):
    m = (rgb == col).all(-1); m[172:] = False
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=4)
    keep = np.flatnonzero(st[:, cv2.CC_STAT_AREA] <= 8); keep = keep[keep > 0]
    return np.isin(lab, keep), lab, keep

g = PixelGame(); g.reset(3); out = {}; panels = []
for name, col, f in (("maze1", [228, 111, 111], None), ("maze2", [101, 111, 228], "runs/m12/late_test/seed50008_board3_r0_l060.pkl")):
    if f:
        restore(g, pickle.load(open(f, "rb")))
    g.ale.act(0)
    rgb = g.ale.getScreenRGB().copy(); gray = g.ale.getScreenGrayscale().copy()
    m, lab, keep = dotmask(rgb, np.array(col))
    bgv = int(np.median(gray[:172][~m[:172] & ((rgb[:172] == [0, 28, 136]).all(-1))]))
    nod = gray.copy(); nod[m] = bgv
    p, p0 = preprocess(gray, gray).astype(int), preprocess(nod, nod).astype(int)
    diff = np.abs(p - p0)
    sig = []
    for k in keep:
        ys, xs = np.nonzero(lab == k); y84 = (ys * 84 / 210).astype(int); x84 = (xs * 84 / 160).astype(int)
        sig.append(int(diff[y84, x84].max()))
    wall = int(np.median(gray[(rgb == col).all(-1) & ~m]))
    out[name] = dict(n_dots=len(keep), dot_gray=int(gray[m].mean()), wall_gray=wall, bg_gray=bgv,
                     dot_signal_84_mean=float(np.mean(sig)), dot_signal_84_min=int(np.min(sig)), dot_signal_84_p10=float(np.percentile(sig, 10)))
    big = lambda a: Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((336, 336), Image.NEAREST).convert("RGB")  # noqa: E731
    pan = Image.new("RGB", (336 * 3, 360)); [pan.paste(big(a), (336 * i, 24)) for i, a in enumerate((p, p0, diff * 3))]
    ImageDraw.Draw(pan).text((6, 6), f"{name}: 84x84 input | dots removed | |diff| x3   dot signal mean {np.mean(sig):.1f} min {np.min(sig)} (gray dot {out[name]['dot_gray']} bg {bgv})", fill=(255, 255, 0))
    panels.append(pan)
img = Image.new("RGB", (1008, 720)); img.paste(panels[0], (0, 0)); img.paste(panels[1], (0, 360)); img.save("runs/m12/diag/visibility.png")
Path("runs/m12/diag/visibility.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out))
