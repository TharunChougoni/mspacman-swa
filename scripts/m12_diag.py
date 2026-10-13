"""M12 stall diagnosis (maze 2, 60 items left). Replays CKPT (pi + vh from ckpt_last) greedy + SAMPLED x tau 0.25
from every runs/m12/late_{train,test}/*.pkl state until the first life loss, board change or MAXT decisions.
Records player/ghost RAM positions, items left, policy entropy, value, deaths. Dots are found on the RGB screen
(maze-2 dots share the wall colour; small connected components of it). Writes runs/m12/diag/*.png + diag.json."""
import glob, json, os, pickle, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msclear import ram as R  # noqa: E402
from msclear.fasteval import _strict_fp32  # noqa: E402
from msclear.m10_net import M10Net  # noqa: E402
from msclear.m11_env import restore  # noqa: E402
from msclear.pixenv import PixelGame, preprocess  # noqa: E402
from scripts.m11_ppo import ValueHead  # noqa: E402
import cv2  # noqa: E402

OUT = Path("runs/m12/diag"); OUT.mkdir(parents=True, exist_ok=True)
WALL2 = np.array([101, 111, 228]); BG = np.array([0, 28, 136]); MAXT = int(os.environ.get("MAXT", "1500"))
SAMPLED = int(os.environ.get("SAMPLED", "2")); TAU = 0.25


def dots_mask(rgb):
    m = (rgb[:172] == WALL2).all(-1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=4)
    small = np.isin(lab, np.flatnonzero(stats[:, cv2.CC_STAT_AREA] <= 8)) & m
    return small                                               # (172, 160) bool: dot pixels (power pellets too)


def main():
    ck = torch.load(os.environ.get("CKPT", "runs/m12/ckpt_last.pt"), map_location="cpu", weights_only=False)
    net = M10Net(**ck["arch"]); net.load_state_dict(ck["pi"]); net = net.cuda().eval()
    vh = ValueHead(ck["arch"]["d"]); vh.load_state_dict(ck["vh"]); vh = vh.cuda().eval()
    files = sorted(glob.glob("runs/m12/late_test/*.pkl") + glob.glob("runs/m12/late_train/*.pkl"))
    g = PixelGame(sticky=0.25, max_steps=10**9); g.reset(0)
    visit = np.zeros((172, 160)); dots_at_start = np.zeros((172, 160)); dots_at_end = np.zeros((172, 160))
    eaten_px = np.zeros((172, 160)); deaths = []; runs = []; bg_img = None
    for fi, f in enumerate(files):
        s = pickle.load(open(f, "rb"))
        for mi, mode in enumerate(["greedy"] + [f"sample{k}" for k in range(SAMPLED)]):
            g.reset(9_000_000 + fi * 10 + mi); restore(g, s); torch.manual_seed(fi * 10 + mi)
            rgb0 = g.ale.getScreenRGB().copy(); d0 = dots_mask(rgb0)
            if bg_img is None:
                bg_img = rgb0.copy()
            dots_at_start += d0
            cache = net.new_cache(1, "cuda"); first = True; stack = g.stack; b0 = R.board(g.ram); lives0 = int(g.ram[R.LIVES])
            traj = []; last_eat = 0; e_prev = int(g.ram[R.DOTS_EATEN]); end = "cap"
            with torch.no_grad(), _strict_fp32():
                for t in range(MAXT):
                    z = net.embed(torch.from_numpy(stack[None].copy()).cuda())
                    if first:
                        net.prime(cache, torch.ones(1, dtype=torch.bool, device="cuda"), z); first = False
                    h = net.step(cache, z); lg = net.adv(h) / TAU; p = F.softmax(lg, -1)[0]
                    ent = float(-(p * p.clamp_min(1e-9).log()).sum()); v = float(vh(h)[0])
                    a = int(lg.argmax(-1)) if mode == "greedy" else int(torch.multinomial(p, 1))
                    r = g.ram; px, py = R.player_xy(r); gh = R.ghosts_xy(r).tolist(); fr = R.frightened(r).tolist()
                    stack, _, _, info = g.step(a)
                    e = int(g.ram[R.DOTS_EATEN])
                    if e != e_prev:
                        last_eat = t
                    e_prev = e
                    traj.append(dict(t=t, x=px, y=py, ghosts=gh, fright=fr, left=154 - e, a=a, ent=ent, v=v, p=p.tolist()))
                    sx, sy = R.to_screen(px, py); ix, iy = int(round(sx)), int(round(sy))
                    if 0 <= iy < 172 and 0 <= ix < 160:
                        visit[max(0, iy - 3):iy + 4, max(0, ix - 2):ix + 3] += 1
                    if info["lost_life"] or g.ale.game_over():
                        end = "death"; break
                    if R.board(g.ram) != b0:
                        end = "clear"; break
            rgb1 = g.ale.getScreenRGB().copy(); d1 = dots_mask(rgb1)
            dots_at_end += d1; eaten_px += d0 & ~d1
            run = dict(file=Path(f).name, mode=mode, end=end, steps=len(traj), left0=154 - int(s["book"]["max_eaten"]) if False else traj[0]["left"],
                       left_end=traj[-1]["left"], lives0=lives0 & 7, since_eat=len(traj) - 1 - last_eat,
                       ent_mean=float(np.mean([x["ent"] for x in traj])), v_mean=float(np.mean([x["v"] for x in traj])))
            # oscillation: distinct 4px cells in the last 60 decisions
            last = traj[-60:]; run["cells_last60"] = len({(x["x"] // 4, x["y"] // 4) for x in last})
            if end == "death":
                x = traj[-1]; dist = [abs(gx - x["x"]) + abs(gy - x["y"]) for gx, gy in x["ghosts"]]
                k = int(np.argmin(dist)); run.update(killer=R.GHOST_NAMES[k], killer_dist=int(dist[k]), x=x["x"], y=x["y"])
                deaths.append(run)
            run["ent_last20"] = float(np.mean([x["ent"] for x in traj[-20:]])); run["v_last20"] = float(np.mean([x["v"] for x in traj[-20:]]))
            run["ent_first20"] = float(np.mean([x["ent"] for x in traj[:20]])); run["v_first20"] = float(np.mean([x["v"] for x in traj[:20]]))
            runs.append(run)
        print(f"{fi + 1}/{len(files)} {Path(f).name}: " + ", ".join(f"{r_['mode']}:{r_['end']}@{r_['steps']} left {r_['left_end']}" for r_ in runs[-1 - SAMPLED:]), flush=True)
    np.savez_compressed(OUT / "maps.npz", visit=visit, dots_at_start=dots_at_start, dots_at_end=dots_at_end, eaten=eaten_px, bg=bg_img)
    json.dump(dict(runs=runs), open(OUT / "diag.json", "w"), indent=1)
    figures(bg_img, visit, dots_at_start, dots_at_end, eaten_px, runs, len(runs))


def _heat(v, cmap):
    v = np.clip(v, 0, 1); c = np.array(cmap, float)
    return (c[0] + (c[1] - c[0]) * v[..., None])


def figures(bg, visit, d0, d1, eaten, runs, n):
    """PIL only (no matplotlib in .venv). Panels 3x upscaled."""
    from PIL import Image, ImageDraw
    bgm = bg[:172].astype(float)
    def panel(over, alpha, title, cmap):
        a = (alpha[..., None]).clip(0, 1)
        im = bgm * 0.45 * (1 - a) + _heat(over, cmap) * a
        img = Image.fromarray(im.clip(0, 255).astype(np.uint8)).resize((480, 516), Image.NEAREST)
        cv = Image.new("RGB", (480, 540)); cv.paste(img, (0, 24)); ImageDraw.Draw(cv).text((6, 6), title, fill=(255, 255, 255)); return cv
    vl = np.log1p(visit); vl = vl / max(vl.max(), 1e-9)
    p1 = panel(vl, (vl > 0) * 0.85, "(a) visited positions (log) from 60-left", [(60, 0, 0), (255, 255, 0)])
    f1 = d1 / n; p2 = panel(f1, (f1 > 0) * 1.0, "(b) dots REMAINING at rollout end (frac)", [(80, 80, 80), (255, 40, 40)])
    fe = eaten / n; p3 = panel(np.clip(fe * 2, 0, 1), (fe > 0) * 1.0, "dots eaten during rollout (frac x2)", [(40, 80, 40), (0, 255, 0)])
    f0 = d0 / n; p4 = panel(f0, (f0 > 0) * 1.0, "dots present at the 60-left start (frac)", [(80, 80, 80), (255, 200, 0)])
    out = Image.new("RGB", (480 * 4, 540)); [out.paste(p, (480 * i, 0)) for i, p in enumerate((p1, p4, p2, p3))]
    out.save(OUT / "maze2_heatmaps.png")
    dr = [r for r in runs if r["end"] == "death"]
    lines = [f"rollouts {len(runs)}  deaths {len(dr)}  clears {sum(r['end'] == 'clear' for r in runs)}  cap {sum(r['end'] == 'cap' for r in runs)}",
             "nearest ghost at death: " + ", ".join(f"{k} {sum(r.get('killer') == k for r in dr)}" for k in R.GHOST_NAMES),
             f"decisions since last item at death: median {np.median([r['since_eat'] for r in dr]):.0f}, p25 {np.percentile([r['since_eat'] for r in dr], 25):.0f}, p75 {np.percentile([r['since_eat'] for r in dr], 75):.0f}",
             f"items eaten from 60 left: mean {np.mean([r['left0'] - r['left_end'] for r in runs]):.1f}",
             f"distinct 4px cells in last 60 decisions: median {np.median([r['cells_last60'] for r in runs]):.0f}",
             f"entropy first20 {np.mean([r['ent_first20'] for r in runs]):.3f} last20 {np.mean([r['ent_last20'] for r in runs]):.3f}; value first20 {np.mean([r['v_first20'] for r in runs]):.2f} last20 {np.mean([r['v_last20'] for r in runs]):.2f}"]
    img = Image.new("RGB", (900, 20 * len(lines) + 10)); dd = ImageDraw.Draw(img)
    for i, l in enumerate(lines):
        dd.text((6, 6 + 20 * i), l, fill=(255, 255, 255))
    img.save(OUT / "deaths_summary.png"); (OUT / "summary.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
