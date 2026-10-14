"""M16 start states: replay teacher demos (start snapshot + recorded actions, same reset seed / sticky reseed / lives
top-up as teacher_search) and dump emulator states where the remaining maze-2 dots are mostly in the upper-centre
block (REGION, screen px y0,y1,x0,x1 from the late-state leftover map). Replay fidelity = recorded frame == replayed frame.
Out: runs/m16/cur_states/<demo>_<i>.pkl (m13_env snapshot format + left, dist, frac_region, lives) and fidelity json."""
import glob, json, os, pickle, sys, time
import multiprocessing as mp
import numpy as np

OUT = "runs/m16/cur_states"; REGION = (15, 125, 20, 140); EVERY = int(os.environ.get("EVERY", "6"))
MAX_LEFT, MIN_FRAC = 50, 0.8


def run(a):
    demo, src = a
    import cv2
    from msclear import ram as R, mazedist as MD
    from msclear.pixenv import PixelGame
    from msclear.m11_env import restore, _book
    z = np.load(demo); acts = z["act"]; fr = z["frames"]
    g = PixelGame(sticky=0.25, max_steps=10**9); g.reset(1234); restore(g, pickle.load(open(src, "rb")))
    last = None; match = 0; div = None; saved = 0; base = os.path.basename(demo)[:-4]
    for i, a_ in enumerate(acts):
        if not np.array_equal(g.stack[-1], fr[i]):
            div = i; break
        match += 1
        if R.board(g.ram) == 3 and i % EVERY == 0:
            walk, dots, _ = MD.masks(g.ale.getScreenRGB())
            n, lab, st, cen = cv2.connectedComponentsWithStats(dots.astype(np.uint8), connectivity=4)
            if 1 < n <= MAX_LEFT + 1:
                c = cen[1:]; y0, y1, x0, x1 = REGION
                frac = float(np.mean((c[:, 1] >= y0) & (c[:, 1] <= y1) & (c[:, 0] >= x0) & (c[:, 0] <= x1)))
                if frac >= MIN_FRAC:
                    f = MD.dist_field(walk, dots); r, cc = MD.player_px(int(g.ram[10]), int(g.ram[16]), walk)
                    d = int(min(f[r, cc], 9999))
                    s = dict(state=g.ale.cloneState(), stack=g.stack.copy(), book=_book(g), left=int(n - 1), dist=d, frac=frac,
                             lives=int(g.ram[R.LIVES]), src=src, demo=base, i=i, board=3, level_byte=int(g.ram[R.LEVEL]))
                    pickle.dump(s, open(f"{OUT}/{base}_{i:05d}.pkl", "wb")); saved += 1
        if i % 2 == 0 and last is not None:                      # teacher's sticky re-seed before each planned step
            root = g.ale.cloneState(include_rng=True)
            for _ in range(3):
                g.ale.act(g.aset[last + 1])
            g.ale.restoreState(root)
        g.step(int(a_)); last = int(a_)
        lv = int(g.ale.getRAM()[R.LIVES])
        if not g.ale.game_over() and (lv & 7) < 2:
            g.ale.setRAM(R.LIVES, (lv & ~7) | 2); g.ram = g.ale.getRAM().copy(); g.prev_lives = int(g.ram[R.LIVES])
    return dict(demo=base, n=len(acts), match=match, div=div, saved=saved)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    srcs = open("runs/teacher/demo_jobs.txt").read().split()
    m = {s.replace("/", "_").replace("runs_", "") + "_r0": s for s in srcs}
    demos = sorted(glob.glob("runs/teacher/demos/*.npz"))
    jobs = [(d, m[os.path.basename(d)[:-4]]) for d in demos if os.path.basename(d)[:-4] in m]
    print(f"{len(jobs)} demos", flush=True); res = []; t0 = time.time()
    with mp.get_context("spawn").Pool(18) as p:
        for r in p.imap_unordered(run, jobs):
            res.append(r)
            if len(res) % 25 == 0:
                print(f"{len(res)} demos, saved {sum(x['saved'] for x in res)}, full-match {np.mean([x['div'] is None for x in res]):.2f}, "
                      f"matched dec frac {sum(x['match'] for x in res) / sum(x['n'] for x in res):.3f} ({time.time() - t0:.0f}s)", flush=True)
    tot = dict(demos=len(res), full_match=float(np.mean([x["div"] is None for x in res])),
               matched_frac=sum(x["match"] for x in res) / sum(x["n"] for x in res),
               div_hist=np.histogram([x["div"] for x in res if x["div"] is not None], bins=[0, 1, 10, 100, 1000, 10**6])[0].tolist(),
               saved=sum(x["saved"] for x in res))
    print("SUMMARY", json.dumps(tot), flush=True); json.dump(dict(summary=tot, demos=res), open("runs/m16/replay_fidelity.json", "w"))
