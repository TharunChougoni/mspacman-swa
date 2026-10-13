"""Evaluate the search teacher from snapshot files (parallel over CPU workers); optional demo recording.
Usage: python scripts/teacher_search.py OUT.json FILE_GLOB [FILE_GLOB...]   env: H R COMMIT EPS WORKERS CAP REP DEMO_DIR
Fresh starts: glob 'fresh:SEED0:N' plays boards from reset (sanity on mazes 1/2, stops at board 3 arrival).
"""
import glob, json, os, pickle, sys, time
import multiprocessing as mp
import numpy as np

E = os.environ.get
H, RR, COMMIT, EPS = int(E("H", "40")), int(E("R", "4")), int(E("COMMIT", "4")), float(E("EPS", "0.15"))
REPLAN = int(E("REPLAN", "4"))
CAP, REP, DEMO = int(E("CAP", "3000")), int(E("REP", "1")), E("DEMO_DIR")


def run(job):
    import torch; torch.set_num_threads(1)
    from msclear import ram as R
    from msclear.pixenv import PixelGame
    from msclear.m11_env import restore
    from msclear.search_teacher import Teacher
    src, rep = job
    t0 = time.time()
    g = PixelGame(sticky=float(E("STICKY", "0.25")), max_steps=10**9)
    if src.startswith("fresh:"):
        seed = int(src.split(":")[1]); g.reset(seed); stop_board = 3
    else:
        s = pickle.load(open(src, "rb")); g.reset(1234 + rep); restore(g, s); stop_board = None
    tch = Teacher(H=H, R_=RR, commit=COMMIT, eps=EPS, seed=hash((src, rep)) % 2**31)
    it0 = int(g.ram[R.DOTS_EATEN]); itmax = it0
    lv0 = int(g.ram[R.LEVEL]); b0 = R.board(g.ram); left0 = None
    obs, acts, deaths, n, items, prev = [], [], 0, 0, 0, None; chk = mis = 0
    while n < CAP:
        if n % REPLAN == 0:
            a, _ = tch.act(g, prev, REPLAN); prev = a
        if DEMO: obs.append(g.stack.copy()); acts.append(a)
        _, _, done, info = g.step(a); n += 1; tch.last = a
        j = (n - 1) % REPLAN
        if j < len(tch.plan) and j < len(tch.best_pos) and a == tch.plan[j]:
            chk += 1; mis += int(tch.best_pos[j] != (int(g.ram[10]), int(g.ram[16])))
        deaths += int(info["lost_life"]); itmax = max(itmax, int(g.ram[R.DOTS_EATEN])) if R.board(g.ram) == b0 else itmax
        b = R.board(g.ram)
        if stop_board is None and b > b0: break
        if stop_board is not None and b >= stop_board: break
        if g.ale.game_over(): deaths += 0 if info["lost_life"] else 1; break
    b = R.board(g.ram)
    res = dict(src=os.path.basename(src), rep=rep, start_board=b0, end_board=b, cleared=b > b0 if stop_board is None else b >= 3,
               boards_cleared=b - b0, deaths=deaths, game_over=bool(g.ale.game_over()), decisions=n, sec=time.time() - t0,
               start_items=it0, fid_checked=chk, fid_mismatch=mis, end_items=itmax)
    if DEMO and obs:
        os.makedirs(DEMO, exist_ok=True)
        np.savez_compressed(f"{DEMO}/{res['src']}_r{rep}.npz", obs=np.stack(obs).astype(np.uint8), act=np.array(acts, np.int8))
    return res


if __name__ == "__main__":
    out = sys.argv[1]; srcs = []
    for gl in sys.argv[2:]:
        if gl.startswith("fresh:"):
            _, s0, k = gl.split(":"); srcs += [f"fresh:{int(s0)+i}" for i in range(int(k))]
        else:
            srcs += sorted(glob.glob(gl))
    jobs = [(s, r) for s in srcs for r in range(REP)]
    print(f"REPLAN={REPLAN} HYST={E('HYST','3')} GHOST_R={E('GHOST_R','16')} MIXMAX={E('MIXMAX','0.5')} H={H} R={RR} COMMIT={COMMIT} EPS={EPS} jobs={len(jobs)}", flush=True)
    t0 = time.time(); res = []
    with mp.get_context("spawn").Pool(int(E("WORKERS", "16"))) as p:
        for r in p.imap_unordered(run, jobs):
            res.append(r)
            print(json.dumps(r), flush=True)
    dec = sum(r["decisions"] for r in res); wall = time.time() - t0
    summ = dict(REPLAN=REPLAN, HYST=E('HYST','3'), GHOST_R=E('GHOST_R','16'), MIXMAX=E('MIXMAX','0.5'), H=H, R=RR, COMMIT=COMMIT, EPS=EPS, n=len(res), clear_rate=float(np.mean([r["cleared"] for r in res])),
                deaths_mean=float(np.mean([r["deaths"] for r in res])), game_over=float(np.mean([r["game_over"] for r in res])),
                decisions=dec, wall_s=wall, dec_per_s=dec / wall, dec_per_s_worker=dec / sum(r["sec"] for r in res))
    print("SUMMARY", json.dumps(summ), flush=True)
    json.dump(dict(summary=summ, games=res), open(out, "w"), indent=1)
