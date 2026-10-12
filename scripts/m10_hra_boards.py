"""M10-A: can the HRA teacher (maze-1 model runs/m1/model.pkl, configs/hra_best.json) play from the board-2/3
start snapshots of runs/multiboard/states? Plays each snapshot until the board changes (clear), game over, or 6000
decisions. Diagnostics per snapshot: HRA's dot channel (maze-1 dot bits) count vs 154 - ram[119], and whether the
dot channel changes when ram[119] does (does the object channel track real items on this maze?).
Env: STATES (runs/multiboard/states), OUT (runs/multiboard/hra_boards.json), WORKERS (14)."""
from __future__ import annotations
import glob, json, math, os, pickle, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CFG = "configs/hra_best.json"


def play(path):
    from msclear import maze, ram as R
    from msclear.env import make_env
    from msclear.hra import HRAAgent, HRAConfig, Tables
    global _T
    if "_T" not in globals():
        _T = Tables()
    kw = json.load(open(CFG)); kw["gammas_g"] = tuple(kw["gammas_g"])
    d = pickle.load(open(path, "rb"))
    env = make_env(frameskip=1, sticky=0.25); env.reset(seed=d["seed"]); ale = env.unwrapped.ale
    aset = env.unwrapped._action_set
    ale.restoreState(d["state"])
    r = ale.getRAM(); b0 = R.board(r); prev_l = int(r[R.LIVES])
    ag = HRAAgent(_T, HRAConfig(**kw)); h = 2
    chan0 = int(R.bits(r)[_T.bits].sum()); true0 = 154 - int(r[R.DOTS_EATEN])
    items = int(r[R.DOTS_EATEN]); deaths = 0; score = 0.0; cleared = False
    ch_moves = eat_moves = both = 0; last_chan = chan0; last_eat = items
    for t in range(6000):
        r = ale.getRAM(); a = ag.act(r, h); p0 = (int(r[R.PLAYER_X]), int(r[R.PLAYER_Y]))
        for _ in range(4):
            score += ale.act(aset[a])
        r = ale.getRAM(); p1 = (int(r[R.PLAYER_X]), int(r[R.PLAYER_Y]))
        h = 2 if maze.teleported([p0, p1]) else maze.heading(p0, p1, h)
        if int(r[R.LIVES]) < prev_l:
            deaths += 1
        prev_l = int(r[R.LIVES])
        if R.board(r) != b0:
            cleared = True; items = 154; break
        e = int(r[R.DOTS_EATEN]); c = int(R.bits(r)[_T.bits].sum())
        de, dc = e != last_eat, c != last_chan
        eat_moves += de; ch_moves += dc; both += de and dc
        last_eat, last_chan = e, c
        items = max(items, e)
        if ale.game_over():
            deaths += 1; break
    env.close()
    return dict(file=os.path.basename(path), seed=d["seed"], board=b0, cleared=cleared, items=items, deaths=deaths,
                steps=t + 1, score=score, game_over=bool(ale.game_over()), chan0=chan0, true_left0=true0,
                eat_events=eat_moves, chan_events=ch_moves, chan_tracks_eat=both / max(eat_moves, 1))


def main():
    import multiprocessing as mp
    files = sorted(glob.glob(os.environ.get("STATES", "runs/multiboard/states") + "/*.pkl"))
    t0 = time.time()
    with mp.Pool(int(os.environ.get("WORKERS", "14"))) as p:
        res = p.map(play, files, chunksize=1)
    out = {}
    for b in sorted({r["board"] for r in res}):
        rs = [r for r in res if r["board"] == b]; n = len(rs)
        cl = np.array([r["cleared"] for r in rs], float)
        out[b] = dict(n=n, clear_rate=cl.mean(), clear_se=cl.std(ddof=1) / math.sqrt(n) if n > 1 else 0,
                      items_mean=np.mean([r["items"] for r in rs]),
                      items_eaten_from_snapshot=np.mean([r["items"] - (154 - r["true_left0"]) for r in rs]),
                      deaths_mean=np.mean([r["deaths"] for r in rs]), game_over=np.mean([r["game_over"] for r in rs]),
                      chan0_mean=np.mean([r["chan0"] for r in rs]), true_left0_mean=np.mean([r["true_left0"] for r in rs]),
                      chan_tracks_eat=np.mean([r["chan_tracks_eat"] for r in rs]))
        print(f"board {b}: " + json.dumps({k: round(float(v), 3) for k, v in out[b].items()}), flush=True)
    Path(os.environ.get("OUT", "runs/multiboard/hra_boards.json")).write_text(
        json.dumps(dict(summary={str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in out.items()}, games=res), indent=1))
    print(f"wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
