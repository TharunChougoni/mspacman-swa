"""Failure analysis of the best HRA config: every death and every failed game, with context.
Env: N (320 games), SEED0 (40000), WORKERS (16), CFG/MODEL. Writes runs/analysis/{deaths.jsonl,games.jsonl,
summary.json,death_heatmap.png}.
Collision point of a death = last decision before the life is lost where a non-frightened ghost was within 8 px
(Manhattan, RAM coords) of Ms. Pac-Man (start of the final contact); context is taken CTX=4 decisions earlier.
Trap = at that context point every available action leads within TRAP steps (GVF) of some non-frightened ghost.
"""
from __future__ import annotations

import collections
import json
import math
import multiprocessing as mp
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
from msclear.hra import HRAAgent, HRAConfig, Tables  # noqa: E402
from msclear.pngio import write_png  # noqa: E402

OUT = Path("runs/analysis"); OUT.mkdir(parents=True, exist_ok=True)
TRAP = 6
CTX = 4          # decisions before first contact used as the decision context
_T = _cfg = None


def _init(path, kw):
    global _T, _cfg
    _T = Tables(path); _cfg = HRAConfig(**kw)


def steps_to(si, ci):
    g = float(np.clip(_T.Gc[si][:, ci], 1e-9, 1).max())
    return math.log(g) / math.log(0.99)


def context(snap):
    r, h, t, last_fr = snap["ram"], snap["h"], snap["t"], snap["last_fright"]
    x, y = int(r[R.PLAYER_X]), int(r[R.PLAYER_Y]); si = _T.m.nearest_state(x, y, h)
    fr = R.frightened(r); gh = []
    for gi in range(4):
        gx, gy = int(r[R.GHOST_X[gi]]), int(r[R.GHOST_Y[gi]]); ci = _T.cell_index(gx, gy)
        gh.append(dict(ghost=R.GHOST_NAMES[gi], px=abs(gx - x) + abs(gy - y), steps=None if ci is None else round(steps_to(si, ci), 1),
                       fright=bool(fr[gi]), ci=ci))
    avail = [a for a in range(4) if _T.m.sa_count[si, a] > 0]
    # moves that actually change position (junction degree)
    moves = set()
    for a in avail:
        st = np.flatnonzero(_T.m.o_sa == si * 4 + a)
        if len(st):
            o = st[np.argmax(_T.m.o_p[st])]; s2 = _T.m.states[_T.m.o_next[o]]
            if (s2[0], s2[1]) != (x, y):
                moves.add(maze.heading((x, y), (s2[0], s2[1]), -1))
    danger = []
    for a in avail:
        d = [math.log(float(np.clip(_T.Gc[si][a, g["ci"]], 1e-9, 1))) / math.log(0.99) for g in gh if g["ci"] is not None and not g["fright"]]
        danger.append(min(d) if d else 99)
    trapped = bool(danger) and max(danger) <= TRAP
    near = [g for g in gh if not g["fright"] and g["steps"] is not None]
    nearest = min(near, key=lambda g: g["steps"]) if near else None
    return dict(t=t, x=x, y=y, tile=(x // 8, y // 12), heading=h, items_left=R.dots_left(r) + int(R.power_left(r).sum()),
                power_left=int(R.power_left(r).sum()), fright_timer=int(r[R.FRIGHT_TIMER]), any_fright=bool(fr.any()),
                since_fright_end=None if last_fr is None else t - last_fr, junction_moves=len(moves), trapped=trapped,
                exits_danger=[round(v, 1) for v in danger], nearest_ghost=None if nearest is None else nearest["ghost"],
                nearest_steps=None if nearest is None else nearest["steps"],
                ghosts=[{k: v for k, v in g.items() if k != "ci"} for g in gh])


def game(seed):
    env = make_env(frameskip=1, sticky=0.25); env.reset(seed=seed); ale = env.unwrapped.ale; aset = env.unwrapped._action_set
    for _ in range(INTRO_FRAMES):
        ale.act(aset[0])
    ag = HRAAgent(_T, _cfg); det = ClearDetector(); det.reset(ale.getRAM())
    h = 2; prev_l = int(ale.getRAM()[R.LIVES]); hist = collections.deque(maxlen=60); deaths = []; last_fright = None; score = 0
    for t in range(_cfg.max_steps):
        r = ale.getRAM().copy()
        if R.frightened(r).any() or r[R.FRIGHT_TIMER] > 0:
            last_fright = t
        hist.append(dict(ram=r, h=h, t=t, last_fright=last_fright))
        a = ag.act(r, h); p0 = (int(r[R.PLAYER_X]), int(r[R.PLAYER_Y]))
        for _ in range(4):
            score += ale.act(aset[a])
        r2 = ale.getRAM(); p1 = (int(r2[R.PLAYER_X]), int(r2[R.PLAYER_Y]))
        h = 2 if maze.teleported([p0, p1]) else maze.heading(p0, p1, h)
        lost = int(r2[R.LIVES]) < prev_l or ale.game_over()
        prev_l = int(r2[R.LIVES])
        if lost:
            hl = list(hist); ci = None

            def touching(rr):
                fr = R.frightened(rr); px, py = int(rr[R.PLAYER_X]), int(rr[R.PLAYER_Y])
                return any(not fr[g] and abs(int(rr[R.GHOST_X[g]]) - px) + abs(int(rr[R.GHOST_Y[g]]) - py) <= 8 for g in range(4))
            for k in range(len(hl) - 1, -1, -1):
                if touching(hl[k]["ram"]):
                    ci = k; break
            while ci is not None and ci > 0 and touching(hl[ci - 1]["ram"]):   # start of the final contact
                ci -= 1
            if ci is not None:
                d = context(hl[max(ci - CTX, 0)]); d["collision_t"] = hl[ci]["t"]; d["seed"] = seed
                col = hl[ci]["ram"]; d["killer"] = min((g for g in range(4) if not R.frightened(col)[g]),
                                                        key=lambda g: abs(int(col[R.GHOST_X[g]]) - int(col[R.PLAYER_X])) + abs(int(col[R.GHOST_Y[g]]) - int(col[R.PLAYER_Y])))
                d["killer"] = R.GHOST_NAMES[d["killer"]]; d["cx"], d["cy"] = int(col[R.PLAYER_X]), int(col[R.PLAYER_Y])
                deaths.append(d)
            else:
                deaths.append(dict(seed=seed, t=t, unlocated=True))
            hist.clear()
        if det.update(r2):
            return dict(seed=seed, cleared=True, steps=t + 1, score=score, items_left=0, deaths=deaths)
        if ale.game_over():
            break
    r2 = ale.getRAM()
    return dict(seed=seed, cleared=False, steps=t + 1, score=score, items_left=R.dots_left(r2) + int(R.power_left(r2).sum()), deaths=deaths)


if __name__ == "__main__":
    N = int(os.environ.get("N", "320")); S0 = int(os.environ.get("SEED0", "40000")); W = int(os.environ.get("WORKERS", "16"))
    kw = json.loads(os.environ.get("CFG") or Path("runs/m4/best.json").read_text()); kw["gammas_g"] = tuple(kw["gammas_g"])
    model = os.environ.get("MODEL", "runs/m1/model.pkl"); t0 = time.time(); games = []
    print(f"failure analysis: {N} games, seeds {S0}..{S0 + N - 1}, {W} workers, model {model}", flush=True)
    with mp.Pool(W, initializer=_init, initargs=(model, kw)) as pool:
        last = t0
        for g in pool.imap_unordered(game, range(S0, S0 + N)):
            games.append(g)
            if time.time() - last > 30 or len(games) == N:
                last = time.time(); el = last - t0
                print(f"  games {len(games)}/{N}  clears {sum(x['cleared'] for x in games)} ({np.mean([x['cleared'] for x in games]):.0%})  "
                      f"deaths {sum(len(x['deaths']) for x in games)}  mean score {np.mean([x['score'] for x in games]):.0f}  "
                      f"elapsed {el:.0f}s  ETA {el / len(games) * (N - len(games)):.0f}s", flush=True)
    deaths = [d for g in games for d in g["deaths"]]
    with open(OUT / "deaths.jsonl", "w") as f:
        for d in deaths:
            f.write(json.dumps(d, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    with open(OUT / "games.jsonl", "w") as f:
        for g in games:
            f.write(json.dumps({k: v for k, v in g.items() if k != "deaths"} | {"n_deaths": len(g["deaths"])}) + "\n")
    loc = [d for d in deaths if not d.get("unlocated")]
    fails = [g for g in games if not g["cleared"]]
    buckets = [(154, 121), (120, 81), (80, 41), (40, 11), (10, 1)]
    phase = {f"{hi}-{lo}": sum(lo <= d["items_left"] <= hi for d in loc) for hi, lo in buckets}
    fail_phase = {f"{hi}-{lo}": sum(lo <= g["items_left"] <= hi for g in fails) for hi, lo in buckets}
    sfe = [d["since_fright_end"] for d in loc if d["since_fright_end"] is not None and not d["any_fright"]]
    summ = dict(games=len(games), clears=sum(g["cleared"] for g in games), deaths=len(deaths), unlocated=len(deaths) - len(loc),
                deaths_by_killer=dict(collections.Counter(d["killer"] for d in loc)),
                deaths_by_phase_items_left=phase, failed_games_by_items_left=fail_phase,
                failed_items_left_median=float(np.median([g["items_left"] for g in fails])) if fails else None,
                trapped=sum(d["trapped"] for d in loc),
                during_fright=sum(d["any_fright"] for d in loc),
                within_10_of_fright_end=sum(s <= 10 for s in sfe), within_20_of_fright_end=sum(s <= 20 for s in sfe),
                at_junction=sum(d["junction_moves"] >= 3 for d in loc), in_corridor=sum(d["junction_moves"] <= 2 for d in loc),
                power_left_at_death=dict(collections.Counter(d["power_left"] for d in loc)),
                nearest_steps_median=float(np.median([d["nearest_steps"] for d in loc if d["nearest_steps"] is not None])),
                top_death_tiles=collections.Counter(f"{d['cx']},{d['cy']}" for d in loc).most_common(8))
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)
    # heatmap over the maze
    env = make_env(frameskip=1, sticky=0.0); env.reset(seed=0); ale = env.unwrapped.ale
    for _ in range(INTRO_FRAMES):
        ale.act(env.unwrapped._action_set[0])
    bg = ale.getScreenRGB().astype(float) * 0.45
    heat = np.zeros(bg.shape[:2])
    for d in loc:
        sx, sy = R.to_screen(d["cx"], d["cy"]); yy, xx = np.mgrid[0:heat.shape[0], 0:heat.shape[1]]
        heat += np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * 3.0 ** 2))
    hn = heat / max(heat.max(), 1e-9)
    img = bg.copy(); img[..., 0] = np.clip(img[..., 0] + 255 * hn, 0, 255); img[..., 1] = np.clip(img[..., 1] + 200 * hn ** 2, 0, 255)
    for d in loc:
        sx, sy = R.to_screen(d["cx"], d["cy"]); xi, yi = int(round(sx)), int(round(sy))
        if 0 <= yi < img.shape[0] and 0 <= xi < img.shape[1]:
            img[yi, xi] = (255, 255, 255)
    write_png(OUT / "death_heatmap.png", img.astype(np.uint8), scale=4)
    print(f"wrote {OUT}/death_heatmap.png, deaths.jsonl, games.jsonl, summary.json in {time.time() - t0:.0f}s", flush=True)
