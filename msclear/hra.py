"""M3: HRA-style agent on decoded RAM, using the M1 GVF tables.

Q(s,a) = P(s,a)                                     pellet head (normalised, see below)
       - w_g * sum_{normal ghosts g} G_{cell(g)}(s,a) ** k_g        ghost heads, k_g = ln(gamma_g)/ln(0.99)
       + w_b * sum_{frightened g}  G_{cell(g)}(s,a) ** k_b          blue-ghost heads (only while timer allows)
       - kappa * n_episode(s,a)                     count term (breaks loops)
P(s,a) = [max_i G_i(s,a) + mix * mean_i G_i(s,a)] / max_a'(same)   over remaining dots + power pellets.

G_t(s,a) ~ 0.99 ** steps, so G ** k ~ gamma_g ** steps: one cell-GVF table gives every ghost discount.
The max (nearest-target) term keeps the pellet pull from fading when few dots are left (HRA's endgame problem);
dividing by its max over actions puts it in [0, 1] so w_g has the same meaning at every stage of the board.
"""
from __future__ import annotations

import math
import pickle
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from msclear import maze
from msclear import ram as R
from msclear.clear import ClearDetector
from msclear.env import INTRO_FRAMES, make_env


@dataclass
class HRAConfig:
    w_g: float = 1.0
    gammas_g: tuple = (0.8,)          # ghost discount grid; heads are summed (each weighted w_g / len)
    w_b: float = 0.5
    gamma_b: float = 0.9
    mix: float = 0.1
    pp_weight: float = 1.0            # weight of power-pellet targets in the pellet head (dots = 1)
    kappa: float = 0.01
    fright_margin: float = 1.0        # chase a blue ghost only if timer (decisions) > margin * est. steps to it
    lead: float = 0.0                 # also score each normal ghost at its position extrapolated `lead` decisions ahead
    max_steps: int = 6000

    def name(self):
        return ",".join(f"{k}={v}" for k, v in asdict(self).items() if k != "max_steps")


class Tables:
    def __init__(self, path="runs/m1/model.pkl"):
        D = pickle.load(open(path, "rb"))
        self.m = D["model"]; self.Gd = D["Gd"]; self.Gc = D["Gc"].astype(np.float32)
        self.bits = np.array(self.m.dot_bits)
        self.cid = self.m.cid

    def cell_index(self, x, y):
        c = (int(x) // 4, int(y) // 4)
        if c in self.cid:
            return self.cid[c]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cc = (c[0] + dx, c[1] + dy)
            if cc in self.cid:
                return self.cid[cc]
        return None                      # ghost in the pen / off-maze


class HRAAgent:
    def __init__(self, tables: Tables, cfg: HRAConfig):
        self.T, self.cfg = tables, cfg
        self.kg = [math.log(g) / math.log(0.99) for g in cfg.gammas_g]
        self.kb = math.log(cfg.gamma_b) / math.log(0.99)
        self.reset()

    def reset(self):
        self.n = {}; self.h = 2; self.prev_g = None

    def act(self, ram, h):
        T, c = self.T, self.cfg
        x, y = int(ram[R.PLAYER_X]), int(ram[R.PLAYER_Y])
        si = T.m.nearest_state(x, y, h)
        left = np.r_[R.bits(ram)[T.bits].astype(bool), R.power_left(ram)]
        g = T.Gd[si][:, left]
        if g.shape[1]:
            is_pp = np.r_[np.zeros(len(T.bits), bool), np.ones(4, bool)][left]
            gw = g * np.where(is_pp, c.pp_weight, 1.0)
            p = gw.max(1) + c.mix * gw.mean(1)
            p = p / max(p.max(), 1e-9)
        else:
            p = np.zeros(4)
        q = p.copy()
        fr = R.frightened(ram); timer = int(ram[R.FRIGHT_TIMER])
        gpos = [(int(ram[R.GHOST_X[gi]]), int(ram[R.GHOST_Y[gi]])) for gi in range(4)]
        for gi in range(4):
            ci = T.cell_index(*gpos[gi])
            if ci is None:
                continue
            gc = np.clip(T.Gc[si][:, ci], 1e-6, 1.0)
            if c.lead and self.prev_g is not None and not fr[gi]:
                vx, vy = gpos[gi][0] - self.prev_g[gi][0], gpos[gi][1] - self.prev_g[gi][1]
                if abs(vx) + abs(vy) <= 8:
                    cj = T.cell_index(gpos[gi][0] + c.lead * vx, gpos[gi][1] + c.lead * vy)
                    if cj is not None:
                        gc = np.maximum(gc, np.clip(T.Gc[si][:, cj], 1e-6, 1.0))
            if fr[gi]:
                steps = math.log(gc.max()) / math.log(0.99)
                if 2 * timer > c.fright_margin * steps:      # ram[116] drops 1 per 8 frames = 2 decisions
                    q += c.w_b * gc ** self.kb
                continue
            for k in self.kg:
                q -= (c.w_g / len(self.kg)) * gc ** k
        self.prev_g = gpos
        q -= c.kappa * np.array([self.n.get((si, b), 0) for b in range(4)])
        a = int(np.argmax(q))
        self.n[(si, a)] = self.n.get((si, a), 0) + 1
        return a + 1


def run_episode(tables, cfg: HRAConfig, seed: int, sticky=0.25, record=False):
    """One game from reset (sticky actions, real ghosts). Ends at the first board clear or game over."""
    env = make_env(frameskip=1, sticky=sticky); env.reset(seed=seed); ale = env.unwrapped.ale
    aset = env.unwrapped._action_set
    score = 0.0
    for _ in range(INTRO_FRAMES):
        score += ale.act(aset[0])
    ag = HRAAgent(tables, cfg); det = ClearDetector(); det.reset(ale.getRAM())
    h = 2; lives0 = int(ale.getRAM()[R.LIVES]); deaths = 0; prev_l = lives0; max_eaten = 0
    for t in range(cfg.max_steps):
        r = ale.getRAM()
        a = ag.act(r, h)
        p0 = (int(r[R.PLAYER_X]), int(r[R.PLAYER_Y]))
        for _ in range(4):
            score += ale.act(aset[a])
        r = ale.getRAM(); p1 = (int(r[R.PLAYER_X]), int(r[R.PLAYER_Y]))
        if maze.teleported([p0, p1]):
            h = 2
        else:
            h = maze.heading(p0, p1, h)
        if int(r[R.LIVES]) < prev_l:
            deaths += 1
        prev_l = int(r[R.LIVES])
        if R.board(r) == 1:
            max_eaten = max(max_eaten, int(r[R.DOTS_EATEN]))
        if det.update(r):
            return dict(seed=seed, cleared=True, steps=t + 1, score=score, eaten=R.N_ITEMS, deaths=deaths)
        if ale.game_over():
            break
    return dict(seed=seed, cleared=False, steps=t + 1, score=score, eaten=max_eaten, deaths=deaths)


_T = None


def _init(path):
    global _T
    _T = Tables(path)


def _job(args):
    cfg, seed, sticky = args
    return run_episode(_T, cfg, seed, sticky)


def evaluate(cfg: HRAConfig, seeds, workers=12, sticky=0.25, path="runs/m1/model.pkl", pool=None, progress=None, every=30.0):
    import multiprocessing as mp
    jobs = [(cfg, s, sticky) for s in seeds]
    own = pool is None
    if own:
        pool = mp.Pool(workers, initializer=_init, initargs=(path,))
    res = []; t0 = time.time(); last = t0
    for r in pool.imap_unordered(_job, jobs):
        res.append(r)
        if progress and (time.time() - last >= every or len(res) == len(jobs)):
            last = time.time(); el = last - t0; n = len(res)
            progress(f"  episodes {n}/{len(jobs)}  clears {sum(x['cleared'] for x in res)} ({np.mean([x['cleared'] for x in res]):.0%})  "
                     f"items eaten {np.mean([x['eaten'] for x in res]):.1f}/154  score {np.mean([x['score'] for x in res]):.0f}  "
                     f"elapsed {el:.0f}s  ETA {el / n * (len(jobs) - n):.0f}s")
    if own:
        pool.close()
    res.sort(key=lambda r: r["seed"])
    return summarize(res), res


def summarize(res):
    n = len(res); cl = np.array([r["cleared"] for r in res], float)
    sc = np.array([r["score"] for r in res]); ea = np.array([r["eaten"] for r in res], float)
    se = lambda v: float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0  # noqa: E731
    return dict(n=n, clear_rate=float(cl.mean()), clear_se=se(cl), score=float(sc.mean()), score_se=se(sc),
                eaten=float(ea.mean()), eaten_se=se(ea), deaths=float(np.mean([r["deaths"] for r in res])))
