"""M13 = M12 envs + lives banking + dot-direction shaping on maze 2.

M13 changes (cfg keys, all default off = M12 behaviour):
  rw_death12   death penalty on boards 1-2 (M12: rw[3] everywhere); boards >= 3 keep rw[3].
  arrive_bonus reward per SPARE life (ram[123] & 7) on the decision Ms. Pac-Man first enters board 3 (once per game).
  shape_k, shape_left, shape_min_board, shape_gamma: potential-based dot shaping, Phi = -shape_k * d, d = MAZE PATH
               distance (px, 4-connected BFS on the screen walk mask incl. side-tunnel wraparound, ghost-house
               interior excluded; msclear.mazedist) from Ms. Pac-Man to the nearest remaining dot; active on boards
               >= shape_min_board with <= shape_left items left; reward += shape_gamma * Phi(s') - Phi(s) (= PPO gamma,
               policy-invariant); Phi resets on a board change / life loss / episode start.
Results gain lives_b3 (spare lives on arrival at board 3, None if never) and arrived_b3.

M12 docstring follows.
M12 = M11 envs + late-board curriculum on maze 2 (board >= 3).

M12 changes: harvest thresholds 120/90/60/45/30/20/15/8 left + pre-death snapshots (ring every 5 decisions,
the entry 40-60 decisions before a life loss with <= 60 left); three per-bucket FIFOs per worker
(early > 60 left, late 31-60, vlate <= 30; cap pool_cap each) so late states are never evicted by early ones;
board-3 starts: p_late of them from late/vlate (vlate with prob p_vlate when non-empty), rest early harvested
(p_harvest) or base board-3 starts. cfg late_files preloads late/vlate (pre-pass + M11 harvest <= 60).

M11 docstring follows.
M11 multi-board PPO envs. PixelGame workers; games play THROUGH board clears (episode = game over or max_steps
decisions from the start). Start mix: p_b1 fresh reset, p_b2 board-2 snapshot, rest board-3+ snapshot (base files or
harvested pool). Maze-agnostic item counting: ram[119] (+1 per dot/pellet, resets to 0 on a new board; 154 per board on maze 1, 150 on maze 2 via R.items_total).

Reward per decision, rw = (item, score, clear, death, k_end):
  rw_item * sum over eaten items (1 + k_end * (1 - rem/30) if rem < 30)  + rw_score * score delta
  + rw_clear * [board index increased] - rw_death * [life lost or game over]  - stall penalty
Stall (items, n, pen, cap): with <= items left on the CURRENT board (items_total - ram[119]) and no item for > n decisions:
  - min(cap, pen + pen * (since - n) / n).
Harvest: on boards >= 3 (any start), a snapshot at the board start and at the first step with <= 120/90/60/30/15 left;
kept in a per-worker FIFO (cap pool_cap) and mirrored to disk (harvest_dir/w{wid}_{slot}.pkl, slot rotates = FIFO).
step() also returns per env the board of the NEW observation (for the maze-1 KL mask).
"""
from __future__ import annotations

import collections
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np

from msclear.pixenv import PixelGame as _PG  # original class (snap_worker patches pixenv.PixelGame)

SEED_BASE = 4_000_000
H_THRESH = (120, 90, 60, 45, 30, 20, 15, 8)


from msclear.mazedist import DotDistance  # noqa: E402  (maze path distance, BFS on the screen walk mask)


def spare_lives(ram):
    return int(ram[123]) & 7


def dot_centres(ale):
    """(K, 2) screen centres (x, y) of the remaining dots (+pellets) on the current board, from the RGB screen."""
    import cv2
    rgb = ale.getScreenRGB()[:172]
    pk = rgb[::4, ::4].astype(np.int32); pk = (pk[..., 0] << 16) | (pk[..., 1] << 8) | pk[..., 2]
    v, cnt = np.unique(pk[pk != (28 << 8 | 136)], return_counts=True)
    c = int(v[np.argmax(cnt)]); col = np.array([c >> 16, (c >> 8) & 255, c & 255], np.uint8)
    m = (rgb == col).all(-1).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    small = np.flatnonzero((st[:, cv2.CC_STAT_AREA] <= 8) & (np.arange(n) > 0))
    return st[small][:, :2] + st[small][:, 2:4] / 2.0


def dist_to(cent, px, py):
    if cent is None or not len(cent):
        return None
    return float(np.min(np.abs(cent[:, 0] - (px - 8.3)) + np.abs(cent[:, 1] - (py + 5.8))))


def nearest_dot_dist(ale, px, py):
    """Screen Manhattan distance from the player (RAM x/y -> screen) to the nearest dot pixel; None if no dots."""
    import cv2
    rgb = ale.getScreenRGB()[:172]
    pk = rgb[::4, ::4].astype(np.int32); pk = (pk[..., 0] << 16) | (pk[..., 1] << 8) | pk[..., 2]
    v, cnt = np.unique(pk[pk != (28 << 8 | 136)], return_counts=True)           # most common non-background colour
    c = int(v[np.argmax(cnt)]); col = np.array([c >> 16, (c >> 8) & 255, c & 255], np.uint8)
    m = (rgb == col).all(-1).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    small = np.flatnonzero((st[:, cv2.CC_STAT_AREA] <= 8) & (np.arange(n) > 0))
    if not len(small):
        return None
    c = st[small][:, :2] + st[small][:, 2:4] / 2.0          # component centres (x, y)
    sx, sy = px - 8.3, py + 5.8
    return float(np.min(np.abs(c[:, 0] - sx) + np.abs(c[:, 1] - sy)))


def bucket(left):
    return 'early' if left > 60 else ('late' if left > 30 else 'vlate')


def _book(g):
    return dict(score=g.score, det=dict(vars(g.det)), prev_lives=g.prev_lives, h=g.h, deaths=g.deaths,
                steps=g.steps, max_eaten=g.max_eaten)


def restore(g, s):
    b = s["book"]
    g.ale.restoreState(s["state"])
    g.score, g.prev_lives, g.h, g.deaths, g.steps, g.max_eaten = (b["score"], b["prev_lives"], b["h"], b["deaths"],
                                                                 b["steps"], b["max_eaten"])
    g.det.__dict__.update(b["det"]); g.ram = g.ale.getRAM().copy(); g.stack = s["stack"].copy()
    g.cleared = False; g.done = False


def _worker(remote, n, wid, cfg):
    from msclear import ram as R
    from msclear.pixenv import PixelGame
    rng = np.random.default_rng(cfg["seed_off"] * 7919 + wid)
    b2 = [pickle.load(open(f, "rb")) for f in cfg["b2_files"]]
    b3 = [pickle.load(open(f, "rb")) for f in cfg["b3_files"]]
    pools = {k: collections.deque(maxlen=cfg["pool_cap"]) for k in ("early", "late", "vlate")}
    slot = {k: 0 for k in pools}
    b2h = collections.deque(maxlen=cfg.get("b2h_cap", 400)); b2dir = cfg.get("b2_dir"); b2n = [0]   # M17 board-2 harvest
    cur = cfg.get("cur_files") or []
    lf = list(cfg.get("late_files", []))
    for j in rng.permutation(len(lf)):                 # per-worker shuffle: each FIFO keeps a different subset
        f = lf[j]; d = pickle.load(open(f, "rb")); pools[bucket(d["left"])].append(d)
    hdir = cfg.get("harvest_dir")
    games = [PixelGame(sticky=0.25, max_steps=10**9) for _ in range(n)]
    meta = [None] * n; epn = [0]; rw = cfg["rw"]; stall = cfg["stall"]

    def snap(g):
        return dict(state=g.ale.cloneState(), stack=g.stack.copy(), book=_book(g), board=R.board(g.ram),
                    left=R.items_total(R.board(g.ram)) - int(g.ram[R.DOTS_EATEN]), lives=int(g.ram[R.LIVES]))

    def harvest(g, s=None):
        s = s or snap(g); k = bucket(s["left"]); pools[k].append(s)
        if hdir:
            os.makedirs(os.path.join(hdir, k), exist_ok=True)
            p = os.path.join(hdir, k, f"w{wid:02d}_{slot[k]:03d}.pkl"); slot[k] = (slot[k] + 1) % cfg["pool_cap"]
            with open(p + ".tmp", "wb") as f:
                pickle.dump(s, f)
            os.replace(p + ".tmp", p)

    def new(i):
        g = games[i]
        seed = SEED_BASE + cfg["seed_off"] * 1000 + wid * 100_000 + epn[0]; epn[0] += 1
        u = rng.random(); start = "b1"
        if u < cfg["p_b1"] or not hasattr(g, "det"):
            g.reset(seed)                     # 280 intro frames; snapshot starts skip it (restoreState is enough)
        if u >= cfg["p_b1"]:
            if u < cfg["p_b1"] + cfg["p_b2"] and b2:
                if b2h and rng.random() < cfg.get("p_b2h", 0.0):
                    restore(g, b2h[int(rng.integers(len(b2h)))]); start = "b2h"       # M17: mid-board-2 harvested state
                else:
                    restore(g, b2[int(rng.integers(len(b2)))]); start = "b2"
            elif cur and rng.random() < cfg.get("p_cur", 0.0):          # M16 reverse curriculum (lazy-loaded files)
                try:
                    lvl = int(open(cfg["level_file"]).read().strip())
                except Exception:
                    lvl = 0
                k = lvl if (lvl == 0 or rng.random() >= cfg.get("p_replay", 0.25)) else int(rng.integers(0, lvl))
                restore(g, pickle.load(open(cur[k][int(rng.integers(len(cur[k])))], "rb"))); start = f"b3cur{k}"
            elif (pools["late"] or pools["vlate"]) and rng.random() < cfg["p_late"]:
                k = "vlate" if pools["vlate"] and (not pools["late"] or rng.random() < cfg["p_vlate"]) else "late"
                restore(g, pools[k][int(rng.integers(len(pools[k])))]); start = "b3" + k
            elif pools["early"] and (not b3 or rng.random() < cfg["p_harvest"]):
                restore(g, pools["early"][int(rng.integers(len(pools["early"])))]); start = "b3h"
            elif b3:
                restore(g, b3[int(rng.integers(len(b3)))]); start = "b3"
        xl = 0
        if start.startswith("b3") and cfg.get("p_extra_lives", 0.0) and rng.random() < cfg["p_extra_lives"]:
            lv = int(g.ale.getRAM()[R.LIVES]); xl = int(rng.integers(1, 3))      # M15 training only: 1-2 spare lives
            if (lv & 7) < xl:
                g.ale.setRAM(R.LIVES, (lv & ~7) | xl); g.ram = g.ale.getRAM().copy(); g.prev_lives = int(g.ram[R.LIVES])
        b = R.board(g.ram)
        meta[i] = dict(start=start, xl=xl, b0=b, cur=b, steps0=g.steps, score0=g.score, since=0,
                       thr=set(), e=int(g.ram[R.DOTS_EATEN]), items=0, deaths=0, bd=0, bs=g.steps,
                       ring=collections.deque(maxlen=13), left0=R.items_total(b) - int(g.ram[R.DOTS_EATEN]),
                       lives_b3=spare_lives(g.ram) if b >= 3 else None, arrived_b3=False, phi=None)
        return g.stack.copy()

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                st = np.stack([new(i) for i in range(n)])
                remote.send((st, np.array([meta[i]["cur"] for i in range(n)], np.int8)))
            elif cmd == "pool":
                remote.send({k: len(v) for k, v in pools.items()})
            elif cmd == "step":
                frames = np.empty((n, 84, 84), np.uint8); rew = np.zeros(n, np.float32); done = np.zeros(n, np.uint8)
                boards = np.zeros(n, np.int8); comp = np.zeros((n, 6), np.float32); results = []; starts = {}
                for i, (g, a) in enumerate(zip(games, data)):
                    m = meta[i]; sc0 = g.score
                    if R.board(g.ram) >= 3 and g.steps % 5 == 0:
                        m["ring"].append((g.steps, snap(g)))
                    _, _, _, info = g.step(int(a))
                    r = g.ram; b = R.board(r); e = int(r[R.DOTS_EATEN])
                    clear = 0.0; cpen = 0.0
                    if b == 2 and cfg.get("b2_every") and not info["lost_life"] and (g.steps - m["steps0"]) % cfg["b2_every"] == 0:
                        s2 = snap(g); b2h.append(s2)
                        if b2dir and b2n[0] < cfg.get("b2_max", 10**9):
                            os.makedirs(b2dir, exist_ok=True); p2 = os.path.join(b2dir, f"w{wid:02d}_{b2n[0]:05d}.pkl"); b2n[0] += 1
                            with open(p2 + ".tmp", "wb") as f:
                                pickle.dump(s2, f)
                            os.replace(p2 + ".tmp", p2)
                    if b != m["cur"]:
                        if b > m["cur"]:
                            cpen = cfg.get("clear_life_pen", 0.0) * m["bd"]          # M17: lives lost on the cleared board
                            if m["bd"] == 0 and cfg.get("clean_bonus"):              # M18: clean clear + speed bonus
                                cpen -= cfg["clean_bonus"] + cfg.get("speed_bonus", 0.0) * max(0.0, 1.0 - (g.steps - m["bs"]) / cfg.get("speed_T", 3000))
                        m["bd"] = 0; m["bs"] = g.steps
                        clear = float(b > m["cur"]); items = R.items_total(m["cur"]) - m["e"] if b > m["cur"] else 0
                        rem0 = R.items_total(m["cur"]) - m["e"]; m["cur"] = b; m["thr"] = set()
                        if b >= 3:
                            harvest(g)
                    else:
                        items = max(0, e - m["e"]); rem0 = R.items_total(b) - m["e"]
                    m["e"] = e
                    bonus = sum(rw[4] * (1 - (rem0 - j) / 30) for j in range(items) if rem0 - j < 30)
                    over = bool(g.ale.game_over()); trunc = g.steps - m["steps0"] >= cfg["max_steps"]
                    if cfg.get("end_board") and b >= cfg["end_board"]:
                        trunc = True                                           # M17: episode ends on reaching end_board
                    oncur = m["start"].startswith("b3cur")                     # M16 curriculum start
                    if oncur and info["lost_life"] and cfg.get("cur_end_on_death"):
                        over = True                                            # curriculum: episode ends on first death
                    if info["lost_life"] and cfg.get("end_on_death"):
                        over = True                                            # M18: training episodes end at first death
                    death = float(info["lost_life"] or over); m["bd"] += int(death)
                    extra = 0.0
                    if b >= 3 and m["lives_b3"] is None:                      # first arrival at board 3
                        m["lives_b3"] = spare_lives(r); m["arrived_b3"] = True
                        extra += cfg.get("arrive_bonus", 0.0) * m["lives_b3"]
                    sk = cfg.get("shape_k", 0.0)
                    if sk and b >= cfg.get("shape_min_board", 3) and R.items_total(b) - e <= cfg.get("shape_left", 60):
                        if clear or info["lost_life"]:
                            m["phi"] = None
                        if "dd" not in m:
                            m["dd"] = DotDistance()
                        # BFS field recomputed only when the dot set changes (item eaten / board / restore)
                        d = m["dd"](g.ale, r, (b, e, id(m)))
                        phi = None if d is None else -sk * d
                        if phi is not None and m["phi"] is not None and not (clear or info["lost_life"]):
                            extra += cfg.get("shape_gamma", 0.995) * phi - m["phi"]       # F = gamma Phi(s') - Phi(s)
                        m["phi"] = phi
                    else:
                        m["phi"] = None
                    dpen = cfg.get("rw_death12", rw[3]) if b <= 2 else (cfg.get("cur_death", rw[3]) if oncur else cfg.get("rw_death3", rw[3]))
                    left = R.items_total(b) - e
                    m["since"] = 0 if items else m["since"] + 1
                    sp = 0.0
                    if stall and left <= stall[0] and m["since"] > stall[1]:
                        sp = min(stall[3], stall[2] + stall[2] * (m["since"] - stall[1]) / stall[1])
                    cm = cfg.get("cur_item_mult", 1.0) if oncur else 1.0; cc = cfg.get("cur_clear_mult", 1.0) if oncur else 1.0
                    rew[i] = rw[0] * cm * (items + bonus) + rw[1] * (g.score - sc0) + rw[2] * cc * clear - dpen * death - sp + extra - cpen - cfg.get("step_cost", 0.0)
                    comp[i] = (items, g.score - sc0, clear, death, bonus, sp)
                    m["items"] += items; m["deaths"] += int(death)
                    if b >= 3 and info["lost_life"] and left <= 60:          # pre-death state, 40-60 decisions back
                        cand = [s_ for t_, s_ in m["ring"] if 40 <= g.steps - t_ <= 60 and s_["board"] == b]
                        if cand:
                            harvest(g, cand[0])
                    if b >= 3 and not (over or trunc):
                        for th in H_THRESH:
                            if left <= th and th not in m["thr"]:
                                m["thr"].update(t for t in H_THRESH if t >= th); harvest(g); break
                    if over or trunc:
                        results.append(dict(start=m["start"], b0=m["b0"], left0=m["left0"], lives_b3=m["lives_b3"] if m["arrived_b3"] else None, board=b, clears=b - m["b0"], items=m["items"],
                                            score=g.score - m["score0"], steps=g.steps - m["steps0"], deaths=m["deaths"],
                                            truncated=bool(trunc and not over), left_end=left))
                        starts[i] = new(i); frames[i] = starts[i][-1]; done[i] = 1; boards[i] = meta[i]["cur"]
                    else:
                        frames[i] = g.frame; boards[i] = b
                remote.send((frames, rew, done, comp, boards, results, starts))
            elif cmd == "close":
                break
    except (KeyboardInterrupt, EOFError):
        pass


class VecGames13:
    """Workers in G groups (for env/GPU overlap): step_send(g, actions) / step_recv(g)."""

    def __init__(self, n_workers, per_worker, groups=2, **cfg):
        ctx = mp.get_context("spawn"); self.per = per_worker; self.nw = n_workers; self.n = n_workers * per_worker
        main = sys.modules["__main__"]
        saved = {k: main.__dict__.pop(k) for k in ("__file__", "__spec__") if k in main.__dict__}
        main.__spec__ = None
        self.remotes, self.procs = [], []
        try:
            for w in range(n_workers):
                a, b = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(b, per_worker, w, cfg), daemon=True)
                p.start(); b.close(); self.remotes.append(a); self.procs.append(p)
        finally:
            del main.__spec__; main.__dict__.update(saved)
        self.gw = [list(range(g, n_workers, groups)) for g in range(groups)]             # workers per group
        self.genv = [np.concatenate([np.arange(w * per_worker, (w + 1) * per_worker) for w in ws]) for ws in self.gw]

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        out = [r.recv() for r in self.remotes]
        return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])

    def step_send(self, g, actions):
        """actions for the envs of group g, in self.genv[g] order."""
        k = self.per
        for j, w in enumerate(self.gw[g]):
            self.remotes[w].send(("step", actions[j * k:(j + 1) * k]))

    def step_recv(self, g):
        k = self.per; out = [self.remotes[w].recv() for w in self.gw[g]]
        starts = {j * k + i: s for j, o in enumerate(out) for i, s in o[6].items()}       # index within the group
        cat = lambda q: np.concatenate([o[q] for o in out])  # noqa: E731
        return cat(0), cat(1), cat(2), cat(3), cat(4), [x for o in out for x in o[5]], starts

    def pool_size(self):
        for r in self.remotes:
            r.send(("pool", None))
        out = {}
        for r in self.remotes:
            for k, v in r.recv().items():
                out[k] = out.get(k, 0) + v
        return out

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


class SnapGame:
    """fasteval-protocol game that starts from snapshot file M11_SNAPS[seed % len] (eval c). Plays until game over
    or MB_CAP. result: start board, board reached, items on the start board, cleared_start_board, deaths."""

    def __init__(self, sticky=0.25, max_steps=None):
        self.g = _PG(sticky=sticky, max_steps=10**9); self.env = self.g.env
        self.files = os.environ["M11_SNAPS"].split(","); self.cap = int(os.environ.get("MB_CAP", "8000"))

    def reset(self, seed):
        from msclear import ram as R
        g = self.g; g.reset(seed); s = pickle.load(open(self.files[seed % len(self.files)], "rb")); restore(g, s)
        self.b0 = R.board(g.ram); self.e0 = int(g.ram[R.DOTS_EATEN]); self.items = 0; self.n = 0; self.d = 0
        self.score0 = g.score; self.done = False; self.cur = self.b0; self.file = os.path.basename(self.files[seed % len(self.files)])
        return g.stack

    @property
    def frame(self):
        return self.g.frame

    def step(self, a):
        from msclear import ram as R
        g = self.g; _, _, _, info = g.step(a); self.n += 1; self.d += int(info["lost_life"])
        b = R.board(g.ram)
        if b == self.b0:
            self.items = max(self.items, int(g.ram[R.DOTS_EATEN]) - self.e0)
        elif self.cur == self.b0:
            self.items = R.items_total(self.b0) - self.e0
        self.cur = max(self.cur, b)
        over = g.ale.game_over()
        self.d += int(over and not info["lost_life"])
        self.done = over or self.n >= self.cap or (os.environ.get("SNAP_STOP_CLEAR") == "1" and self.cur > self.b0)   # M17

    def result(self, seed):
        from msclear import ram as R
        return dict(seed=seed, file=self.file, b0=self.b0, board=self.cur, cleared=self.cur > self.b0, items=self.items,
                    left0=R.items_total(self.b0) - self.e0, steps=self.n, deaths=self.d, score=self.g.score - self.score0, eaten=self.items,
                    cap_hit=bool(self.n >= self.cap and not self.g.ale.game_over()))


def snap_worker(remote):
    from msclear import fasteval, pixenv
    pixenv.PixelGame = SnapGame
    fasteval._worker(remote)
