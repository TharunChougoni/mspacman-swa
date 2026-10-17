"""M18: clean-clear PPO on boards 1-2 (init M17 best). Starts P_B1 fresh board 1 / rest board 2 (P_B2H of those from the
online mid-board-2 FIFO, else multiboard board-2 arrival snapshots). Training episodes END AT FIRST DEATH and on reaching
board 3. Reward: items RW[0] + score RW[1] - STEP_COST per decision + on clear: RW[2] (any clear) + CLEAN_BONUS +
SPEED_BONUS*max(0, 1 - board_decisions/SPEED_T) if no death on that board, - RW_DEATH12 per death (high), stall penalty.
Reward ordering (fast clean > slow clean > scrappy > death) is checked and logged at start. Evals (fp32 greedy, true lives,
stop at the board clear): board-1 set = fresh resets of seeds 0..N_B1-1 (snapshots in RUN/b1_test), board-2 set = M17
runs/m17/b2_test (200). Main metric: no-death clear rate on both (clean). best.pt = max mean clean. Every EVAL_EVERY
updates; ckpt every CKPT_S s; metrics.jsonl + tensorboard scalars in RUN/tb. DEADLINE (epoch s) / STOP file.

M17 docstring follows.
M17: board-2 focused PPO. Starts: P_B1 fresh board 1, rest board 2 (P_B2H of those from a FIFO of mid-board-2 states
harvested online every B2_EVERY decisions, else board-2 arrival snapshots of the M9/M10 multiboard train seeds). Episodes end
on reaching board 3 (END_BOARD). Reward: items + score + clear bonus RW[2] - RW_DEATH12 per death on boards 1-2
- CLEAR_LIFE_PEN per life lost on the board just cleared, stall penalty. KL leash to INIT on board-1 steps, entropy floor.
No curriculum / aux CE / board-3 training. Evals (fp32 greedy): board-1 clear seeds 0-99; board-2 clear on a FIXED held-out
set of 200 board-2 snapshots (runs/m17/b2_test, from the M11 multiboard seeds, never trained on); multiboard EVAL_MB_N games
(reach3, spare lives at board 3). best.pt = max b2_clear with b1 >= MIN_B1. GEN_TEST=1 builds the test set; EVAL_ONLY=ckpt
evaluates one checkpoint. Resumes from RUN/ckpt_last.pt; STOP file / DEADLINE.

M16 docstring follows.
M16: M15 + reverse curriculum on upper-centre maze-2 states (runs/m16/buckets.json from scripts/m16_states.py):
P_CUR of maze-2 starts from bucket `level` (P_REPLAY from lower buckets), promote when train clear >= PROMOTE over
>= PROMOTE_N episodes. Init runs/m15/ckpt_last.pt. Eval adds per-bucket clear on held-out bucket states (true lives).

M15 (env defaults below): blue-maze-heavy PPO from M13 best: start mix 25% b1 / 15% b2 / 60% maze-2 snapshots
(harvest buckets spread over dots-left), RW_DEATH12 20, ARRIVE_BONUS 5/spare life, P_EXTRA_LIVES of maze-2 starts get 1-2
spare lives (training only), shaping on all of maze 2, ENT 0.003 x4 when running entropy < ENT_FLOOR, auxiliary CE to
search-teacher actions (demos + DAgger r1, runs/m14d1/data) annealed AUX0 -> 0 over AUX_UPD updates.

M13: M12 continued with lives banking + dot-direction shaping (msclear.m13_env).
Init: policy from INIT (runs/m12/best.pt), value head from INIT_VH (runs/m12/ckpt_last.pt), fresh optimizer;
KL leash on maze-1 steps toward frozen INIT policy (M12 best). New: RW_DEATH12 (death penalty on boards 1-2),
ARRIVE_BONUS per spare life on entering board 3, SHAPE_K potential shaping toward the nearest dot on boards >= 3 with
<= SHAPE_LEFT items left; metrics lives_b3 (train arrivals) and mb_lives_b3 (multiboard eval).

M12 docstring follows.
M12: M11 continued (runs/m11/ckpt_last.pt: policy + value head, fresh optimizer) with a late-board curriculum on
maze 2 (msclear.m12_env: per-bucket FIFOs early/late/vlate, P_LATE of board-3 starts from <= 60 left, pre-death
snapshots), KL leash on maze-1 steps toward frozen M11, start mix 20/25/55, extra eval (d): clear rate from
HELD-OUT late board-3 snapshots (runs/m12/late_test, harvested from the 7 test seeds). best.pt key:
(multiboard clears, late-snapshot clear, snap3 clear) with b1 clear >= MIN_B1.

M11 docstring follows.
M11: multi-board PPO of the M10 KV-cache net (runs/m10/best.pt), games playing THROUGH board clears.

Envs: msclear.m11_env.VecGames11 (ENV_WORKERS x PER_WORKER, two worker groups so the env step of one half overlaps
the GPU step of the other). Start mix P_B1 fresh / P_B2 board-2 snapshots / rest board-3 snapshots (train split of
the M9+M10 multiboard board-3 states + a harvested FIFO pool of board>=3 states, P_HARVEST share once non-empty).
Reward (m11_env): items (maze-agnostic ram[119]) + endgame bonus, score, clear bonus, death penalty, stall penalty.
Loss: PPO clip + value (separate ValueHead on h, VWARM value-only updates) - entropy + beta * KL(pi || frozen M10)
on samples whose observation is on boards 1-2 (maze 1) only; no KL on boards >= 3 (maze 2).
Training sample = 15-token history clamped at the episode start (exact streaming semantics of M10Net).
Evals every EVAL_EVERY updates, strict fp32 greedy: (a) board-1 clear seeds 0-99, (b) multiboard EVAL_MB_N games
seeds 60000+ (mean boards cleared, board-3 items / clear), (c) held-out board-3 snapshots (test seeds) x EVAL_SNAP_REP.
best.pt = max (b) mean clears with (a) >= MIN_B1 (tie: (c) clear). DEADLINE env (epoch s) / STOP file -> clean exit
(STOP: exit 3). Resumes from RUN/ckpt_last.pt.
"""
from __future__ import annotations

import copy, glob, json, os, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msclear import fasteval, multiboard, m11_env, m13_env  # noqa: E402
from msclear.m10_fasteval import M10Evaluator  # noqa: E402
from msclear.m10_net import M10Net  # noqa: E402

E = os.environ.get
RUN = Path(E("RUN", "runs/m18")); INIT = E("INIT", "runs/m17/best.pt"); INIT_VH = E("INIT_VH", "none")
RW_DEATH12, ARRIVE_BONUS = float(E("RW_DEATH12", "50.0")), float(E("ARRIVE_BONUS", "0.0"))
CLEAR_LIFE_PEN, END_BOARD = float(E("CLEAR_LIFE_PEN", "0.0")), int(E("END_BOARD", "3"))
STEP_COST, CLEAN_BONUS, SPEED_BONUS, SPEED_T = float(E("STEP_COST", "0.002")), float(E("CLEAN_BONUS", "50")), float(E("SPEED_BONUS", "20")), float(E("SPEED_T", "3000"))
N_B1 = int(E("N_B1", "100"))
P_B2H, B2_EVERY, B2H_CAP = float(E("P_B2H", "0.5")), int(E("B2_EVERY", "40")), int(E("B2H_CAP", "400"))
SHAPE_K, SHAPE_LEFT = float(E("SHAPE_K", "0.005")), int(E("SHAPE_LEFT", "154"))
NW, PER, T = int(E("ENV_WORKERS", "18")), int(E("PER_WORKER", "6")), int(E("T", "128"))
EPOCHS = int(E("EPOCHS", "2"))
NMB = int(E("MINIBATCHES", "0")) or -(-int(E("ENV_WORKERS", "18")) * int(E("PER_WORKER", "6")) // 12)   # ~12 envs (1536 samples) per minibatch: 6 GB GPU
GAMMA, LAM, CLIP = float(E("GAMMA", "0.995")), float(E("GAE_LAMBDA", "0.95")), float(E("CLIP", "0.1"))
LR_PI, LR_V = float(E("LR_PI", "5e-5")), float(E("LR_V", "3e-4"))
VF, ENT, BETA, TAU = float(E("VF", "0.5")), float(E("ENT", "0.003")), float(E("BETA", "0.05")), float(E("TAU", "0.25"))
VWARM, MAX_GRAD = int(E("VWARM", "10")), float(E("MAX_GRAD", "0.5"))
RW = [float(x) for x in E("RW", "0.1,0.0005,15.0,50.0,2.0").split(",")]
STALL = [float(x) for x in E("STALL", "10,150,0.005,0.01").split(",")]
P_B1, P_B2, P_HARVEST = float(E("P_B1", "0.50")), float(E("P_B2", "0.50")), float(E("P_HARVEST", "0.6"))
P_LATE, P_VLATE, EVAL_LATE_REP = float(E("P_LATE", "0.4")), float(E("P_VLATE", "0.4")), int(E("EVAL_LATE_REP", "2"))
POOL_CAP, MAX_STEPS = int(E("POOL_CAP", "60")), int(E("MAX_STEPS", "8000"))
EVAL_EVERY, EVAL_MB_N, EVAL_SNAP_REP = int(E("EVAL_EVERY", "150")), int(E("EVAL_MB_N", "0")), int(E("EVAL_SNAP_REP", "1"))
EVAL_WORKERS, MIN_B1 = int(E("EVAL_WORKERS", "8")), float(E("MIN_B1", "0.75"))
N_TEST = int(E("N_TEST", "7")); CKPT_S = float(E("CKPT_S", "120"))
P_EXTRA_LIVES, ENT_FLOOR = float(E("P_EXTRA_LIVES", "0.5")), float(E("ENT_FLOOR", "0.25"))
AUX0, AUX_UPD, B_AUX, AUX_DATA = float(E("AUX0", "0.0")), int(E("AUX_UPD", "1000000000")), int(E("B_AUX", "32")), E("AUX_DATA", "runs/m16/ce_data")  # M16: teacher CE on curriculum states, no anneal
P_CUR, P_REPLAY, PROMOTE, PROMOTE_N = float(E("P_CUR", "0.0")), float(E("P_REPLAY", "0.25")), float(E("PROMOTE", "0.5")), int(E("PROMOTE_N", "40"))
EVAL_CUR_N = int(E("EVAL_CUR_N", "12")); CUR_ITEM_MULT, CUR_CLEAR_MULT = float(E("CUR_ITEM_MULT", "5")), float(E("CUR_CLEAR_MULT", "3")); RW_DEATH3, CUR_DEATH, CUR_END_ON_DEATH = float(E("RW_DEATH3", "10")), float(E("CUR_DEATH", "15")), int(E("CUR_END_ON_DEATH", "1"))
DEADLINE = float(E("DEADLINE", "0")) or float("inf"); MAX_UPDATES = int(E("MAX_UPDATES", "1000000"))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def jwrite(row):
    with open(RUN / "metrics.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def save(obj, p):
    torch.save(obj, str(p) + ".tmp"); os.replace(str(p) + ".tmp", p)


class ValueHead(nn.Module):
    def __init__(self, d):
        super().__init__(); self.f = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, h):
        return self.f(h)[:, 0]


def split_b3():
    """Board-3 states of the M9 and M10 multiboard evals; N_TEST seeds held out (fixed rng 0)."""
    files = sorted(glob.glob("runs/multiboard/states/*_board3.pkl") + glob.glob("runs/multiboard_m10/states/*_board3.pkl"))
    seeds = sorted({int(Path(f).name.split("_")[0][4:]) for f in files})
    test = set(np.random.default_rng(0).choice(seeds, N_TEST, replace=False).tolist())
    return ([f for f in files if int(Path(f).name.split("_")[0][4:]) not in test],
            [f for f in files if int(Path(f).name.split("_")[0][4:]) in test], sorted(test))


def make_b1_set():
    """Board-1 eval set: fresh-reset snapshots of seeds 0..N_B1-1 (built once, kept in RUN/b1_test)."""
    import pickle
    d = RUN / "b1_test"; d.mkdir(parents=True, exist_ok=True)
    if len(list(d.glob("*.pkl"))) < N_B1:
        from msclear.pixenv import PixelGame
        from msclear import ram as Rm
        for sd in range(N_B1):
            g = PixelGame(sticky=0.25, max_steps=10**9); g.reset(sd)
            s = dict(state=g.ale.cloneState(), stack=g.stack.copy(), book=m11_env._book(g), board=Rm.board(g.ram),
                     left=Rm.items_total(Rm.board(g.ram)) - int(g.ram[Rm.DOTS_EATEN]), lives=int(g.ram[Rm.LIVES]))
            pickle.dump(s, open(d / f"b1_seed{sd:03d}.pkl", "wb"))
    return sorted(str(p) for p in d.glob("*.pkl"))[:N_B1]


def reward_check():
    """Episode return (undiscounted) of stylised board episodes under the current reward; must be ordered."""
    items = 154 * RW[0]; sc = 2000 * RW[1]
    def ret(steps, deaths, clear, frac=1.0):
        r = frac * items + frac * sc - STEP_COST * steps - RW_DEATH12 * deaths
        if clear:
            r += RW[2] + (CLEAN_BONUS + SPEED_BONUS * max(0.0, 1 - steps / SPEED_T) if deaths == 0 else -CLEAR_LIFE_PEN * deaths)
        return r
    rows = [("fast clean clear (1200 dec)", ret(1200, 0, True)), ("slow clean clear (2800 dec)", ret(2800, 0, True)),
            ("very slow clean clear (5000 dec)", ret(5000, 0, True)), ("scrappy clear, 1 death (eval only)", ret(1800, 1, True)),
            ("death at 90% eaten (training ends)", ret(1500, 1, False, 0.9)), ("death at 30% eaten", ret(500, 1, False, 0.3))]
    for n, v in rows:
        log(f"REWARD CHECK {n:38s} {v:8.2f}")
    v = [x for _, x in rows]
    ok = v[0] > v[1] > v[2] > v[3] > v[4] and v[3] > v[5]
    log(f"REWARD CHECK ordering fast-clean > slow-clean > very-slow-clean > scrappy > death: {'OK' if ok else 'VIOLATED'}")
    if not ok and E("ALLOW_BAD_ORDER") != "1":
        raise SystemExit("reward ordering violated")


def main():
    RUN.mkdir(parents=True, exist_ok=True); (RUN / "harvest").mkdir(exist_ok=True); dev = torch.device(E("DEVICE", "cuda"))
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    torch.set_num_threads(2)
    ck0 = torch.load(INIT, map_location="cpu", weights_only=False); ARCH = ck0["arch"]
    pi = M10Net(**ARCH); pi.load_state_dict(ck0["pi"] if "pi" in ck0 else ck0["model"]); pi = pi.to(dev)
    ref = copy.deepcopy(pi).eval().requires_grad_(False)                   # frozen M11 (maze-1 KL leash)
    vh = ValueHead(ARCH["d"]).to(dev); C = pi.ctx; N = NW * PER
    if "vh" in ck0:
        vh.load_state_dict(ck0["vh"])
    elif INIT_VH and Path(INIT_VH).exists():
        vh.load_state_dict(torch.load(INIT_VH, map_location="cpu", weights_only=False)["vh"])
    SEPV = int(E("SEPV", "0"))                                              # M22: separate value network (own CNN+trunk, copy of INIT)
    vnet = copy.deepcopy(pi).train() if SEPV else None
    if SEPV and E("VNET_INIT", "vh") == "vh" and INIT_VH and Path(INIT_VH).exists():   # trunk the INIT_VH value head was trained with
        vnet.load_state_dict(torch.load(INIT_VH, map_location="cpu", weights_only=False)["pi"]); log(f"SEPV: value net trunk from {INIT_VH}")
    EMA = float(E("EMA", "0"))                                              # M20: evaluate/save an EMA of the policy weights
    ema = copy.deepcopy(pi).eval().requires_grad_(False) if EMA else None
    groups = [dict(params=pi.parameters(), lr=LR_PI), dict(params=vh.parameters(), lr=LR_V)]
    if SEPV:
        groups.append(dict(params=vnet.parameters(), lr=float(E("LR_VNET", "1e-4"))))
    opt = torch.optim.Adam(groups, eps=1e-5)
    st = dict(update=int(E("START_UPDATE", "0")), env_steps=int(E("START_STEPS", "0")), best=None, evals=[], wall=0.0, resumes=0, init_update=ck0.get("state", {}).get("update"))   # M18 s2: continue numbering from INIT
    last = RUN / "ckpt_last.pt"
    if last.exists():
        c = torch.load(last, map_location=dev, weights_only=False)
        pi.load_state_dict(c["pi"]); vh.load_state_dict(c["vh"]); opt.load_state_dict(c["opt"]); st = c["state"]
        if EMA and c.get("ema"):
            ema.load_state_dict(c["ema"])
        if SEPV and c.get("vnet"):
            vnet.load_state_dict(c["vnet"])
        st["resumes"] += 1; log(f"resumed {last} at update {st['update']} ({st['env_steps']:,} steps)")
    BK = json.load(open("runs/m16/buckets.json")); NB = len(BK["train"])
    st.setdefault("level", 0); (RUN / "level").write_text(str(st["level"]))
    b3_train, b3_test, test_seeds = split_b3()
    b2 = sorted(glob.glob(E("B2_GLOB", "runs/multiboard/states/*_board2.pkl")) + glob.glob(E("B2_GLOB2", "runs/multiboard_m10/states/*_board2.pkl")))
    b2_test = sorted(glob.glob(E("B2_TEST", "runs/m17/b2_test") + "/*.pkl"))[:int(E("N_B2", "200"))]
    import pickle
    late_files = []; late_test = b2_test
    b1_test = make_b1_set(); reward_check()
    cfg = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, list))}
    cfg.update(ARCH=ARCH, N=N, b3_train=len(b3_train), b3_test=len(b3_test), test_seeds=test_seeds, b2=len(b2),
               late_files=len(late_files), late_test=len(late_test), init_update=st["init_update"])
    (RUN / "config.json").write_text(json.dumps(cfg, indent=1)); log(f"config {json.dumps(cfg)}")
    envs = m13_env.VecGames13(NW, PER, groups=2, rw=RW, rw_death12=RW_DEATH12, arrive_bonus=ARRIVE_BONUS,
                              shape_k=SHAPE_K, shape_left=SHAPE_LEFT, shape_min_board=int(E("SHAPE_MIN_BOARD", "3")), shape_skip_eat=int(E("SHAPE_SKIP_EAT", "0")), shape_gamma=GAMMA, stall=STALL, p_b1=P_B1, p_b2=P_B2, p_harvest=P_HARVEST,
                              p_late=0.0, p_vlate=P_VLATE, late_files=late_files,
                              b2_files=b2, b3_files=[], clear_life_pen=CLEAR_LIFE_PEN, end_on_death=int(E("END_ON_DEATH", "1")), step_cost=STEP_COST, clean_bonus=CLEAN_BONUS, speed_bonus=SPEED_BONUS, speed_T=SPEED_T, end_board=END_BOARD, p_b2h=P_B2H, b2_every=B2_EVERY, b2h_cap=B2H_CAP,
                              b2_dir=E("B2_DIR") or None, b2_max=int(E("B2_MAX", "1000000000")), pool_cap=POOL_CAP, max_steps=MAX_STEPS, p_extra_lives=P_EXTRA_LIVES, cur_files=BK["train"], p_cur=P_CUR, p_replay=P_REPLAY, cur_item_mult=CUR_ITEM_MULT, cur_clear_mult=CUR_CLEAR_MULT, rw_death3=RW_DEATH3, cur_death=CUR_DEATH, cur_end_on_death=CUR_END_ON_DEATH, level_file=str((RUN / "level").resolve()),
                              harvest_dir=str((RUN / "harvest").resolve()), seed_off=st["update"] % 997)
    # evaluators: (a) board-1 fasteval, (b) multiboard, (c) held-out board-3 snapshots
    fasteval._MAX_STEPS = 20001
    ev_a = None
    os.environ["MB_CAP"] = "20000"; os.environ.pop("MB_STATES", None)
    w0 = fasteval._worker; ev_b = None
    os.environ["MB_CAP"] = "8000"; os.environ["SNAP_STOP_CLEAR"] = "1"; fasteval._worker = m13_env.snap_worker
    os.environ["M11_SNAPS"] = ",".join(late_test); ev_d = M10Evaluator(workers=EVAL_WORKERS, device=E("DEVICE", "cuda")) if late_test else None
    os.environ["M11_SNAPS"] = ",".join(b1_test); ev_f = M10Evaluator(workers=EVAL_WORKERS, device=E("DEVICE", "cuda")); fasteval._worker = w0
    try:
        from torch.utils.tensorboard import SummaryWriter
        tbw = SummaryWriter(str(RUN / "tb"))
    except Exception as ex:  # noqa: BLE001
        tbw = None; log(f"no tensorboard writer: {ex}")

    def evaluate():
        pi.eval(); t0 = time.time()
        en = ema if EMA else pi
        _, rf = ev_f.evaluate(en, list(range(len(b1_test))))
        _, rd = ev_d.evaluate(en, list(range(len(late_test))))
        pi.train()
        cl = lambda rr: float(np.mean([g["cleared"] and g["deaths"] == 0 for g in rr]))  # noqa: E731
        s = dict(update=st["update"], env_steps=st["env_steps"], secs=round(time.time() - t0),
                 b1_clean=cl(rf), b1_clear=float(np.mean([g["cleared"] for g in rf])), b1_deaths=float(np.mean([g["deaths"] for g in rf])),
                 b1_steps_clean=float(np.mean([g["steps"] for g in rf if g["cleared"] and g["deaths"] == 0] or [0])),
                 b2_clean=cl(rd), b2_clear=float(np.mean([g["cleared"] for g in rd])), b2_deaths=float(np.mean([g["deaths"] for g in rd])),
                 b2_steps_clean=float(np.mean([g["steps"] for g in rd if g["cleared"] and g["deaths"] == 0] or [0])),
                 b2_trunc=float(np.mean([bool(g.get("cap_hit")) for g in rd])))
        s["clean_mean"] = (s["b1_clean"] + s["b2_clean"]) / 2
        st["evals"].append(s); jwrite(dict(type="eval", **s))
        if tbw:
            for k in ("b1_clean", "b1_clear", "b1_deaths", "b2_clean", "b2_clear", "b2_deaths", "clean_mean", "b2_steps_clean"):
                tbw.add_scalar(k, s[k], st["env_steps"])
            tbw.flush()
        better = st["best"] is None or s["clean_mean"] > st["best"]["clean_mean"]
        if better:
            st["best"] = s; save(dict(model=(ema if EMA else pi).state_dict(), arch=ARCH, kind="m10", step=st["update"]), RUN / "best.pt")
        log(f"EVAL upd {st['update']} ({st['env_steps'] / 1e6:.2f}M): CLEAN b1 {s['b1_clean']:.3f} b2 {s['b2_clean']:.3f} (mean {s['clean_mean']:.3f})  "
            f"clear b1 {s['b1_clear']:.3f} b2 {s['b2_clear']:.3f}  deaths b1 {s['b1_deaths']:.2f} b2 {s['b2_deaths']:.2f}  clean steps b1 {s['b1_steps_clean']:.0f} b2 {s['b2_steps_clean']:.0f}  b2 trunc {s['b2_trunc']:.2f}  ({s['secs']}s){'  *best*' if better else ''}")
        return s

    def ckpt():
        save(dict(pi=pi.state_dict(), vh=vh.state_dict(), opt=opt.state_dict(), arch=ARCH, state=st, ema=ema.state_dict() if EMA else None, vnet=vnet.state_dict() if SEPV else None), last)

    if AUX0 > 0:
      FRA = np.load(f"{AUX_DATA}/frames.npy", mmap_mode="r"); mdA = np.load(f"{AUX_DATA}/demo_meta.npz")
      labA = mdA["lab"] & ~mdA["val_games"][mdA["gid"]]; ixA = np.flatnonzero(labA); arng = np.random.default_rng(st["update"] + 5)
      wA = np.where(mdA["pd"], 0.1, 1.0).astype(np.float32)

    QSOFT = E("Q_DATA", "runs/m16/qdata.npz") if int(E("QSOFT", "0")) else None     # M16: soft targets from teacher values
    if QSOFT:
        qd = np.load(QSOFT); qperm = np.random.default_rng(0).permutation(len(qd["dec"])); nqv = max(1, len(qperm) // 20)
        qval, qtr = qperm[:nqv], qperm[nqv:]                                     # 5% held-out rows for aux_ce_val

    def aux_batch(val=False):
        if QSOFT:
            r = arng.choice(qval if val else qtr, B_AUX); ix = qd["dec"][r]
            dec = np.maximum(ix[:, None] + np.arange(-C + 1, 1)[None], mdA["g0"][ix][:, None])
            fi = mdA["fbase"][dec][..., None] + np.arange(4); u, inv = np.unique(fi, return_inverse=True)
            x = np.asarray(FRA[u])[inv.reshape(fi.shape)]
            return (torch.from_numpy(x).to(dev), torch.from_numpy(qd["p"][r]).to(dev), torch.ones(B_AUX, device=dev))
        ix = arng.choice(ixA, B_AUX); dec = np.maximum(ix[:, None] + np.arange(-C + 1, 1)[None], mdA["g0"][ix][:, None])
        fi = mdA["fbase"][dec][..., None] + np.arange(4); u, inv = np.unique(fi, return_inverse=True)
        x = np.asarray(FRA[u])[inv.reshape(fi.shape)]
        return (torch.from_numpy(x).to(dev), torch.from_numpy(mdA["act"][ix].astype(np.int64)).to(dev), torch.from_numpy(wA[ix]).to(dev))
    ent_run = 1.0
    if E("EVAL_ONLY"):
        pi.load_state_dict(torch.load(E("EVAL_ONLY"), map_location=dev, weights_only=False).get("model") or torch.load(E("EVAL_ONLY"), map_location=dev, weights_only=False)["pi"])
        st["update"] = -1; s = evaluate(); envs.close(); return
    gen_n = int(E("GEN_TEST", "0"))
    s0, b0 = envs.reset()
    stack = torch.from_numpy(s0).to(dev); bobs = torch.from_numpy(b0).to(dev).long()
    new = torch.ones(N, dtype=torch.bool, device=dev); estart = torch.full((N,), C - 1, dtype=torch.long, device=dev)
    H = torch.zeros(N, C - 1 + T, 4, 84, 84, dtype=torch.uint8, device=dev)
    A = torch.zeros(N, T, dtype=torch.long, device=dev); LP = torch.zeros(N, T, device=dev); V = torch.zeros(N, T + 1, device=dev)
    R = torch.zeros(N, T, device=dev); D = torch.zeros(N, T, device=dev); ES = torch.zeros(N, T, dtype=torch.long, device=dev)
    BRD = torch.zeros(N, T, dtype=torch.long, device=dev)
    gidx = [torch.from_numpy(g).to(dev) for g in envs.genv]; caches = [pi.new_cache(len(g), dev) for g in envs.genv]
    vcaches = [pi.new_cache(len(g), dev) for g in envs.genv] if SEPV else None
    recent = []; comp_sum = np.zeros(6); first = True; t_ck = time.time()
    any_new = [True, True]; pool_sz = {}
    if not st["evals"] and E("EVAL0", "1") == "1":
        evaluate()
    while st["update"] < MAX_UPDATES:
        if time.time() >= DEADLINE:
            log("deadline reached"); break
        if (RUN / "STOP").exists():
            ckpt(); log(f"STOP file: saved ckpt_last.pt at update {st['update']}, exiting 3"); sys.exit(3)
        t0 = time.time(); pi.eval(); vnet is not None and vnet.eval()
        with torch.no_grad():
            if not first:
                H[:, :C - 1] = H[:, T:T + C - 1].clone(); estart = (estart - T).clamp(min=0)
            first = False
            for t in range(T + 1):
                for g, gi in enumerate(gidx):
                    if t > 0:                                       # results of the action sent at t-1
                        fr, rew, dn, comp, brd, res, starts = envs.step_recv(g)
                        R[gi, t - 1] = torch.from_numpy(rew).to(dev); dnt = torch.from_numpy(dn).to(dev).bool()
                        D[gi, t - 1] = dnt.float(); comp_sum += comp.sum(0); recent.extend(res)
                        sg = torch.cat([stack[gi, 1:], torch.from_numpy(fr).to(dev)[:, None]], 1)
                        for i, s_ in starts.items():
                            sg[i] = torch.from_numpy(s_).to(dev)
                        stack[gi] = sg; new[gi] = dnt; bobs[gi] = torch.from_numpy(brd).to(dev).long(); any_new[g] = bool(dn.any())
                    if t == T:
                        continue
                    pos = C - 1 + t; x = stack[gi]
                    H[gi, pos] = x
                    if any_new[g]:                                  # CPU-side flag: no GPU sync when nothing reset
                        ng = new[gi]
                        estart[gi] = torch.where(ng, torch.full_like(estart[gi], pos), estart[gi])
                    ES[gi, t] = estart[gi]
                    BRD[gi, t] = bobs[gi]
                    z = pi.embed(x)
                    if any_new[g]:
                        pi.prime(caches[g], ng, z)
                    h = pi.step(caches[g], z)
                    if SEPV:
                        zv = vnet.embed(x)
                        if any_new[g]:
                            vnet.prime(vcaches[g], ng, zv)
                        V[gi, t] = vh(vnet.step(vcaches[g], zv))
                    lg = pi.adv(h) / TAU
                    if not SEPV:
                        V[gi, t] = vh(h)
                    a = torch.distributions.Categorical(logits=lg).sample()
                    A[gi, t] = a; LP[gi, t] = F.log_softmax(lg, -1).gather(1, a[:, None])[:, 0]
                    envs.step_send(g, a.cpu().numpy())
            for g, gi in enumerate(gidx):                           # bootstrap V(obs_T) on a copy of the caches
                x = stack[gi]; z = pi.embed(x); cc = [c.clone() for c in caches[g]]; pi.prime(cc, new[gi], z)
                if SEPV:
                    zv = vnet.embed(x); cv = [c.clone() for c in vcaches[g]]; vnet.prime(cv, new[gi], zv); V[gi, T] = vh(vnet.step(cv, zv))
                else:
                    V[gi, T] = vh(pi.step(cc, z))
            # GAE
            adv = torch.zeros(N, T, device=dev); last_gae = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nonterm = 1.0 - D[:, t]
                delta = R[:, t] + GAMMA * V[:, t + 1] * nonterm - V[:, t]
                last_gae = delta + GAMMA * LAM * nonterm * last_gae; adv[:, t] = last_gae
            ret = adv + V[:, :T]
            pos = torch.arange(T, device=dev) + C - 1
            IDX = torch.maximum((pos[:, None] + torch.arange(-C + 1, 1, device=dev)[None])[None].expand(N, -1, -1), ES[:, :, None])
            # frozen-reference log-probs (maze-1 KL), batched once per update
            RLP = torch.zeros(N, T, 4, device=dev)
            for mb in torch.arange(N, device=dev).chunk(NMB * 2):
                n = len(mb); z = ref.embed(H[mb].reshape(-1, 4, 84, 84)).view(n, C - 1 + T, -1)
                ctx = z[torch.arange(n, device=dev)[:, None, None], IDX[mb]].reshape(n * T, C, -1)
                RLP[mb] = F.log_softmax(ref.adv(ref.trunk_last(ctx)) / TAU, -1).view(n, T, 4)
        t_roll = time.time() - t0; t1 = time.time(); pi.train(); vnet is not None and vnet.train()
        if gen_n:
            k = len(glob.glob(f"{E('B2_DIR')}/*.pkl")); log(f"GEN harvested {k}")
            if k >= gen_n:
                envs.close(); return
            continue
        kmask = (BRD == 1).float()                                              # M17: KL leash on board-1 steps only
        advn = (adv - adv.mean()) / (adv.std() + 1e-8)
        vwarm = st["update"] - st.setdefault("start_update", int(E("START_UPDATE", "0"))) < VWARM   # fix: warmup counted from the run start (was absolute: no warmup when START_UPDATE>=VWARM)
        stats = []
        for _ in range(EPOCHS):
            for mb in torch.randperm(N, device=dev).chunk(NMB):
                n = len(mb)
                z = pi.embed(H[mb].reshape(-1, 4, 84, 84)).view(n, C - 1 + T, -1)
                ctx = z[torch.arange(n, device=dev)[:, None, None], IDX[mb]].reshape(n * T, C, -1)
                h = pi.trunk_last(ctx)
                if SEPV:                                                    # M22: value loss never touches policy features
                    zv = vnet.embed(H[mb].reshape(-1, 4, 84, 84)).view(n, C - 1 + T, -1)
                    v = vh(vnet.trunk_last(zv[torch.arange(n, device=dev)[:, None, None], IDX[mb]].reshape(n * T, C, -1)))
                else:
                    v = vh(h.detach() if vwarm else h)
                vl = F.mse_loss(v, ret[mb].reshape(-1))
                if vwarm:
                    loss = VF * vl; pl = kl = ent = torch.zeros((), device=dev); cf = 0.0
                else:
                    lp_all = F.log_softmax(pi.adv(h) / TAU, -1)
                    lp = lp_all.gather(1, A[mb].reshape(-1, 1))[:, 0]
                    ratio = torch.exp(lp - LP[mb].reshape(-1)); am = advn[mb].reshape(-1)
                    pl = -torch.min(ratio * am, ratio.clamp(1 - CLIP, 1 + CLIP) * am).mean()
                    ent = -(lp_all.exp() * lp_all).sum(-1).mean()
                    klv = (lp_all.exp() * (lp_all - RLP[mb].reshape(-1, 4))).sum(-1)
                    km = kmask[mb].reshape(-1); kl = (klv * km).sum() / km.sum().clamp(min=1)
                    aw = AUX0 * max(0.0, 1.0 - st["update"] / AUX_UPD); ace = torch.zeros((), device=dev)
                    if aw > 0:
                        xa, aa, wa = aux_batch(); za = pi.embed(xa.reshape(-1, 4, 84, 84)).view(B_AUX, C, -1)
                        lg = pi.adv(pi.trunk_last(za)) / TAU
                        ace = (-(aa * F.log_softmax(lg, -1)).sum(-1)).mean() if QSOFT else (F.cross_entropy(lg, aa, reduction="none") * wa).sum() / wa.sum()
                    ec = ENT * (4.0 if ent_run < ENT_FLOOR else 1.0)
                    loss = pl + VF * vl - ec * ent + BETA * kl + aw * ace
                    cf = ((ratio - 1).abs() > CLIP).float().mean().item()
                opt.zero_grad(set_to_none=True); loss.backward()
                if SEPV:                                                    # separate clips: value grads can't shrink the policy step
                    nn.utils.clip_grad_norm_(list(pi.parameters()), MAX_GRAD); nn.utils.clip_grad_norm_(list(vh.parameters()) + list(vnet.parameters()), MAX_GRAD)
                else:
                    nn.utils.clip_grad_norm_(list(pi.parameters()) + list(vh.parameters()), MAX_GRAD)
                opt.step()
                stats.append((pl.item(), vl.item(), kl.item(), ent.item(), cf)); ent_run = 0.9 * ent_run + 0.1 * ent.item() if not vwarm else ent_run
                aux_ce_last = float(ace) if not vwarm else None
        if QSOFT and AUX0 > 0:
            with torch.no_grad():
                xa, aa, _ = aux_batch(val=True); za = pi.embed(xa.reshape(-1, 4, 84, 84)).view(B_AUX, C, -1)
                aux_ce_val = float((-(aa * F.log_softmax(pi.adv(pi.trunk_last(za)) / TAU, -1)).sum(-1)).mean())
                aux_tgt_ent = float((-(aa * torch.log(aa.clamp(min=1e-12))).sum(-1)).mean())
        if EMA:
            with torch.no_grad():
                for pe, pp in zip(ema.parameters(), pi.parameters()):
                    pe.mul_(EMA).add_(pp.detach(), alpha=1 - EMA)
        st["update"] += 1; st["env_steps"] += N * T; st["wall"] += time.time() - t0
        pl, vl, kl, ent, cf = np.mean(stats, 0)
        recent = recent[-300:]
        def grp(f):  # noqa: E306
            rs = [r for r in recent if f(r)]
            return dict(n=len(rs), clears=float(np.mean([r["clears"] for r in rs])), items=float(np.mean([r["items"] for r in rs])),
                        deaths=float(np.mean([r["deaths"] for r in rs]))) if rs else None
        rv = ret.reshape(-1); vv = V[:, :T].reshape(-1)
        if st["update"] % 5 == 0 or not pool_sz:
            pool_sz = envs.pool_size()
        row = dict(type="train", update=st["update"], env_steps=st["env_steps"], sps=N * T / (time.time() - t0),
                   t_roll=t_roll, t_train=time.time() - t1, pg=pl, vloss=vl, kl_ref_maze1=kl, ent=ent, clipfrac=cf,
                   aux_ce=locals().get("aux_ce_last"), aux_ce_val=locals().get("aux_ce_val"), aux_tgt_ent=locals().get("aux_tgt_ent"), aux_w=AUX0 * max(0.0, 1.0 - st["update"] / AUX_UPD), ent_run=ent_run,
                   expl_var=float(1 - (rv - vv).var() / (rv.var() + 1e-8)), maze1_frac=float(kmask.mean()),
                   board_mean=float(BRD.float().mean()), board3p_frac=float((BRD >= 3).float().mean()),
                   train_recent=dict(b1=grp(lambda r: r["start"] == "b1"), b2=grp(lambda r: r["start"] == "b2"),
                                     b3=grp(lambda r: r["start"] in ("b2", "b2h")),
                                     b3late=grp(lambda r: r["start"] in ("b3late", "b3vlate")),
                                     lives_b3=(lambda L: dict(n=len(L), mean=float(np.mean(L))) if L else None)(
                                         [r["lives_b3"] for r in recent if r.get("lives_b3") is not None and r["b0"] < 3])),
                   pool=pool_sz)
        lvl = st["level"]; rs = [r for r in recent if r["start"] == f"b3cur{lvl}"][-60:]
        row["cur"] = dict(level=lvl, n=len(rs), clear=float(np.mean([r["clears"] > 0 for r in rs])) if rs else None,
                          per_bucket={k: (lambda L: round(float(np.mean(L)), 3) if L else None)([r["clears"] > 0 for r in recent if r["start"] == f"b3cur{k}"]) for k in range(NB)})
        if len(rs) >= PROMOTE_N and row["cur"]["clear"] >= PROMOTE and lvl < NB - 1:
            st["level"] = lvl + 1; (RUN / "level").write_text(str(st["level"])); recent = [r for r in recent if r["start"] != f"b3cur{lvl}"]
            log(f"PROMOTE curriculum level {lvl} -> {st['level']} (train clear {row['cur']['clear']:.2f} over {len(rs)})")
        jwrite(row)
        if tbw and st["update"] % 5 == 0:
            tbw.add_scalar("entropy", ent, st["env_steps"]); tbw.add_scalar("sps", row["sps"], st["env_steps"])
            for k in ("b1", "b3"):
                if row["train_recent"][k]:
                    tbw.add_scalar(f"train_{'b1' if k == 'b1' else 'b2'}_clears", row["train_recent"][k]["clears"], st["env_steps"])
        if st["update"] % 5 == 0:
            r3 = row["train_recent"]["b3"]; r1 = row["train_recent"]["b1"]
            log(f"upd {st['update']} {st['env_steps'] / 1e6:.2f}M sps {row['sps']:.0f} (roll {t_roll:.1f}s train {row['t_train']:.1f}s) "
                f"kl1 {kl:.3f} ent {ent:.3f} cf {cf:.3f} vl {vl:.3f} EV {row['expl_var']:.2f} boards>=3 {row['board3p_frac']:.2f} "
                f"b1-start clears {r1 and round(r1['clears'], 2)} b2-start clear {r3 and round(r3['clears'], 2)} deaths {r3 and round(r3['deaths'], 2)} trunc {np.mean([bool(r.get('truncated')) and r['board'] < 3 for r in recent]) if recent else None:.2f} "
                f"cur L{st['level']} n {row['cur']['n']} clear {row['cur']['clear']}  late-start clear {row['train_recent']['b3late'] and round(row['train_recent']['b3late']['clears'], 2)} pools {pool_sz}")
        if time.time() - t_ck > CKPT_S:
            ckpt(); t_ck = time.time()
        if st["update"] % EVAL_EVERY == 0:
            evaluate(); ckpt()
    ckpt(); (RUN / "final.json").write_text(json.dumps(dict(update=st["update"], env_steps=st["env_steps"], best=st["best"])))
    envs.close(); [e.close() for e in (ev_a, ev_b, ev_d, ev_f) if e]; log("done")


if __name__ == "__main__":
    main()
