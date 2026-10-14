"""M15 (env defaults below): blue-maze-heavy PPO from M13 best: start mix 25% b1 / 15% b2 / 60% maze-2 snapshots
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
RUN = Path(E("RUN", "runs/m15")); INIT = E("INIT", "runs/m13/best.pt"); INIT_VH = E("INIT_VH", "runs/m13/ckpt_last.pt")
RW_DEATH12, ARRIVE_BONUS = float(E("RW_DEATH12", "20.0")), float(E("ARRIVE_BONUS", "5.0"))
SHAPE_K, SHAPE_LEFT = float(E("SHAPE_K", "0.005")), int(E("SHAPE_LEFT", "154"))
NW, PER, T = int(E("ENV_WORKERS", "18")), int(E("PER_WORKER", "6")), int(E("T", "128"))
EPOCHS = int(E("EPOCHS", "2"))
NMB = int(E("MINIBATCHES", "0")) or -(-int(E("ENV_WORKERS", "18")) * int(E("PER_WORKER", "6")) // 12)   # ~12 envs (1536 samples) per minibatch: 6 GB GPU
GAMMA, LAM, CLIP = float(E("GAMMA", "0.995")), float(E("GAE_LAMBDA", "0.95")), float(E("CLIP", "0.1"))
LR_PI, LR_V = float(E("LR_PI", "5e-5")), float(E("LR_V", "3e-4"))
VF, ENT, BETA, TAU = float(E("VF", "0.5")), float(E("ENT", "0.003")), float(E("BETA", "0.05")), float(E("TAU", "0.25"))
VWARM, MAX_GRAD = int(E("VWARM", "10")), float(E("MAX_GRAD", "0.5"))
RW = [float(x) for x in E("RW", "0.1,0.0005,10.0,5.0,2.0").split(",")]
STALL = [float(x) for x in E("STALL", "10,150,0.005,0.01").split(",")]
P_B1, P_B2, P_HARVEST = float(E("P_B1", "0.25")), float(E("P_B2", "0.15")), float(E("P_HARVEST", "0.6"))
P_LATE, P_VLATE, EVAL_LATE_REP = float(E("P_LATE", "0.4")), float(E("P_VLATE", "0.4")), int(E("EVAL_LATE_REP", "2"))
POOL_CAP, MAX_STEPS = int(E("POOL_CAP", "60")), int(E("MAX_STEPS", "8000"))
EVAL_EVERY, EVAL_MB_N, EVAL_SNAP_REP = int(E("EVAL_EVERY", "40")), int(E("EVAL_MB_N", "50")), int(E("EVAL_SNAP_REP", "4"))
EVAL_WORKERS, MIN_B1 = int(E("EVAL_WORKERS", "8")), float(E("MIN_B1", "0.75"))
N_TEST = int(E("N_TEST", "7")); CKPT_S = float(E("CKPT_S", "120"))
P_EXTRA_LIVES, ENT_FLOOR = float(E("P_EXTRA_LIVES", "0.5")), float(E("ENT_FLOOR", "0.25"))
AUX0, AUX_UPD, B_AUX, AUX_DATA = float(E("AUX0", "0.3")), int(E("AUX_UPD", "400")), int(E("B_AUX", "32")), E("AUX_DATA", "runs/m14d1/data")
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


def main():
    RUN.mkdir(parents=True, exist_ok=True); (RUN / "harvest").mkdir(exist_ok=True); dev = torch.device("cuda")
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
    opt = torch.optim.Adam([dict(params=pi.parameters(), lr=LR_PI), dict(params=vh.parameters(), lr=LR_V)], eps=1e-5)
    st = dict(update=0, env_steps=0, best=None, evals=[], wall=0.0, resumes=0, init_update=ck0.get("state", {}).get("update"))
    last = RUN / "ckpt_last.pt"
    if last.exists():
        c = torch.load(last, map_location=dev, weights_only=False)
        pi.load_state_dict(c["pi"]); vh.load_state_dict(c["vh"]); opt.load_state_dict(c["opt"]); st = c["state"]
        st["resumes"] += 1; log(f"resumed {last} at update {st['update']} ({st['env_steps']:,} steps)")
    b3_train, b3_test, test_seeds = split_b3()
    b2 = sorted(glob.glob("runs/multiboard/states/*_board2.pkl"))
    import pickle
    late_files = sorted(glob.glob("runs/m12/late_train/*.pkl"))
    for f in sorted(glob.glob("runs/m11/harvest/*.pkl")):                    # M11 harvest (train seeds only)
        d = pickle.load(open(f, "rb"))
        if d["left"] <= 60:
            late_files.append(f)
    late_test = sorted(glob.glob("runs/m12/late_test/*.pkl"))
    cfg = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, list))}
    cfg.update(ARCH=ARCH, N=N, b3_train=len(b3_train), b3_test=len(b3_test), test_seeds=test_seeds, b2=len(b2),
               late_files=len(late_files), late_test=len(late_test), init_update=st["init_update"])
    (RUN / "config.json").write_text(json.dumps(cfg, indent=1)); log(f"config {json.dumps(cfg)}")
    envs = m13_env.VecGames13(NW, PER, groups=2, rw=RW, rw_death12=RW_DEATH12, arrive_bonus=ARRIVE_BONUS,
                              shape_k=SHAPE_K, shape_left=SHAPE_LEFT, shape_min_board=3, shape_gamma=GAMMA, stall=STALL, p_b1=P_B1, p_b2=P_B2, p_harvest=P_HARVEST,
                              p_late=P_LATE, p_vlate=P_VLATE, late_files=late_files,
                              b2_files=b2, b3_files=b3_train, pool_cap=POOL_CAP, max_steps=MAX_STEPS, p_extra_lives=P_EXTRA_LIVES,
                              harvest_dir=str((RUN / "harvest").resolve()), seed_off=st["update"] % 997)
    # evaluators: (a) board-1 fasteval, (b) multiboard, (c) held-out board-3 snapshots
    fasteval._MAX_STEPS = 20001
    ev_a = M10Evaluator(workers=EVAL_WORKERS)
    os.environ["MB_CAP"] = "20000"; os.environ.pop("MB_STATES", None)
    w0 = fasteval._worker; fasteval._worker = multiboard.mb_worker; ev_b = M10Evaluator(workers=EVAL_WORKERS)
    os.environ["MB_CAP"] = "8000"; os.environ["M11_SNAPS"] = ",".join(b3_test)
    fasteval._worker = m11_env.snap_worker; ev_c = M10Evaluator(workers=EVAL_WORKERS)
    os.environ["M11_SNAPS"] = ",".join(late_test); ev_d = M10Evaluator(workers=EVAL_WORKERS); fasteval._worker = w0

    def evaluate():
        pi.eval(); t0 = time.time()
        sa, _ = ev_a.evaluate(pi, list(range(100)))
        _, rb = ev_b.evaluate(pi, list(range(60000, 60000 + EVAL_MB_N)))
        _, rc = ev_c.evaluate(pi, list(range(len(b3_test) * EVAL_SNAP_REP)))
        _, rd = ev_d.evaluate(pi, list(range(len(late_test) * EVAL_LATE_REP)))
        pi.train()
        b3 = [g["boards"]["3"]["items"] for g in rb if "3" in g["boards"]]
        s = dict(update=st["update"], env_steps=st["env_steps"], secs=round(time.time() - t0),
                 b1_clear=sa["clear_rate"], b1_items=sa["eaten"],
                 mb_clears=float(np.mean([g["clears"] for g in rb])), mb_score=float(np.mean([g["score"] for g in rb])),
                 mb_reach3=float(np.mean([g["max_board"] >= 3 for g in rb])), mb_reach4=float(np.mean([g["max_board"] >= 4 for g in rb])),
                 mb_b3_items=float(np.mean(b3)) if b3 else None,
                 mb_lives_b3=float(np.mean([g["boards"]["3"]["lives"] for g in rb if "3" in g["boards"]])) if b3 else None,
                 snap3_clear=float(np.mean([g["cleared"] for g in rc])), snap3_items=float(np.mean([g["items"] for g in rc])),
                 snap3_deaths=float(np.mean([g["deaths"] for g in rc])),
                 late3_clear=float(np.mean([g["cleared"] for g in rd])), late3_left0=float(np.mean([g["left0"] for g in rd])),
                 late3_items_frac=float(np.mean([g["items"] / max(g["left0"], 1) for g in rd])))
        st["evals"].append(s); jwrite(dict(type="eval", **s))
        ok = s["b1_clear"] >= MIN_B1
        kf = lambda x: (x["mb_clears"], x.get("late3_clear", 0.0), x["snap3_clear"])  # noqa: E731
        better = ok and (st["best"] is None or kf(s) > kf(st["best"]))
        if better:
            st["best"] = s; save(dict(model=pi.state_dict(), arch=ARCH, kind="m10", step=st["update"]), RUN / "best.pt")
        log(f"EVAL upd {st['update']} ({st['env_steps'] / 1e6:.2f}M): b1 clear {s['b1_clear']:.2f}  multiboard clears {s['mb_clears']:.2f} "
            f"reach3 {s['mb_reach3']:.2f} reach4 {s['mb_reach4']:.2f} b3 items {s['mb_b3_items']}  snap3 clear {s['snap3_clear']:.2f} "
            f"items {s['snap3_items']:.1f} deaths {s['snap3_deaths']:.2f}  late3 clear {s['late3_clear']:.2f} (from {s['late3_left0']:.0f} left, ate {s['late3_items_frac']:.2f})  ({s['secs']}s){'  *best*' if better else ''}")

    def ckpt():
        save(dict(pi=pi.state_dict(), vh=vh.state_dict(), opt=opt.state_dict(), arch=ARCH, state=st), last)

    FRA = np.load(f"{AUX_DATA}/frames.npy", mmap_mode="r"); mdA = np.load(f"{AUX_DATA}/demo_meta.npz")
    labA = mdA["lab"] & ~mdA["val_games"][mdA["gid"]]; ixA = np.flatnonzero(labA); arng = np.random.default_rng(st["update"] + 5)
    wA = np.where(mdA["pd"], 0.1, 1.0).astype(np.float32)

    def aux_batch():
        ix = arng.choice(ixA, B_AUX); dec = np.maximum(ix[:, None] + np.arange(-C + 1, 1)[None], mdA["g0"][ix][:, None])
        fi = mdA["fbase"][dec][..., None] + np.arange(4); u, inv = np.unique(fi, return_inverse=True)
        x = np.asarray(FRA[u])[inv.reshape(fi.shape)]
        return (torch.from_numpy(x).to(dev), torch.from_numpy(mdA["act"][ix].astype(np.int64)).to(dev), torch.from_numpy(wA[ix]).to(dev))
    ent_run = 1.0
    s0, b0 = envs.reset()
    stack = torch.from_numpy(s0).to(dev); bobs = torch.from_numpy(b0).to(dev).long()
    new = torch.ones(N, dtype=torch.bool, device=dev); estart = torch.full((N,), C - 1, dtype=torch.long, device=dev)
    H = torch.zeros(N, C - 1 + T, 4, 84, 84, dtype=torch.uint8, device=dev)
    A = torch.zeros(N, T, dtype=torch.long, device=dev); LP = torch.zeros(N, T, device=dev); V = torch.zeros(N, T + 1, device=dev)
    R = torch.zeros(N, T, device=dev); D = torch.zeros(N, T, device=dev); ES = torch.zeros(N, T, dtype=torch.long, device=dev)
    BRD = torch.zeros(N, T, dtype=torch.long, device=dev)
    gidx = [torch.from_numpy(g).to(dev) for g in envs.genv]; caches = [pi.new_cache(len(g), dev) for g in envs.genv]
    recent = []; comp_sum = np.zeros(6); first = True; t_ck = time.time()
    any_new = [True, True]; pool_sz = {}
    if st["update"] == 0 and not st["evals"] and E("EVAL0", "1") == "1":
        evaluate()
    while st["update"] < MAX_UPDATES:
        if time.time() >= DEADLINE:
            log("deadline reached"); break
        if (RUN / "STOP").exists():
            ckpt(); log(f"STOP file: saved ckpt_last.pt at update {st['update']}, exiting 3"); sys.exit(3)
        t0 = time.time(); pi.eval()
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
                    lg = pi.adv(h) / TAU; V[gi, t] = vh(h)
                    a = torch.distributions.Categorical(logits=lg).sample()
                    A[gi, t] = a; LP[gi, t] = F.log_softmax(lg, -1).gather(1, a[:, None])[:, 0]
                    envs.step_send(g, a.cpu().numpy())
            for g, gi in enumerate(gidx):                           # bootstrap V(obs_T) on a copy of the caches
                x = stack[gi]; z = pi.embed(x); cc = [c.clone() for c in caches[g]]; pi.prime(cc, new[gi], z)
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
        t_roll = time.time() - t0; t1 = time.time(); pi.train()
        kmask = (BRD <= 2).float()
        advn = (adv - adv.mean()) / (adv.std() + 1e-8)
        vwarm = st["update"] < VWARM
        stats = []
        for _ in range(EPOCHS):
            for mb in torch.randperm(N, device=dev).chunk(NMB):
                n = len(mb)
                z = pi.embed(H[mb].reshape(-1, 4, 84, 84)).view(n, C - 1 + T, -1)
                ctx = z[torch.arange(n, device=dev)[:, None, None], IDX[mb]].reshape(n * T, C, -1)
                h = pi.trunk_last(ctx)
                v = vh(h.detach() if vwarm else h); vl = F.mse_loss(v, ret[mb].reshape(-1))
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
                        ace = (F.cross_entropy(pi.adv(pi.trunk_last(za)) / TAU, aa, reduction="none") * wa).sum() / wa.sum()
                    ec = ENT * (4.0 if ent_run < ENT_FLOOR else 1.0)
                    loss = pl + VF * vl - ec * ent + BETA * kl + aw * ace
                    cf = ((ratio - 1).abs() > CLIP).float().mean().item()
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(list(pi.parameters()) + list(vh.parameters()), MAX_GRAD); opt.step()
                stats.append((pl.item(), vl.item(), kl.item(), ent.item(), cf)); ent_run = 0.9 * ent_run + 0.1 * ent.item() if not vwarm else ent_run
                aux_ce_last = float(ace) if not vwarm else None
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
                   aux_ce=locals().get("aux_ce_last"), aux_w=AUX0 * max(0.0, 1.0 - st["update"] / AUX_UPD), ent_run=ent_run,
                   expl_var=float(1 - (rv - vv).var() / (rv.var() + 1e-8)), maze1_frac=float(kmask.mean()),
                   board_mean=float(BRD.float().mean()), board3p_frac=float((BRD >= 3).float().mean()),
                   train_recent=dict(b1=grp(lambda r: r["start"] == "b1"), b2=grp(lambda r: r["start"] == "b2"),
                                     b3=grp(lambda r: r["start"] in ("b3", "b3h")),
                                     b3late=grp(lambda r: r["start"] in ("b3late", "b3vlate")),
                                     lives_b3=(lambda L: dict(n=len(L), mean=float(np.mean(L))) if L else None)(
                                         [r["lives_b3"] for r in recent if r.get("lives_b3") is not None and r["b0"] < 3])),
                   pool=pool_sz)
        jwrite(row)
        if st["update"] % 5 == 0:
            r3 = row["train_recent"]["b3"]; r1 = row["train_recent"]["b1"]
            log(f"upd {st['update']} {st['env_steps'] / 1e6:.2f}M sps {row['sps']:.0f} (roll {t_roll:.1f}s train {row['t_train']:.1f}s) "
                f"kl1 {kl:.3f} ent {ent:.3f} cf {cf:.3f} vl {vl:.3f} EV {row['expl_var']:.2f} boards>=3 {row['board3p_frac']:.2f} "
                f"b1-start clears {r1 and round(r1['clears'], 2)} b3-start clear {r3 and round(r3['clears'], 2)} items {r3 and round(r3['items'], 1)} "
                f"late-start clear {row['train_recent']['b3late'] and round(row['train_recent']['b3late']['clears'], 2)} pools {pool_sz}")
        if time.time() - t_ck > CKPT_S:
            ckpt(); t_ck = time.time()
        if st["update"] % EVAL_EVERY == 0:
            evaluate(); ckpt()
    ckpt(); (RUN / "final.json").write_text(json.dumps(dict(update=st["update"], env_steps=st["env_steps"], best=st["best"])))
    envs.close(); [e.close() for e in (ev_a, ev_b, ev_c, ev_d)]; log("done")


if __name__ == "__main__":
    main()
