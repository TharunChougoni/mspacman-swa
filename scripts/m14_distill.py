"""M14: distill the emulator-lookahead search teacher (runs/teacher/demos, maze 2, lives top-up) into the M10 net.

Init: INIT (default runs/m13/best.pt). Loss per step:
  CE(teacher action | 15-token window) on maze-2 demo windows, weight W_PD for post-death states (deaths_before > 0)
  + BETA * KL(pi || frozen INIT) on maze-1 windows (boards 1-2) collected once from the frozen INIT policy.
Logits = adv(h) / TAU (same as PPO policy). Windows: 15 tokens (net.ctx), clamped at the game / episode start.
Data: demos packed once into memmaps under RUN/data (frames + per-decision index), 5% games held out (val acc).
Evals (strict fp32, M10Evaluator): (a) board-1 clear seeds 0-99, (c) held-out board-3 snapshots (7 test seeds),
(d) late_test 60-left. Multiboard later (separate script). best.pt = max (late3 clear, snap3 clear) with b1 >= MIN_B1.
Resumable: RUN/ckpt_last.pt every CKPT_S s; STOP file -> ckpt + exit 3.
"""
from __future__ import annotations

import copy, glob, json, os, sys, threading, time, queue, zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

E = os.environ.get
os.environ.setdefault("RUN", "runs/m14")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import m13_ppo as M  # noqa: E402  (log/jwrite/save/split_b3 + reward cfg; RUN from env)
from msclear import fasteval, m11_env, m13_env  # noqa: E402
from msclear.m10_fasteval import M10Evaluator  # noqa: E402
from msclear.m10_net import M10Net  # noqa: E402

RUN = M.RUN; DATA = RUN / "data"
INIT = E("INIT", "runs/m13/best.pt"); DEMO = E("DEMO_DIR", "runs/teacher/demos")   # comma-separated dirs allowed
WD, P_DAG = float(E("WD", "0.0")), float(E("P_DAG", "0.5"))   # P_DAG: batch share of DAgger labels
B_T, B_M1 = int(E("B_T", "128")), int(E("B_M1", "32")); LR = float(E("LR", "1e-4")); TAU = float(E("TAU", "0.25"))
BETA, W_PD = float(E("BETA", "1.0")), float(E("W_PD", "0.3")); MAX_STEPS = int(E("MAX_STEPS", "30000"))
EVAL_EVERY, CKPT_S = int(E("EVAL_EVERY", "1500")), float(E("CKPT_S", "300")); M1_STEPS = int(E("M1_STEPS", "60000"))
EVAL_WORKERS, MIN_B1, VAL_FRAC = int(E("EVAL_WORKERS", "8")), float(E("MIN_B1", "0.75")), float(E("VAL_FRAC", "0.05"))
log, jwrite, save = M.log, M.jwrite, M.save


def pack_demos(C):
    """frames.u8 (Nf,84,84) with 3 pad frames per game; per decision: base frame idx of its stack, game start decision idx."""
    if (DATA / "demo_meta.npz").exists():
        return
    DATA.mkdir(parents=True, exist_ok=True)
    fs = sorted(f for d in DEMO.split(",") for f in glob.glob(f"{d}/*.npz") if not f.endswith(".tmp.npz")); nf = 0; nd = 0; meta = []
    for f in fs:
        z = np.load(f); n = len(z["act"]); meta.append((f, n)); nf += n + 3; nd += n
    FR = np.lib.format.open_memmap(DATA / "frames.npy", "w+", np.uint8, (nf, 84, 84))
    act = np.zeros(nd, np.int8); fbase = np.zeros(nd, np.int64); g0 = np.zeros(nd, np.int64); pd = np.zeros(nd, bool)
    gid = np.zeros(nd, np.int32); lab = np.ones(nd, bool); fo = 0; do = 0
    for k, (f, n) in enumerate(meta):
        z = np.load(f)
        FR[fo:fo + 3] = z["stack0"][:3]; FR[fo + 3:fo + 3 + n] = z["frames"]
        act[do:do + n] = z["act"]; fbase[do:do + n] = fo + np.arange(n); g0[do:do + n] = do
        pd[do:do + n] = z["deaths_before"] > 0; gid[do:do + n] = k
        if "labeled" in z:
            lab[do:do + n] = z["labeled"]
        fo += n + 3; do += n
    FR.flush()
    val = np.array([(zlib.crc32(Path(f).name.encode()) % 1000) < VAL_FRAC * 1000 for f, _ in meta])
    np.savez(DATA / "demo_meta.npz", lab=lab, act=act, fbase=fbase, g0=g0, pd=pd, gid=gid, val_games=val, files=np.array([f for f, _ in meta]))
    log(f"packed {len(meta)} games, {nd:,} decisions, {nf:,} frames ({nf * 7056 / 1e9:.1f} GB)")


def collect_maze1(ref, dev, C):
    """Frozen-INIT rollouts on boards 1-2 (fresh + board-2 snapshots); stores 4-stacks + 15-token window indices."""
    if (DATA / "m1_win.npy").exists():
        return
    DATA.mkdir(parents=True, exist_ok=True)
    b3_train, _, _ = M.split_b3(); b2 = sorted(glob.glob("runs/multiboard/states/*_board2.pkl"))
    NW, PER = 12, 4; N = NW * PER
    envs = m13_env.VecGames13(NW, PER, groups=1, rw=M.RW, rw_death12=M.RW_DEATH12, arrive_bonus=M.ARRIVE_BONUS,
                              shape_k=0.0, shape_left=M.SHAPE_LEFT, shape_min_board=3, shape_gamma=M.GAMMA, stall=M.STALL,
                              p_b1=0.6, p_b2=0.4, p_harvest=0.0, p_late=0.0, p_vlate=0.0, late_files=[], b2_files=b2,
                              b3_files=b3_train, pool_cap=1, max_steps=M.MAX_STEPS, harvest_dir="/tmp/m14_harvest", seed_off=7)
    os.makedirs("/tmp/m14_harvest", exist_ok=True)
    ST = np.lib.format.open_memmap(DATA / "m1_stacks.npy", "w+", np.uint8, (M1_STEPS, 4, 84, 84))
    WIN = np.zeros((M1_STEPS, C), np.int64); hist = [[] for _ in range(N)]; m = 0
    s0, b0 = envs.reset(); stack = torch.from_numpy(s0).to(dev); brd = b0.copy()
    gi = torch.from_numpy(envs.genv[0]).to(dev); cache = ref.new_cache(N, dev); new = torch.ones(N, dtype=torch.bool, device=dev)
    t0 = time.time()
    with torch.no_grad():
        while m < M1_STEPS:
            x = stack[gi]; z = ref.embed(x)
            if new.any():
                ref.prime(cache, new, z)
            a = torch.distributions.Categorical(logits=ref.adv(ref.step(cache, z)) / TAU).sample()
            xs = x.cpu().numpy(); idx = envs.genv[0]
            for j, e in enumerate(idx):
                if new[j]:
                    hist[e] = []
                if brd[e] <= 2 and m < M1_STEPS:
                    ST[m] = xs[j]; hist[e].append(m)
                    h = hist[e][-C:]; WIN[m] = [h[0]] * (C - len(h)) + h; m += 1
            envs.step_send(0, a.cpu().numpy())
            fr, rew, dn, comp, b, res, starts = envs.step_recv(0)
            sg = torch.cat([stack[gi, 1:], torch.from_numpy(fr).to(dev)[:, None]], 1)
            for i, s_ in starts.items():
                sg[i] = torch.from_numpy(s_).to(dev)
            stack[gi] = sg; new = torch.from_numpy(dn).to(dev).bool(); brd[idx] = b
    envs.close(); ST.flush(); np.save(DATA / "m1_win.npy", WIN)
    log(f"maze-1 buffer: {M1_STEPS:,} steps in {time.time() - t0:.0f}s")


def main():
    RUN.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    ck0 = torch.load(INIT, map_location="cpu", weights_only=False); ARCH = ck0["arch"]
    pi = M10Net(**ARCH); pi.load_state_dict(ck0["pi"] if "pi" in ck0 else ck0["model"]); pi = pi.to(dev)
    ref = copy.deepcopy(pi).eval().requires_grad_(False); C = pi.ctx
    opt = torch.optim.AdamW(pi.parameters(), lr=LR, eps=1e-5, weight_decay=WD)
    st = dict(step=0, best=None, evals=[], wall=0.0, resumes=0)
    last = RUN / "ckpt_last.pt"
    if last.exists():
        c = torch.load(last, map_location=dev, weights_only=False)
        pi.load_state_dict(c["pi"]); opt.load_state_dict(c["opt"]); st = c["state"]; st["resumes"] += 1
        log(f"resumed at step {st['step']}")
    pack_demos(C); collect_maze1(ref, dev, C)
    FR = np.load(DATA / "frames.npy", mmap_mode="r"); md = np.load(DATA / "demo_meta.npz")
    act, fbase, g0, pd, gid = md["act"], md["fbase"], md["g0"], md["pd"], md["gid"]
    lab = md["lab"] if "lab" in md else np.ones(len(act), bool)
    isval = md["val_games"][gid]; tr_idx = np.flatnonzero(~isval & lab); va_idx = np.flatnonzero(isval & lab)
    dag = np.array(["dagger" in str(f) for f in md["files"]])[gid]
    tr_dag, tr_dem = tr_idx[dag[tr_idx]], tr_idx[~dag[tr_idx]]
    M1 = np.load(DATA / "m1_stacks.npy", mmap_mode="r"); M1W = np.load(DATA / "m1_win.npy")
    cfg = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str))}
    cfg.update(ARCH=ARCH, ctx=C, demo_decisions=int(len(act)), train_dec=int(len(tr_idx)), val_dec=int(len(va_idx)),
               val_games=int(md["val_games"].sum()), dagger_train_labels=int((dag[tr_idx]).sum()) if len(tr_idx) else 0, postdeath_frac=float(pd.mean()), m1_steps=int(len(M1W)))
    (RUN / "config.json").write_text(json.dumps(cfg, indent=1)); log(f"config {json.dumps(cfg)}")

    def demo_batch(ix):
        dec = np.maximum(ix[:, None] + np.arange(-C + 1, 1)[None], g0[ix][:, None])      # (B, C) decision idx, clamped
        fi = fbase[dec][..., None] + np.arange(4)                                          # (B, C, 4) frame idx
        u, inv = np.unique(fi, return_inverse=True)
        x = np.asarray(FR[u])[inv.reshape(fi.shape)]                                       # (B, C, 4, 84, 84)
        return x, act[ix].astype(np.int64), np.where(pd[ix], W_PD, 1.0).astype(np.float32)

    def m1_batch(ix):
        w = M1W[ix]; u, inv = np.unique(w, return_inverse=True)
        return np.asarray(M1[u])[inv.reshape(w.shape)]

    rng = np.random.default_rng(st["step"] + 1)
    q = queue.Queue(maxsize=6)

    def loader():
        while True:
            nd_ = int(rng.binomial(B_T, P_DAG)) if len(tr_dag) else 0
            ix = np.concatenate([rng.choice(tr_dag, nd_) if nd_ else np.zeros(0, np.int64), rng.choice(tr_dem, B_T - nd_)])
            x, a, w = demo_batch(ix); xm = m1_batch(rng.integers(0, len(M1W), B_M1))
            q.put((torch.from_numpy(x).pin_memory(), torch.from_numpy(a), torch.from_numpy(w), torch.from_numpy(xm).pin_memory()))
    for _ in range(3):
        threading.Thread(target=loader, daemon=True).start()
    vrng = np.random.default_rng(123); vix = vrng.choice(va_idx, min(4096, len(va_idx)), replace=False)

    def logits(net, x):
        B = x.shape[0]; z = net.embed(x.reshape(-1, 4, 84, 84)).view(B, C, -1)
        return net.adv(net.trunk_last(z)) / TAU

    def val():
        pi.eval(); acc = []; nll = []; accl = []
        with torch.no_grad():
            for ch in np.array_split(vix, max(1, len(vix) // 128)):
                x, a, w = demo_batch(ch); x = torch.from_numpy(x).to(dev); a = torch.from_numpy(a).to(dev)
                lg = logits(pi, x); acc.append((lg.argmax(-1) == a).float().cpu()); nll.append(F.cross_entropy(lg, a, reduction="none").cpu())
                accl.append(torch.from_numpy(~pd[ch]))
        pi.train(); acc = torch.cat(acc); nll = torch.cat(nll); alive = torch.cat(accl)
        return float(acc.mean()), float(nll.mean()), float(acc[alive].mean()) if alive.any() else None

    fasteval._MAX_STEPS = 20001
    b3_train, b3_test, test_seeds = M.split_b3(); late_test = sorted(glob.glob("runs/m12/late_test/*.pkl"))
    ev_a = M10Evaluator(workers=EVAL_WORKERS); w0 = fasteval._worker
    os.environ["MB_CAP"] = "8000"; os.environ["M11_SNAPS"] = ",".join(b3_test)
    fasteval._worker = m11_env.snap_worker; ev_c = M10Evaluator(workers=EVAL_WORKERS)
    os.environ["M11_SNAPS"] = ",".join(late_test); ev_d = M10Evaluator(workers=EVAL_WORKERS); fasteval._worker = w0

    def evaluate():
        pi.eval(); t0 = time.time()
        sa, _ = ev_a.evaluate(pi, list(range(100)))
        _, rc = ev_c.evaluate(pi, list(range(len(b3_test) * 4)))
        _, rd = ev_d.evaluate(pi, list(range(len(late_test) * 2)))
        pi.train(); va = val()
        s = dict(step=st["step"], secs=round(time.time() - t0), b1_clear=sa["clear_rate"], b1_items=sa["eaten"],
                 snap3_clear=float(np.mean([g["cleared"] for g in rc])), snap3_items=float(np.mean([g["items"] for g in rc])),
                 snap3_deaths=float(np.mean([g["deaths"] for g in rc])),
                 late3_clear=float(np.mean([g["cleared"] for g in rd])),
                 late3_items_frac=float(np.mean([g["items"] / max(g["left0"], 1) for g in rd])),
                 val_acc=va[0], val_nll=va[1], val_acc_alive=va[2])
        st["evals"].append(s); jwrite(dict(type="eval", env_steps=st["step"] * B_T, **s))
        kf = lambda x: (x["late3_clear"], x["snap3_clear"], x["late3_items_frac"])  # noqa: E731
        better = s["b1_clear"] >= MIN_B1 and (st["best"] is None or kf(s) > kf(st["best"]))
        if better:
            st["best"] = s; save(dict(model=pi.state_dict(), arch=ARCH, kind="m10", step=st["step"]), RUN / "best.pt")
        log(f"EVAL step {st['step']}: b1 clear {s['b1_clear']:.2f}  snap3 clear {s['snap3_clear']:.2f} items {s['snap3_items']:.1f} "
            f"deaths {s['snap3_deaths']:.2f}  late3 clear {s['late3_clear']:.2f} ate {s['late3_items_frac']:.2f}  "
            f"val acc {va[0]:.3f} (alive {va[2]:.3f}) nll {va[1]:.3f}  ({s['secs']}s){'  *best*' if better else ''}")

    def ckpt():
        save(dict(pi=pi.state_dict(), opt=opt.state_dict(), arch=ARCH, state=st), last)

    if st["step"] == 0 and not st["evals"] and E("EVAL0", "1") == "1":
        evaluate()
    t_ck = time.time(); acc_s = []; t_last = time.time()
    while st["step"] < MAX_STEPS:
        if (RUN / "STOP").exists():
            ckpt(); log(f"STOP: ckpt at step {st['step']}, exit 3"); sys.exit(3)
        x, a, w, xm = q.get()
        x = x.to(dev, non_blocking=True); a = a.to(dev); w = w.to(dev); xm = xm.to(dev, non_blocking=True)
        lg = logits(pi, x); ce = (F.cross_entropy(lg, a, reduction="none") * w).sum() / w.sum()
        lm = F.log_softmax(logits(pi, xm), -1)
        with torch.no_grad():
            lr_ = F.log_softmax(logits(ref, xm), -1)
        kl = (lm.exp() * (lm - lr_)).sum(-1).mean()
        loss = ce + BETA * kl
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt.step()
        st["step"] += 1
        acc_s.append(((lg.argmax(-1) == a).float().mean().item(), ce.item(), kl.item()))
        if st["step"] % 100 == 0:
            dt = time.time() - t_last; t_last = time.time(); st["wall"] += dt
            ac, cl, kv = np.mean(acc_s, 0); acc_s = []
            jwrite(dict(type="train", env_steps=st["step"] * B_T, step=st["step"], ce=cl, kl_m1=kv, acc=ac, steps_per_s=100 / dt, samples=st["step"] * B_T, qsize=q.qsize()))
            log(f"step {st['step']} ce {cl:.3f} acc {ac:.3f} kl_m1 {kv:.4f} {100 / dt:.2f} it/s (queue {q.qsize()})")
        if time.time() - t_ck > CKPT_S:
            ckpt(); t_ck = time.time()
        if st["step"] % EVAL_EVERY == 0:
            evaluate(); ckpt()
    ckpt(); log("done")
    [e.close() for e in (ev_a, ev_c, ev_d)]


if __name__ == "__main__":
    main()
