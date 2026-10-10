"""M9 run: PPO fine-tune of the M8 endgame student (runs/m8/best.pt, confirm 22.3%) on the option-A fast paths.

Same PPO machinery and endgame start pool as scripts/m8_ppo.py (see its docstring and m7_ppo.py). Changes:
  * SWAv2.trunk with the exact last-token final layer (msclear/swa_net.py, MSCLEAR_FAST_LAST=1, commit e481d2f).
  * REF_AT_TRAIN=1: the frozen KL reference (M8 best) is NOT run per rollout step; its log-probs are computed once per
    update, batched over the stored segment with the same H / IDX window machinery as training (fp32).
    CHECK_REF=1 also runs the old per-step path and logs/asserts max |diff| < 1e-4.
  * EVAL_MODE=gpu: periodic evals with msclear.fasteval.FastEvaluator (fp32 GPU, 14 torch-free env workers,
    synchronous, ~1 min for 100 games) instead of 6 async CPU eval workers. EVAL_BF16 stays 0 (bf16 eval: 0.23 vs 0.32).
  * Stall penalty (M8 failure mode: 23% of games ran into the 6000-step cap with a few pellets left):
    STALL="items,n,pen,cap" (default 10,150,0.005,0.01): with <= 10 items left and no item for > 150 decisions,
    reward -= min(0.01, 0.005 + 0.005 (since - 150) / 150) per decision. STALL=0 disables it.
  * KL leash to frozen M8 best, beta 0.05 constant; value head from runs/m8/ckpt_last.pt + VWARM value-only updates.
All in fp32 (no bf16 anywhere).
"""
from __future__ import annotations

import collections
import contextlib
import copy
import threading
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msclear import m6eval  # noqa: E402
from msclear.fasteval import FastEvaluator  # noqa: E402
from msclear.endgame_env import VecGames  # noqa: E402
from msclear.swa_net import SWAv2  # noqa: E402

E = os.environ.get
RUN = Path(E("RUN", "runs/m9run"))
INIT = E("INIT", "runs/m8/best.pt")
VH_INIT = E("VH_INIT", "runs/m8/ckpt_last.pt")
REF_AT_TRAIN = E("REF_AT_TRAIN", "1") != "0"
CHECK_REF = E("CHECK_REF", "0") == "1"
EVAL_MODE = E("EVAL_MODE", "gpu")
EVAL_BF16 = E("EVAL_BF16", "0") == "1"
OVERLAP = E("OVERLAP", "0") == "1"   # double-buffered rollout/train: exact but no faster here (GIL-bound), off
STALL = None if E("STALL", "10,150,0.005,0.01") == "0" else tuple(float(x) for x in E("STALL", "10,150,0.005,0.01").split(","))
P_POOL = float(E("P_POOL", "0.6"))
POOL_CAP = int(E("POOL_CAP", "200"))
NW, PER = int(E("ENV_WORKERS", "14")), int(E("PER_WORKER", "3"))
T = int(E("T", "128"))
EPOCHS = int(E("EPOCHS", "3"))
NMB = int(E("MINIBATCHES", "3"))
GAMMA, LAM = float(E("GAMMA", "0.995")), float(E("GAE_LAMBDA", "0.95"))
CLIP = float(E("CLIP", "0.1"))
LR_PI, LR_V = float(E("LR_PI", "5e-5")), float(E("LR_V", "3e-4"))
VF, ENT = float(E("VF", "0.5")), float(E("ENT", "0.001"))
BETA0, BETA1, KL_ANNEAL = float(E("BETA0", "0.05")), float(E("BETA1", "0.05")), int(E("KL_ANNEAL", "4000"))
TAU = float(E("TAU", "0.25"))   # sampled pi at tau 1 eats ~100 items (greedy 134); tau 0.25 ~ greedy (131, measured)
VWARM = int(E("VWARM", "10"))
MAX_GRAD = float(E("MAX_GRAD", "0.5"))
RW = tuple(float(x) for x in E("RW", "0.1,0.0005,10.0,5.0,2.0").split(","))   # item, score, clear, death, K_END
UPDATES = int(E("UPDATES", "100000"))
DEADLINE = float(E("DEADLINE", "0")) or float("inf")
EVAL_EVERY = int(E("EVAL_EVERY", "150"))
EVAL_N = int(E("EVAL_N", "100"))
EVAL_SEED0 = int(E("EVAL_SEED0", "0"))
EVAL_WORKERS = int(E("EVAL_WORKERS", "14"))
CKPT_S = float(E("CKPT_S", "300"))
MAX_STEPS = int(E("MAX_STEPS", "6000"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def jwrite(row):
    with open(RUN / "metrics.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n"); f.flush(); os.fsync(f.fileno())


def atomic_save(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def trunk(net: SWAv2, z: torch.Tensor) -> torch.Tensor:
    """(B, W, d) tokens -> h (B, d) at the newest token (SWAv2.trunk: exact last-token final layer)."""
    return net.trunk(z)


class _Done:
    """Finished eval handle with the multiprocessing AsyncResult interface used by finish_eval."""
    def __init__(self, res):
        self.res = res

    def ready(self):
        return True

    def get(self):
        return self.res


def logits_of(net, h):
    return net.adv(h) / TAU


class ValueHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, h):
        return self.f(h)[:, 0]


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    dev = torch.device(E("DEV", "cuda"))
    tf32 = E("TF32", "1") == "1"   # TF32 matmul/conv for training as in M8 (strict fp32 was 2.5x slower); evals are strict fp32
    torch.backends.cuda.matmul.allow_tf32 = tf32; torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    ck0 = torch.load(INIT, map_location="cpu", weights_only=False)
    ARCH = ck0["arch"]; W = ARCH["window"]
    pi = SWAv2(**ARCH).to(dev); pi.load_state_dict(ck0["model"])
    ref = SWAv2(**ARCH).to(dev); ref.load_state_dict(ck0["model"]); ref.eval().requires_grad_(False)
    vh = ValueHead(ARCH["d"]).to(dev)
    opt = torch.optim.Adam([dict(params=pi.parameters(), lr=LR_PI), dict(params=vh.parameters(), lr=LR_V)], eps=1e-5)
    st = dict(update=0, env_steps=0, episodes=0, best=None, evals=[], started=time.time(), wall=0.0, resumes=0)
    last = RUN / "ckpt_last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        pi.load_state_dict(ck["pi"]); vh.load_state_dict(ck["vh"]); opt.load_state_dict(ck["opt"])
        st = ck["state"]; st["resumes"] = st.get("resumes", 0) + 1
        log(f"resumed {last} at update {st['update']} ({st['env_steps']:,} env steps, {st['episodes']} episodes)")
    else:
        msg = "fresh value head"
        if VH_INIT and Path(VH_INIT).exists():
            vh.load_state_dict(torch.load(VH_INIT, map_location=dev, weights_only=False)["vh"]); msg = f"value head from {VH_INIT}"
        log(f"init actor from {INIT} (step {ck0.get('step')}), {msg}")
    cfg = dict(INIT=INIT, N=NW * PER, T=T, EPOCHS=EPOCHS, NMB=NMB, GAMMA=GAMMA, LAM=LAM, CLIP=CLIP, LR_PI=LR_PI, LR_V=LR_V,
               VF=VF, ENT=ENT, BETA0=BETA0, BETA1=BETA1, KL_ANNEAL=KL_ANNEAL, TAU=TAU, VWARM=VWARM, RW=RW, ARCH=ARCH,
               EVAL_EVERY=EVAL_EVERY, REF_AT_TRAIN=REF_AT_TRAIN, EVAL_MODE=EVAL_MODE, EVAL_BF16=EVAL_BF16, STALL=STALL, TF32=E("TF32", "1") == "1", OVERLAP=OVERLAP,
               FAST_LAST=__import__("msclear.swa_net", fromlist=["x"]).FAST_LAST, P_POOL=P_POOL, POOL_CAP=POOL_CAP, VH_INIT=VH_INIT, EVAL_N=EVAL_N, EVAL_SEED0=EVAL_SEED0, MAX_STEPS=MAX_STEPS)
    (RUN / "config.json").write_text(json.dumps(cfg, indent=1))
    N = NW * PER
    log(f"config {json.dumps(cfg)}")
    envs = VecGames(NW, PER, sticky=0.25, max_steps=MAX_STEPS, rw=RW, seed_off=st["update"] * 17 % 90_000,
                    p_pool=P_POOL, pool_cap=POOL_CAP, stall=STALL)
    if EVAL_MODE == "gpu":
        fev = FastEvaluator(workers=EVAL_WORKERS, device="cuda", bf16=EVAL_BF16); pool = None
    else:
        fev = None; pool = m6eval.make_pool(EVAL_WORKERS)
    seeds = list(range(EVAL_SEED0, EVAL_SEED0 + EVAL_N))
    pending = None

    def save():
        st["wall"] = st.get("wall", 0.0)
        atomic_save(dict(pi=pi.state_dict(), vh=vh.state_dict(), opt=opt.state_dict(), arch=ARCH, state=st), last)

    def start_eval(u):
        cand = RUN / f"cand_{u:06d}.pt"
        atomic_save(dict(model=pi.state_dict(), arch=ARCH, step=u), cand)
        if fev is not None:                                   # synchronous fp32 GPU eval of the live net
            t0 = time.time(); _, res = fev.evaluate(pi, seeds, sticky=0.25); pi.train()
            return (u, cand, _Done(res), t0, st["env_steps"])
        return (u, cand, pool.map_async(m6eval.play, [(str(cand), s, 0.25) for s in seeds]), time.time(), st["env_steps"])

    def finish_eval(p):
        u, cand, ar, t0, es = p
        res = sorted(ar.get(), key=lambda r: r["seed"])
        s = m6eval.summarize(res); s.update(update=u, env_steps=es, secs=round(time.time() - t0))
        st["evals"].append(s); jwrite(dict(type="eval", **s))
        key = (s["clear_rate"], s["eaten"])
        better = st["best"] is None or key > (st["best"]["clear_rate"], st["best"]["eaten"])
        if better:
            st["best"] = s; shutil.copyfile(cand, RUN / "best.pt.tmp"); os.replace(RUN / "best.pt.tmp", RUN / "best.pt")
        cand.unlink(missing_ok=True)
        log(f"EVAL update {u} ({es / 1e6:.2f}M steps): clear {s['clear_rate']:.2f} ± {s['clear_se']:.2f}  items {s['eaten']:.1f}  "
            f"score {s['score']:.0f}  deaths {s['deaths']:.2f}  steps {s['steps']:.0f}  ({s['secs']}s){'  *best*' if better else ''}")

    # ---- rollout state (S: carried across segments; B: per-segment buffers, two of them when OVERLAP)
    S = dict(stack=torch.from_numpy(envs.reset()).to(dev).contiguous(),            # (N, 4, 84, 84)
             new=torch.ones(N, dtype=torch.bool, device=dev),
             estart=torch.full((N,), W - 1, dtype=torch.long, device=dev),
             tok=torch.zeros(N, W, ARCH["d"], device=dev), rtok=torch.zeros(N, W, ARCH["d"], device=dev))

    def make_buf():
        return dict(H=torch.zeros(N, W - 1 + T, 4, 84, 84, dtype=torch.uint8, device=dev),
                    A=torch.zeros(N, T, dtype=torch.long, device=dev), LP=torch.zeros(N, T, device=dev),
                    V=torch.zeros(N, T + 1, device=dev), R=torch.zeros(N, T, device=dev), D=torch.zeros(N, T, device=dev),
                    RLP=torch.zeros(N, T, 4, device=dev), ES=torch.zeros(N, T, dtype=torch.long, device=dev))
    bufs = [make_buf()] + ([make_buf()] if OVERLAP else [])
    # behaviour copies: with OVERLAP the next segment is collected by the pre-update policy while the GPU trains on
    # the current one (one update of policy lag; the PPO ratio uses the logged behaviour log-probs LP, V is behaviour V)
    beh, bvh = (copy.deepcopy(pi).eval().requires_grad_(False), copy.deepcopy(vh).eval().requires_grad_(False)) if OVERLAP else (pi, vh)

    main_stream = torch.cuda.current_stream() if dev.type == "cuda" else None
    rstream = torch.cuda.Stream() if (OVERLAP and dev.type == "cuda") else None

    def rollout(B, prev, net, vnet):
        """Fill buffer B with T steps of all N envs using (net, vnet). prev = buffer of the previous segment (context)."""
        t0 = time.time(); comp_sum = 0; res_all = []
        H, A, LP, V, R, D, RLP, ES = (B[k] for k in ("H", "A", "LP", "V", "R", "D", "RLP", "ES"))
        if rstream is not None:                               # own CUDA stream so small rollout kernels don't queue
            rstream.wait_stream(main_stream)                  # behind training kernels; sees the latest beh weights
        with torch.no_grad(), (torch.cuda.stream(rstream) if rstream is not None else contextlib.nullcontext()):
            if prev is not None:                              # carry the last W-1 stacks as context
                H[:, :W - 1] = prev["H"][:, T:T + W - 1]
                S["estart"] = (S["estart"] - T).clamp(min=0)
            stack, new, estart, tok, rtok = S["stack"], S["new"], S["estart"], S["tok"], S["rtok"]
            for t in range(T):
                pos = W - 1 + t
                H[:, pos] = stack
                estart = torch.where(new, torch.full_like(estart, pos), estart)
                ES[:, t] = estart
                z = net.embed(stack)
                nn_ = new[:, None, None]
                tok = torch.where(nn_, z[:, None].expand(-1, W, -1), torch.cat([tok[:, 1:], z[:, None]], 1))
                if not REF_AT_TRAIN or CHECK_REF:                 # old path: frozen reference every rollout step
                    rz = ref.embed(stack)
                    rtok = torch.where(nn_, rz[:, None].expand(-1, W, -1), torch.cat([rtok[:, 1:], rz[:, None]], 1))
                    RLP[:, t] = F.log_softmax(logits_of(ref, trunk(ref, rtok)), -1)
                h = trunk(net, tok)
                lg = logits_of(net, h); V[:, t] = vnet(h)
                a = torch.distributions.Categorical(logits=lg).sample()
                A[:, t] = a; LP[:, t] = F.log_softmax(lg, -1).gather(1, a[:, None])[:, 0]
                fr, rew, done, comp, results, starts = envs.step(a.cpu().numpy())
                R[:, t] = torch.from_numpy(rew).to(dev); dn = torch.from_numpy(done).to(dev)
                D[:, t] = dn.float(); comp_sum = comp_sum + comp.sum(0)
                fr = torch.from_numpy(fr).to(dev)
                new = dn.bool()
                stack = torch.cat([stack[:, 1:], fr[:, None]], 1)
                for i, s0 in starts.items():                              # new episode: its own start stack
                    stack[i] = torch.from_numpy(s0).to(dev)               # (normal: 4 x first frame; pool: stored)
                stack = stack.contiguous()
                res_all.extend(results)
            # bootstrap value of the obs after the last step (token window of the next obs)
            z = net.embed(stack)
            btok = torch.where(new[:, None, None], z[:, None].expand(-1, W, -1), torch.cat([tok[:, 1:], z[:, None]], 1))
            V[:, T] = vnet(trunk(net, btok))
            if dev.type == "cuda":
                torch.cuda.synchronize()
        S.update(stack=stack, new=new, estart=estart, tok=tok, rtok=rtok)
        return dict(B=B, comp=comp_sum, results=res_all, t_roll=time.time() - t0)

    class Job:                                                # rollout in a background thread (OVERLAP) or inline
        def __init__(self, *args):
            self.out = None; self.err = None
            if OVERLAP:
                self.th = threading.Thread(target=self._run, args=args, daemon=True); self.th.start()
            else:
                self.out = rollout(*args)

        def _run(self, *args):
            try:
                self.out = rollout(*args)
            except BaseException as e:                        # re-raised in the main thread
                self.err = e

        def result(self):
            if OVERLAP:
                self.th.join()
            if self.err is not None:
                raise self.err
            return self.out

    ar = torch.arange(W, device=dev)
    recent = collections.deque(maxlen=200)
    recent_pool = collections.deque(maxlen=300)
    comp_acc = np.zeros(6 if STALL else 5); t_ck = time.time(); t_run0 = time.time(); steps0 = st["env_steps"]
    stopped = "updates"

    t_eval = 0.0; k_buf = 0
    job = Job(bufs[0], None, beh, bvh) if OVERLAP else None
    while st["update"] < UPDATES:
        if time.time() > DEADLINE:
            stopped = "deadline"; break
        if (RUN / "STOP").exists():
            if job is not None:
                job.result()
            save(); envs.close(); log(f"STOP file: saved ckpt_last.pt at update {st['update']}, exiting 3")
            (RUN / "STOP").unlink(missing_ok=True)
            if pool is not None:
                pool.terminate()
            if fev is not None:
                fev.close()
            os._exit(3)
        u = st["update"]; t_u = time.time()
        # ------------------------------------------------------------- rollout
        if OVERLAP:
            out = job.result(); t_wait = time.time() - t_u
            beh.load_state_dict(pi.state_dict()); bvh.load_state_dict(vh.state_dict())    # pre-update policy
            k_buf ^= 1
            job = Job(bufs[k_buf], out["B"], beh, bvh)       # next segment collected while we train on this one
        else:
            pi.eval()
            out = rollout(bufs[0], bufs[0] if u > 0 or st["env_steps"] > 0 else None, pi, vh); t_wait = out["t_roll"]
        B = out["B"]; t_roll = out["t_roll"]
        comp_acc += out["comp"]
        for r in out["results"]:
            (recent if r["start"] == "normal" else recent_pool).append(r); st["episodes"] += 1
        H, A, LP, V, R, D, RLP, ES = (B[k] for k in ("H", "A", "LP", "V", "R", "D", "RLP", "ES"))
        st["env_steps"] += N * T
        # ------------------------------------------------------------- GAE
        adv = torch.zeros(N, T, device=dev); g = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nonterm = 1.0 - D[:, t]
            delta = R[:, t] + GAMMA * V[:, t + 1] * nonterm - V[:, t]
            g = delta + GAMMA * LAM * nonterm * g
            adv[:, t] = g
        ret = adv + V[:, :T]
        # window indices: (T, W) positions in H, clamped at the episode start
        pos = (W - 1 + torch.arange(T, device=dev))[:, None] - (W - 1) + ar[None]       # (T, W) oldest first
        IDX = torch.maximum(pos[None].expand(N, -1, -1), ES[:, :, None])               # (N, T, W)
        t_ref = time.time()
        if REF_AT_TRAIN:                                      # KL reference, batched once per update (fp32)
            rlp_old = RLP.clone() if CHECK_REF else None
            prev_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # ref in strict fp32
            with torch.no_grad():
                for mb in torch.arange(N, device=dev).chunk(NMB):
                    nb = len(mb)
                    Zr = ref.embed(H[mb].reshape(-1, 4, 84, 84)).view(nb, W - 1 + T, -1)
                    win = Zr[torch.arange(nb, device=dev)[:, None, None], IDX[mb]]
                    RLP[mb] = F.log_softmax(logits_of(ref, trunk(ref, win.reshape(nb * T, W, -1))), -1).view(nb, T, 4)
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev_tf32
            if CHECK_REF:
                dmax = float((RLP - rlp_old).abs().max())
                log(f"CHECK_REF update {u}: max |ref logp (train-time batched) - ref logp (per rollout step)| = {dmax:.2e}")
                jwrite(dict(type="check_ref", update=u, max_abs_diff=dmax))
                assert dmax < (3e-2 if prev_tf32[0] else 1e-4), dmax   # TF32 per-step path is only ~1e-2 exact
            if dev.type == "cuda":
                torch.cuda.synchronize()                      # so t_ref measures the pre-pass, not just its launch
        t_ref = time.time() - t_ref
        # ------------------------------------------------------------- update
        warm = u < VWARM
        beta = BETA0 + (BETA1 - BETA0) * min(1.0, max(0, u - VWARM) / max(1, KL_ANNEAL))
        pi.train()
        stats = collections.defaultdict(list)
        t_tr = time.time()
        for ep in range(EPOCHS):
            perm = torch.randperm(N, device=dev)
            for mb in perm.chunk(NMB):
                nb = len(mb)
                with torch.set_grad_enabled(not warm):
                    Z = pi.embed(H[mb].reshape(-1, 4, 84, 84)).view(nb, W - 1 + T, -1)
                    win = Z[torch.arange(nb, device=dev)[:, None, None], IDX[mb]]           # (nb, T, W, d)
                    h = trunk(pi, win.reshape(nb * T, W, -1))
                    lg = logits_of(pi, h)
                v = vh(h.detach() if warm else h)
                retm = ret[mb].reshape(-1)
                vloss = F.mse_loss(v, retm)
                logp_all = F.log_softmax(lg, -1)
                am = A[mb].reshape(-1)
                logp = logp_all.gather(1, am[:, None])[:, 0]
                ratio = torch.exp(logp - LP[mb].reshape(-1))
                ad = adv[mb].reshape(-1); ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                pg = -torch.min(ratio * ad, ratio.clamp(1 - CLIP, 1 + CLIP) * ad).mean()
                p_all = logp_all.exp()
                kl = (p_all * (logp_all - RLP[mb].reshape(-1, 4))).sum(1).mean()
                ent = -(p_all * logp_all).sum(1).mean()
                loss = VF * vloss if warm else pg + VF * vloss + beta * kl - ENT * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(list(pi.parameters()) + list(vh.parameters()), MAX_GRAD)
                opt.step()
                with torch.no_grad():
                    stats["pg"].append(pg.item()); stats["v"].append(vloss.item()); stats["kl"].append(kl.item())
                    stats["ent"].append(ent.item()); stats["gn"].append(gn.item())
                    stats["clipfrac"].append(((ratio - 1).abs() > CLIP).float().mean().item())
                    stats["approx_kl_old"].append(((ratio - 1) - torch.log(ratio)).mean().item())
        with torch.no_grad():
            yv = ret.reshape(-1); ev = 1 - torch.var(yv - V[:, :T].reshape(-1)) / (torch.var(yv) + 1e-8)
        t_train = time.time() - t_tr
        st["update"] = u = u + 1
        # ------------------------------------------------------------- logging / eval / ckpt
        m = {k: float(np.mean(v)) for k, v in stats.items()}
        sps = (st["env_steps"] - steps0) / (time.time() - t_run0)
        rc = list(recent)
        tr = dict(clear=float(np.mean([r["cleared"] for r in rc])) if rc else None,
                  items=float(np.mean([r["eaten"] for r in rc])) if rc else None,
                  deaths=float(np.mean([r["deaths"] for r in rc])) if rc else None, n=len(rc))
        rp = list(recent_pool)
        pl = dict(clear=float(np.mean([r["cleared"] for r in rp])) if rp else None,
                  items0=float(np.mean([r["items0"] for r in rp])) if rp else None, n=len(rp), size=envs.pool_size)
        row = dict(type="train", update=u, env_steps=st["env_steps"], episodes=st["episodes"], warm=warm, beta=beta,
                   sps=sps, t_roll=t_roll, t_wait=t_wait, t_ref=t_ref, t_train=t_train, t_eval=t_eval, expl_var=float(ev), rew_items=comp_acc[0], rew_score=comp_acc[1],
                   n_clear=comp_acc[2], n_death=comp_acc[3], end_bonus=comp_acc[4], stall_pen=float(comp_acc[5]) if STALL else 0.0, train_recent=tr, pool=pl, **m)
        jwrite(row); comp_acc[:] = 0
        if u % 5 == 0 or u <= 3:
            best = st["best"]; bs = "-" if not best else f"{best['clear_rate']:.2f}@{best['update']}"
            trs = "-" if not rc else f"clear {tr['clear']:.3f} items {tr['items']:.1f} deaths {tr['deaths']:.2f} (last {tr['n']})"
            log(f"upd {u}{' [value warm-up]' if warm else ''}  {st['env_steps'] / 1e6:.2f}M steps  {sps:.0f} steps/s "
                f"(roll {t_roll:.1f}s wait {t_wait:.1f}s ref {t_ref:.2f}s train {t_train:.1f}s eval {t_eval:.0f}s)  eps {st['episodes']}  normal-start games: {trs}  "
                f"pool {pl['size']} pool-start clear {('-' if pl['clear'] is None else format(pl['clear'], '.3f'))} "
                f"(from {('-' if pl['items0'] is None else format(pl['items0'], '.1f'))} items left, last {pl['n']})  "
                f"kl_ref {m['kl']:.3f} beta {beta:.3f}  ent {m['ent']:.3f}  clipfrac {m['clipfrac']:.3f}  "
                f"vloss {m['v']:.3f} EV {float(ev):.2f}  best eval {bs}")
        if pending is not None and pending[2].ready():
            finish_eval(pending); pending = None
        t_eval = time.time()
        if u % EVAL_EVERY == 0 and pending is None:
            pending = start_eval(u)
            if fev is not None:
                finish_eval(pending); pending = None
        t_eval = time.time() - t_eval
        if time.time() - t_ck >= CKPT_S:
            save(); t_ck = time.time()
    if job is not None:
        job.result()
    save()
    envs.close()
    if pending is not None:
        finish_eval(pending); pending = None
    if not st["evals"] or st["evals"][-1]["update"] != st["update"]:
        finish_eval(start_eval(st["update"])); save()
    if pool is not None:
        pool.close(); pool.join()
    if fev is not None:
        fev.close()
    final = dict(stopped=stopped, update=st["update"], env_steps=st["env_steps"], episodes=st["episodes"], best=st["best"],
                 last_eval=st["evals"][-1])
    (RUN / "final.json").write_text(json.dumps(final, indent=1))
    log(f"done ({stopped}) at update {st['update']}: best {st['best']}")


if __name__ == "__main__":
    main()
