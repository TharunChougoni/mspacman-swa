"""M7: PPO fine-tune of the M6 pixel SWA student (runs/m6/best.pt) with a KL leash to the frozen M6 policy.

Policy  pi(a|s) = softmax(adv(h) / TAU), h = SWA trunk output at the newest token (same net as M6;
        greedy argmax of pi == argmax of the M6 dueling Q, so evals/videos reuse m6eval unchanged).
Value   V(s) = MLP(h) (fresh). First VWARM updates train the value head only (trunk detached, actor frozen).
Loss    PPO-clip(CLIP) + VF * (V - R)^2 + beta(u) * KL(pi || pi_M6) - ENT * H(pi)
        beta anneals linearly BETA0 -> BETA1 over KL_ANNEAL updates (exact KL over the 4 actions).
Rollout N envs x T steps; the SWA token window (W=8) is cached per env during rollout and
        rebuilt exactly in training: one CNN pass per stack per minibatch (segment + 7 context stacks),
        windows gathered with clamping at episode starts (= Student's cache behaviour at reset).
Eval    every EVAL_EVERY updates: greedy pixel eval, 100 games, seeds 0..99, sticky 0.25 (same as M6 periodic
        evals) in a background CPU pool; best.pt by (clear rate, items). Final 300-game confirm is run by m7_run.sh.
Resume  RUN/ckpt_last.pt every CKPT_S s (+ at exit); the envs restart fresh episodes on resume (new seeds).
        Touch RUN/STOP for a clean stop: saves ckpt_last.pt and exits 3 (no final.json; m7_run.sh does not retry).
Tweak   at update ~2360 (13:40 IST 9 Oct): 15 evals flat (clear 0.02-0.07, items 136-143, deaths 2.0), kl_ref ~0.03,
        approx_kl_old ~5e-4 -> policy barely moving. Defaults changed: LR_PI 2e-5 -> 5e-5, death penalty 2.5 -> 5.0,
        BETA0 0.5 -> 0.25 and KL_ANNEAL 6000 -> 4000 (beta ~0.32 -> ~0.13 at the resume point, 0.05 from update ~4015).
"""
from __future__ import annotations

import collections
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
from msclear.ppo_env import VecGames  # noqa: E402
from msclear.swa_net import SWAv2  # noqa: E402

E = os.environ.get
RUN = Path(E("RUN", "runs/m7"))
INIT = E("INIT", "runs/m6/best.pt")
NW, PER = int(E("ENV_WORKERS", "14")), int(E("PER_WORKER", "3"))
T = int(E("T", "128"))
EPOCHS = int(E("EPOCHS", "3"))
NMB = int(E("MINIBATCHES", "3"))
GAMMA, LAM = float(E("GAMMA", "0.995")), float(E("GAE_LAMBDA", "0.95"))
CLIP = float(E("CLIP", "0.1"))
LR_PI, LR_V = float(E("LR_PI", "5e-5")), float(E("LR_V", "3e-4"))
VF, ENT = float(E("VF", "0.5")), float(E("ENT", "0.001"))
BETA0, BETA1, KL_ANNEAL = float(E("BETA0", "0.25")), float(E("BETA1", "0.05")), int(E("KL_ANNEAL", "4000"))
TAU = float(E("TAU", "0.25"))   # sampled pi at tau 1 eats ~100 items (greedy 134); tau 0.25 ~ greedy (131, measured)
VWARM = int(E("VWARM", "15"))
MAX_GRAD = float(E("MAX_GRAD", "0.5"))
RW = tuple(float(x) for x in E("RW", "0.1,0.0005,5.0,5.0").split(","))   # item, score, clear, death
UPDATES = int(E("UPDATES", "100000"))
DEADLINE = float(E("DEADLINE", "0")) or float("inf")
EVAL_EVERY = int(E("EVAL_EVERY", "150"))
EVAL_N = int(E("EVAL_N", "100"))
EVAL_SEED0 = int(E("EVAL_SEED0", "0"))
EVAL_WORKERS = int(E("EVAL_WORKERS", "6"))
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
    """SWAv2.head up to ln_f: (B, W, d) tokens -> h (B, d) at the newest token."""
    z = z + net.pe_age(torch.arange(z.shape[1], device=z.device))[None]
    for blk in net.blocks:
        z = blk(z)
    return net.ln_f(z[:, -1])


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
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
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
        log(f"init actor from {INIT} (step {ck0.get('step')}), fresh value head")
    cfg = dict(INIT=INIT, N=NW * PER, T=T, EPOCHS=EPOCHS, NMB=NMB, GAMMA=GAMMA, LAM=LAM, CLIP=CLIP, LR_PI=LR_PI, LR_V=LR_V,
               VF=VF, ENT=ENT, BETA0=BETA0, BETA1=BETA1, KL_ANNEAL=KL_ANNEAL, TAU=TAU, VWARM=VWARM, RW=RW, ARCH=ARCH,
               EVAL_EVERY=EVAL_EVERY, EVAL_N=EVAL_N, EVAL_SEED0=EVAL_SEED0, MAX_STEPS=MAX_STEPS)
    (RUN / "config.json").write_text(json.dumps(cfg, indent=1))
    N = NW * PER
    log(f"config {json.dumps(cfg)}")
    envs = VecGames(NW, PER, sticky=0.25, max_steps=MAX_STEPS, rw=RW, seed_off=st["update"] * 17 % 90_000)
    pool = m6eval.make_pool(EVAL_WORKERS)
    seeds = list(range(EVAL_SEED0, EVAL_SEED0 + EVAL_N))
    pending = None

    def save():
        st["wall"] = st.get("wall", 0.0)
        atomic_save(dict(pi=pi.state_dict(), vh=vh.state_dict(), opt=opt.state_dict(), arch=ARCH, state=st), last)

    def start_eval(u):
        cand = RUN / f"cand_{u:06d}.pt"
        atomic_save(dict(model=pi.state_dict(), arch=ARCH, step=u), cand)
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

    # ---- rollout state
    frames = torch.from_numpy(envs.reset()).to(dev)                  # (N, 84, 84)
    stack = frames[:, None].repeat(1, 4, 1, 1).contiguous()          # (N, 4, 84, 84)
    new = torch.ones(N, dtype=torch.bool, device=dev)
    H = torch.zeros(N, W - 1 + T, 4, 84, 84, dtype=torch.uint8, device=dev)
    estart = torch.full((N,), W - 1, dtype=torch.long, device=dev)
    tok = torch.zeros(N, W, ARCH["d"], device=dev); rtok = torch.zeros_like(tok)
    A = torch.zeros(N, T, dtype=torch.long, device=dev); LP = torch.zeros(N, T, device=dev)
    V = torch.zeros(N, T + 1, device=dev); R = torch.zeros(N, T, device=dev); D = torch.zeros(N, T, device=dev)
    RLP = torch.zeros(N, T, 4, device=dev); ES = torch.zeros(N, T, dtype=torch.long, device=dev)
    ar = torch.arange(W, device=dev)
    recent = collections.deque(maxlen=200)
    comp_acc = np.zeros(4); t_ck = time.time(); t_run0 = time.time(); steps0 = st["env_steps"]
    stopped = "updates"

    while st["update"] < UPDATES:
        if time.time() > DEADLINE:
            stopped = "deadline"; break
        if (RUN / "STOP").exists():
            save(); envs.close(); log(f"STOP file: saved ckpt_last.pt at update {st['update']}, exiting 3")
            (RUN / "STOP").unlink(missing_ok=True); pool.terminate(); os._exit(3)
        u = st["update"]; t_u = time.time()
        # ------------------------------------------------------------- rollout
        pi.eval()
        with torch.no_grad():
            for t in range(T):
                pos = W - 1 + t
                H[:, pos] = stack
                estart = torch.where(new, torch.full_like(estart, pos), estart)
                ES[:, t] = estart
                z = pi.embed(stack); rz = ref.embed(stack)
                nn_ = new[:, None, None]
                tok = torch.where(nn_, z[:, None].expand(-1, W, -1), torch.cat([tok[:, 1:], z[:, None]], 1))
                rtok = torch.where(nn_, rz[:, None].expand(-1, W, -1), torch.cat([rtok[:, 1:], rz[:, None]], 1))
                h = trunk(pi, tok)
                lg = logits_of(pi, h); V[:, t] = vh(h)
                RLP[:, t] = F.log_softmax(logits_of(ref, trunk(ref, rtok)), -1)
                a = torch.distributions.Categorical(logits=lg).sample()
                A[:, t] = a; LP[:, t] = F.log_softmax(lg, -1).gather(1, a[:, None])[:, 0]
                fr, rew, done, comp, results = envs.step(a.cpu().numpy())
                R[:, t] = torch.from_numpy(rew).to(dev); dn = torch.from_numpy(done).to(dev)
                D[:, t] = dn.float(); comp_acc += comp.sum(0)
                fr = torch.from_numpy(fr).to(dev)
                new = dn.bool()
                stack = torch.where(new[:, None, None, None], fr[:, None].expand(-1, 4, -1, -1),
                                    torch.cat([stack[:, 1:], fr[:, None]], 1)).contiguous()
                for r in results:
                    recent.append(r); st["episodes"] += 1
            # bootstrap value of the obs after the last step (token window of the next obs)
            z = pi.embed(stack)
            btok = torch.where(new[:, None, None], z[:, None].expand(-1, W, -1), torch.cat([tok[:, 1:], z[:, None]], 1))
            V[:, T] = vh(trunk(pi, btok))
        t_roll = time.time() - t_u
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
        # carry the last W-1 stacks as context for the next segment
        H[:, :W - 1] = H[:, T:T + W - 1].clone()
        estart = (estart - T).clamp(min=0)
        st["update"] = u = u + 1
        # ------------------------------------------------------------- logging / eval / ckpt
        m = {k: float(np.mean(v)) for k, v in stats.items()}
        sps = (st["env_steps"] - steps0) / (time.time() - t_run0)
        rc = list(recent)
        tr = dict(clear=float(np.mean([r["cleared"] for r in rc])) if rc else None,
                  items=float(np.mean([r["eaten"] for r in rc])) if rc else None,
                  deaths=float(np.mean([r["deaths"] for r in rc])) if rc else None, n=len(rc))
        row = dict(type="train", update=u, env_steps=st["env_steps"], episodes=st["episodes"], warm=warm, beta=beta,
                   sps=sps, t_roll=t_roll, t_train=t_train, expl_var=float(ev), rew_items=comp_acc[0], rew_score=comp_acc[1],
                   n_clear=comp_acc[2], n_death=comp_acc[3], train_recent=tr, **m)
        jwrite(row); comp_acc[:] = 0
        if u % 5 == 0 or u <= 3:
            best = st["best"]; bs = "-" if not best else f"{best['clear_rate']:.2f}@{best['update']}"
            trs = "-" if not rc else f"clear {tr['clear']:.3f} items {tr['items']:.1f} deaths {tr['deaths']:.2f} (last {tr['n']})"
            log(f"upd {u}{' [value warm-up]' if warm else ''}  {st['env_steps'] / 1e6:.2f}M steps  {sps:.0f} steps/s "
                f"(roll {t_roll:.1f}s train {t_train:.1f}s)  eps {st['episodes']}  sampled-policy games: {trs}  "
                f"kl_ref {m['kl']:.3f} beta {beta:.3f}  ent {m['ent']:.3f}  clipfrac {m['clipfrac']:.3f}  "
                f"vloss {m['v']:.3f} EV {float(ev):.2f}  best eval {bs}")
        if pending is not None and pending[2].ready():
            finish_eval(pending); pending = None
        if u % EVAL_EVERY == 0 and pending is None:
            pending = start_eval(u)
        if time.time() - t_ck >= CKPT_S:
            save(); t_ck = time.time()
    save()
    envs.close()
    if pending is not None:
        finish_eval(pending); pending = None
    if not st["evals"] or st["evals"][-1]["update"] != st["update"]:
        finish_eval(start_eval(st["update"])); save()
    pool.close(); pool.join()
    final = dict(stopped=stopped, update=st["update"], env_steps=st["env_steps"], episodes=st["episodes"], best=st["best"],
                 last_eval=st["evals"][-1])
    (RUN / "final.json").write_text(json.dumps(final, indent=1))
    log(f"done ({stopped}) at update {st['update']}: best {st['best']}")


if __name__ == "__main__":
    main()
