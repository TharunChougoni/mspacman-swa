"""M10 distillation: M9 (SWAv2, runs/m9run/best.pt) -> M10Net (ALiBi + KV cache), DAgger style.

Init: every shape-matching M9 tensor (CNN, fc, blocks, ln_f, val, adv); pe_age dropped; ALiBi slopes = lam (0.05).
Rollouts: N = ENV_WORKERS x PER_WORKER envs (msclear.m10_env: games play through board clears; P_SNAP of starts
from the 87 board-2 snapshots), T steps per segment. Acting: M9 teacher with prob P_TEACH (per step, annealed
P_TEACH0 -> 0 over TEACH_ANNEAL updates), else the student sampled at TAU (KV-cache incremental path).
Labels: M9 logits adv/TAU at every visited state (M9's own 8-token window, reset at episode start like fasteval).
Loss: KL(teacher || student) on adv/TAU (T=TAU) + VAL_W * MSE(student val, teacher val) (dueling V of M9).
Training sample = 15-token history (2 layers x 7 + 1) clamped at the episode start = exact streaming semantics.
TF32 for training; evals strict fp32 (M10Evaluator, seeds 0..99, greedy). Stop: STOP file, eval >= TARGET twice,
PATIENCE evals without a new best, or MAX_UPDATES. Logs: runs/m10/driver via stdout, runs/m10/metrics.jsonl (tb_sync).
"""
from __future__ import annotations

import glob, json, os, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msclear.m10_env import VecGames10  # noqa: E402
from msclear.m10_fasteval import M10Evaluator  # noqa: E402
from msclear.m10_net import M10Net, from_swav2  # noqa: E402
from msclear.swa_net import SWAv2  # noqa: E402

E = os.environ.get
RUN = Path(E("RUN", "runs/m10")); TEACH = E("TEACHER", "runs/m9run/best.pt")
NW, PER, T = int(E("ENV_WORKERS", "14")), int(E("PER_WORKER", "3")), int(E("T", "128"))
TAU = float(E("TAU", "0.25")); LR = float(E("LR", "1e-4")); EPOCHS = int(E("EPOCHS", "2")); NMB = int(E("MINIBATCHES", "3"))
VAL_W = float(E("VAL_W", "0.001")); P_SNAP = float(E("P_SNAP", "0.3"))
P_TEACH0 = float(E("P_TEACH0", "0.5")); TEACH_ANNEAL = int(E("TEACH_ANNEAL", "300"))
EVAL_EVERY = int(E("EVAL_EVERY", "50")); TARGET = float(E("TARGET", "0.85")); PATIENCE = int(E("PATIENCE", "8"))
MAX_UPDATES = int(E("MAX_UPDATES", "5000")); MAX_STEPS = int(E("MAX_STEPS", "8000"))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def jwrite(row):
    with open(RUN / "metrics.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def save(obj, p):
    torch.save(obj, str(p) + ".tmp"); os.replace(str(p) + ".tmp", p)


def main():
    RUN.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    ck = torch.load(TEACH, map_location="cpu", weights_only=False); ARCH = ck["arch"]
    tea = SWAv2(**ARCH); tea.load_state_dict(ck["model"]); tea = tea.to(dev).eval().requires_grad_(False)
    net, loaded = from_swav2(ck["model"], ARCH); net = net.to(dev)
    W = ARCH["window"]; C = net.ctx; N = NW * PER
    opt = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5)
    st = dict(update=0, env_steps=0, best=None, evals=[], hit=0, since_best=0)
    if (RUN / "ckpt_last.pt").exists():
        c = torch.load(RUN / "ckpt_last.pt", map_location=dev, weights_only=False)
        net.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); st = c["state"]; log(f"resumed at update {st['update']}")
    else:
        log(f"init from {TEACH}: {len(loaded)} tensors loaded (missing: {sorted(set(net.state_dict()) - set(loaded))})")
    cfg = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str))}; cfg["ARCH"] = ARCH
    (RUN / "config.json").write_text(json.dumps(cfg, indent=1)); log(f"config {json.dumps(cfg)}")
    snaps = sorted(glob.glob("runs/multiboard/states/*_board2.pkl"))
    envs = VecGames10(NW, PER, max_steps=MAX_STEPS, p_snap=P_SNAP, snap_files=snaps, seed_off=st["update"] % 997)
    fev = M10Evaluator(workers=14); seeds = list(range(100))
    stack = torch.from_numpy(envs.reset()).to(dev)
    new = torch.ones(N, dtype=torch.bool, device=dev)
    H = torch.zeros(N, C - 1 + T, 4, 84, 84, dtype=torch.uint8, device=dev)
    ES = torch.zeros(N, T, dtype=torch.long, device=dev); estart = torch.full((N,), C - 1, dtype=torch.long, device=dev)
    TL = torch.zeros(N, T, 4, device=dev); TV = torch.zeros(N, T, device=dev)
    ttok = torch.zeros(N, W, net.d, device=dev); cache = net.new_cache(N, dev)
    recent = []; first = True

    def evaluate():
        net.eval(); t0 = time.time(); s, _ = fev.evaluate(net, seeds); net.train()
        s.update(update=st["update"], env_steps=st["env_steps"], secs=round(time.time() - t0)); jwrite(dict(type="eval", **s))
        st["evals"].append(s); better = st["best"] is None or (s["clear_rate"], s["eaten"]) > (st["best"]["clear_rate"], st["best"]["eaten"])
        if better:
            st["best"] = s; st["since_best"] = 0
            save(dict(model=net.state_dict(), arch=ARCH, kind="m10", step=st["update"]), RUN / "best.pt")
        else:
            st["since_best"] += 1
        st["hit"] = st["hit"] + 1 if s["clear_rate"] >= TARGET else 0
        log(f"EVAL update {st['update']} ({st['env_steps'] / 1e6:.2f}M): clear {s['clear_rate']:.2f} ± {s['clear_se']:.2f} items {s['eaten']:.1f} "
            f"deaths {s['deaths']:.2f} ({s['secs']}s){' *best*' if better else ''}")

    if st["update"] == 0 and st["best"] is None:
        evaluate()
    while st["update"] < MAX_UPDATES:
        t0 = time.time(); p_teach = max(0.0, P_TEACH0 * (1 - st["update"] / TEACH_ANNEAL))
        net.eval()
        with torch.no_grad():
            if not first:
                H[:, :C - 1] = H[:, T:T + C - 1].clone(); estart = (estart - T).clamp(min=0)
            first = False
            for t in range(T):
                pos = C - 1 + t
                H[:, pos] = stack; estart = torch.where(new, torch.full_like(estart, pos), estart); ES[:, t] = estart
                zt = tea.embed(stack)
                ttok = torch.where(new[:, None, None], zt[:, None].expand(-1, W, -1), torch.cat([ttok[:, 1:], zt[:, None]], 1))
                ht = tea.trunk(ttok); tl = tea.adv(ht) / TAU; TL[:, t] = tl; TV[:, t] = tea.val(ht)[:, 0]
                zs = net.embed(stack); net.prime(cache, new, zs); hs = net.step(cache, zs)
                a = torch.distributions.Categorical(logits=net.adv(hs) / TAU).sample()
                if p_teach > 0:
                    use_t = torch.rand(N, device=dev) < p_teach
                    a = torch.where(use_t, torch.distributions.Categorical(logits=tl).sample(), a)
                fr, done, results, starts = envs.step(a.cpu().numpy())
                new = torch.from_numpy(done).to(dev).bool()
                stack = torch.cat([stack[:, 1:], torch.from_numpy(fr).to(dev)[:, None]], 1)
                for i, s0 in starts.items():
                    stack[i] = torch.from_numpy(s0).to(dev)
                recent.extend(results)
        t_roll = time.time() - t0; t1 = time.time(); net.train()
        # training on the segment: index = clamp(pos-C+1 .. pos, ES)
        pos = torch.arange(T, device=dev) + C - 1
        idx = (pos[:, None] + torch.arange(-C + 1, 1, device=dev)[None])[None].expand(N, -1, -1)
        idx = torch.maximum(idx, ES[:, :, None])                                    # (N, T, C)
        stats = []
        for _ in range(EPOCHS):
            for mb in torch.randperm(N, device=dev).chunk(NMB):
                Hm = H[mb]; n = len(mb)
                z = net.embed(Hm.reshape(-1, 4, 84, 84)).view(n, C - 1 + T, -1)
                ctx = z[torch.arange(n, device=dev)[:, None, None], idx[mb]]       # (n, T, C, d)
                h = net.trunk_seq(ctx.reshape(n * T, C, -1))[:, -1]
                lg = net.adv(h) / TAU; tl = TL[mb].reshape(-1, 4)
                kl = F.kl_div(F.log_softmax(lg, -1), F.log_softmax(tl, -1), log_target=True, reduction="batchmean")
                vl = F.mse_loss(net.val(h)[:, 0], TV[mb].reshape(-1))
                loss = kl + VAL_W * vl
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                agree = (lg.argmax(-1) == tl.argmax(-1)).float().mean()
                stats.append((kl.item(), vl.item(), agree.item()))
        st["update"] += 1; st["env_steps"] += N * T
        kl, vl, ag = np.mean(stats, 0)
        recent = recent[-200:]
        nr = [r for r in recent if r["start"] == "normal"]; br = [r for r in recent if r["start"] == "board2"]
        row = dict(type="train", update=st["update"], env_steps=st["env_steps"], kl=kl, val_mse=vl, agree=ag, p_teach=p_teach,
                   t_roll=t_roll, t_train=time.time() - t1, sps=N * T / (time.time() - t0),
                   train_recent=dict(normal_clears=float(np.mean([r["clears"] for r in nr])) if nr else None,
                                     normal_score=float(np.mean([r["score"] for r in nr])) if nr else None,
                                     board2_clears=float(np.mean([r["clears"] for r in br])) if br else None))
        jwrite(row)
        if st["update"] % 5 == 0:
            log(f"upd {st['update']} {st['env_steps'] / 1e6:.2f}M kl {kl:.4f} agree {ag:.3f} vmse {vl:.2f} p_teach {p_teach:.2f} "
                f"sps {row['sps']:.0f} (roll {t_roll:.1f}s train {row['t_train']:.1f}s) recent {row['train_recent']}")
        if st["update"] % 25 == 0:
            save(dict(model=net.state_dict(), opt=opt.state_dict(), arch=ARCH, state=st), RUN / "ckpt_last.pt")
        if st["update"] % EVAL_EVERY == 0:
            evaluate()
            if st["hit"] >= 2 or st["since_best"] >= PATIENCE:
                log(f"stop: hit {st['hit']} since_best {st['since_best']}"); break
        if (RUN / "STOP").exists():
            log("STOP file"); break
    save(dict(model=net.state_dict(), opt=opt.state_dict(), arch=ARCH, state=st), RUN / "ckpt_last.pt")
    (RUN / "final.json").write_text(json.dumps(dict(update=st["update"], env_steps=st["env_steps"], best=st["best"])))
    envs.close(); fev.close(); log("done")


if __name__ == "__main__":
    main()
