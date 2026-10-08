"""M6 training: distil the HRA teacher into the SWA-v2 pixel student.

loss = CE(Q, a_HRA) + LAM_M * DQfD large margin  [max_a (Q(s,a) + MARGIN*[a != a_HRA]) - Q(s,a_HRA)]
An EMA of the weights (decay EMA) is kept next to the raw weights; validation agreement is
logged for both, pixel evals and best.pt use the EMA weights.

Resumable: RUN/ckpt_last.pt every CKPT_S s (model, EMA, optimizer, LR schedule, step, torch RNG,
best, eval/val history). Batch k is a pure function of (SEED, k), so a resumed run sees the same
sample stream. A pixel eval (EVAL_N games, sticky 0.25, seeds EVAL_SEED0..) runs every EVAL_EVERY
steps in a CPU pool while training continues; best.pt = best EMA eval so far (clear rate, then items).
Stops at STEPS or at DEADLINE (unix time), whichever comes first; writes RUN/final.json.
"""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msclear import m6eval  # noqa: E402
from msclear.m6data import Windows  # noqa: E402
from msclear.swa_net import SWAv2, load_stem  # noqa: E402

E = os.environ.get
RUN = Path(E("RUN", "runs/m6/bc"))
PACKS = [Path(p) for p in E("PACKS", "runs/m6/demos/pack").split(",")]
PACK_WEIGHTS = [float(x) for x in E("PACK_WEIGHTS", "").split(",")] if E("PACK_WEIGHTS") else None
INIT = E("INIT", "")
ARCHIVE = str(Path.home() / "HermesWorkspace/_archive/mspacman-rainbow-2026-10-08/ckpts/swa_rtc_attn_c3_300k.pt")
STEPS = int(E("STEPS", "60000"))
B = int(E("BATCH", "128"))
W = int(E("W", "16"))
LR = float(E("LR", "3e-4"))
WD = float(E("WD", "0.01"))
WARMUP = int(E("WARMUP", "1000"))
LR_MIN = float(E("LR_MIN", "0.05"))
MARGIN = float(E("MARGIN", "0.8"))
LAM_M = float(E("LAM_M", "1.0"))
EMA = float(E("EMA", "0.999"))
CKPT_S = float(E("CKPT_S", "300"))
VAL_EVERY = int(E("VAL_EVERY", "2000"))
VAL_N = int(E("VAL_N", "16384"))
EVAL_EVERY = int(E("EVAL_EVERY", "8000"))
EVAL_N = int(E("EVAL_N", "100"))
EVAL_SEED0 = int(E("EVAL_SEED0", "0"))
EVAL_WORKERS = int(E("EVAL_WORKERS", "8"))
LOADER = int(E("LOADER_WORKERS", "6"))
DEADLINE = float(E("DEADLINE", "0")) or float("inf")
SEED = int(E("SEED", "0"))
TEMP_MAX = int(E("TEMP_MAX", "88"))
ARCH = dict(n_actions=4, d=512, n_heads=8, n_layers=2, window=W, lam=0.05)


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


def gpu_temp():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=5).stdout.split()[0])
    except Exception:
        return -1


def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    p = min(1.0, (step - WARMUP) / max(1, STEPS - WARMUP))
    return LR * (LR_MIN + (1 - LR_MIN) * 0.5 * (1 + math.cos(math.pi * p)))


def losses(Q, lab):
    ce = F.cross_entropy(Q, lab)
    marg = (Q + MARGIN * (1 - F.one_hot(lab, Q.shape[1]))).max(1).values - Q.gather(1, lab[:, None])[:, 0]
    return ce, marg.mean()


@torch.no_grad()
def validate(nets, val, vset, dev):
    """Action agreement with HRA on held-out demo shards (EMA and raw), split by game outcome / phase."""
    out = {}
    for name, net in nets.items():
        net.eval()
        hit, cl, ph = [], [], []
        ce_sum = 0.0
        for pi, rows in vset:
            meta = val.packs[pi].meta
            for k in range(0, len(rows), 256):
                t = rows[k:k + 256]
                f, lab, _ = val.gather(pi, t)
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    Q = net.forward_seq(torch.from_numpy(f).to(dev)).float()
                lab_t = torch.from_numpy(lab).to(dev)
                ce_sum += float(F.cross_entropy(Q, lab_t, reduction="sum"))
                hit.append((Q.argmax(1) == lab_t).cpu().numpy())
                cl.append(meta["cleared"][t]); ph.append(meta["items"][t])
        hit, cl, ph = np.concatenate(hit), np.concatenate(cl), np.concatenate(ph)
        out[name] = dict(agree=float(hit.mean()), ce=ce_sum / len(hit), agree_cleared=float(hit[cl].mean()) if cl.any() else None,
                         agree_failed=float(hit[~cl].mean()) if (~cl).any() else None,
                         agree_endgame=float(hit[ph <= 20].mean()) if (ph <= 20).any() else None)
        net.train()
    return out


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    dev = torch.device(E("DEV", "cuda"))
    torch.backends.cudnn.benchmark = True
    net = SWAv2(**ARCH).to(dev)
    ema = copy.deepcopy(net).requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.98))
    st = dict(step=0, best=None, evals=[], vals=[], started=time.time())
    last = RUN / "ckpt_last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        torch.set_rng_state(ck["rng_cpu"].cpu()); torch.cuda.set_rng_state(ck["rng_cuda"].cpu()) if dev.type == 'cuda' else None
        st = ck["state"]
        log(f"resumed {last} at step {st['step']}")
    elif INIT == "archive":
        keys = load_stem(net, ARCHIVE); ema.load_state_dict(net.state_dict())
        log(f"init from archived RTCNet ({len(keys)} tensors: conv/fc/attention; 9-way head skipped)")
    elif INIT:
        ck = torch.load(INIT, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"]); ema.load_state_dict(ck["model"])
        log(f"init from {INIT}")
    else:
        log("init: random")
    n_par = sum(p.numel() for p in net.parameters())
    data = Windows(PACKS, W, B, "train", seed=SEED, weights=PACK_WEIGHTS)
    val = Windows(PACKS, W, B, "val")
    vset = val.fixed(VAL_N)
    log(f"params {n_par:,}  train rows {[len(r) for r in data.rows]}  val rows {sum(len(r) for _, r in vset)}  "
        f"steps {st['step']}->{STEPS}  batch {B}  window {W}  lr {LR}  margin {MARGIN} x {LAM_M}  ema {EMA}")
    with open(RUN / "config.json", "w") as f:
        json.dump(dict(PACKS=[str(p) for p in PACKS], INIT=INIT, STEPS=STEPS, B=B, W=W, LR=LR, WD=WD, MARGIN=MARGIN,
                       LAM_M=LAM_M, EMA=EMA, ARCH=ARCH, PACK_WEIGHTS=PACK_WEIGHTS), f, indent=1)
    loader = torch.utils.data.DataLoader(data, batch_size=None, sampler=range(st["step"], STEPS), num_workers=LOADER,
                                         pin_memory=dev.type == 'cuda', prefetch_factor=4, persistent_workers=False)
    pool = m6eval.make_pool(EVAL_WORKERS)
    seeds = list(range(EVAL_SEED0, EVAL_SEED0 + EVAL_N))
    pending = None                                   # (step, cand_path, AsyncResult, t0)

    def save():
        atomic_save(dict(model=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), arch=ARCH,
                         rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state() if dev.type == 'cuda' else torch.get_rng_state(), state=st), last)

    def start_eval(step):
        cand = RUN / f"cand_{step:07d}.pt"
        atomic_save(dict(model=ema.state_dict(), arch=ARCH, step=step), cand)
        return (step, cand, pool.map_async(m6eval.play, [(str(cand), s, 0.25) for s in seeds]), time.time())

    def finish_eval(p):
        step, cand, ar, t0 = p
        res = sorted(ar.get(), key=lambda r: r["seed"])
        s = m6eval.summarize(res); s.update(step=step, secs=round(time.time() - t0))
        st["evals"].append(s); jwrite(dict(type="eval", **s))
        key = (s["clear_rate"], s["eaten"])
        better = st["best"] is None or key > (st["best"]["clear_rate"], st["best"]["eaten"])
        if better:
            st["best"] = s; shutil.copyfile(cand, RUN / "best.pt.tmp"); os.replace(RUN / "best.pt.tmp", RUN / "best.pt")
        cand.unlink(missing_ok=True)
        log(f"EVAL step {step}: clear {s['clear_rate']:.2f} ± {s['clear_se']:.2f}  items {s['eaten']:.1f}  "
            f"score {s['score']:.0f}  deaths {s['deaths']:.2f}  steps {s['steps']:.0f}  actions {s['action_frac']}  "
            f"({s['secs']}s){'  *best*' if better else ''}")

    t_ck = t_log = time.time(); t0 = time.time(); s0 = st["step"]
    acc_t = torch.zeros(4, device=dev); n_acc = 0
    temp = gpu_temp(); stopped = "steps"
    net.train()
    for f, lab, _q in loader:
        step = st["step"]
        if time.time() > DEADLINE:
            stopped = "deadline"; break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        f = f.to(dev, non_blocking=True); lab = lab.to(dev, non_blocking=True)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            Q = net.forward_seq(f)
        Q = Q.float()
        ce, marg = losses(Q, lab)
        loss = ce + LAM_M * marg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(EMA, (1 + step) / (10 + step))
            for pe, p in zip(ema.parameters(), net.parameters()):
                pe.lerp_(p, 1 - d)
        st["step"] = step = step + 1
        acc_t = acc_t + torch.stack([loss.detach(), ce.detach(), marg.detach(), (Q.argmax(1) == lab).float().mean()])
        n_acc += 1

        if pending is not None and pending[2].ready():
            finish_eval(pending); pending = None
        if step % VAL_EVERY == 0 or step == STEPS:
            v = validate(dict(ema=ema, raw=net), val, vset, dev)
            st["vals"].append(dict(step=step, **v)); jwrite(dict(type="val", step=step, **v))
            log(f"VAL step {step}: agree ema {v['ema']['agree']:.3f} raw {v['raw']['agree']:.3f}  "
                f"(cleared {v['ema']['agree_cleared']}, failed {v['ema']['agree_failed']}, endgame {v['ema']['agree_endgame']})  "
                f"ce ema {v['ema']['ce']:.3f}")
        if step % EVAL_EVERY == 0 and pending is None:
            pending = start_eval(step)
        now = time.time()
        if now - t_log >= 30:
            sps = (step - s0) / (now - t0)
            temp = gpu_temp()
            m = (acc_t / max(n_acc, 1)).tolist()
            row = dict(type="train", step=step, loss=m[0], ce=m[1], margin=m[2], acc=m[3], lr=lr_at(step),
                       steps_per_s=sps, gpu_temp=temp)
            jwrite(row)
            best = st["best"]
            best_s = "-" if not best else f"{best['clear_rate']:.2f}@{best['step']}"
            log(f"step {step}/{STEPS}  {sps * B:.0f} samples/s  loss {row['loss']:.3f} (ce {row['ce']:.3f}, margin {row['margin']:.3f})  "
                f"train acc {row['acc']:.3f}  lr {row['lr']:.2e}  GPU {temp}C  "
                f"best eval {best_s}  "
                f"ETA {(STEPS - step) / max(sps, 1e-9) / 60:.0f} min")
            acc_t = torch.zeros(4, device=dev); n_acc = 0; t_log = now
            while temp >= TEMP_MAX:
                log(f"GPU at {temp}C >= {TEMP_MAX}C: pausing 30 s"); time.sleep(30); temp = gpu_temp()
        if now - t_ck >= CKPT_S:
            save(); t_ck = now
    save()
    if pending is not None:
        finish_eval(pending); pending = None
    if not st["evals"] or st["evals"][-1]["step"] != st["step"]:
        finish_eval(start_eval(st["step"]))
        save()
    pool.close(); pool.join()
    final = dict(stopped=stopped, step=st["step"], best=st["best"], last_eval=st["evals"][-1],
                 last_val=st["vals"][-1] if st["vals"] else None, hours=(time.time() - st["started"]) / 3600)
    (RUN / "final.json").write_text(json.dumps(final, indent=1))
    log(f"done ({stopped}) at step {st['step']}: best {st['best']}")


if __name__ == "__main__":
    main()
