"""M23 distillation: student policy KL(M22 || student) + separate value net regression to M22's value, on 32-frame
chunks of the m23_collect shards. bf16 autocast, activation checkpointing, grad accumulation. Resumable from
OUT/ckpt.pt (written atomically every --ckpt_every steps). Metrics -> OUT/metrics.jsonl (+ stdout progress lines).
Usage: m23_train.py OUT --student cnn|vit --data DIR[:weight] [DIR[:weight] ...] [--init ckpt] [--steps N]
"""
import argparse, concurrent.futures as cf, json, math, os, pickle, random, sys, threading, queue, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, ".")
from msclear.m23_net import build, ARCHS
import cv2
cv2.setNumThreads(1)

p = argparse.ArgumentParser()
p.add_argument("out"); p.add_argument("--student", required=True); p.add_argument("--data", nargs="+", required=True)
p.add_argument("--init", default=None); p.add_argument("--steps", type=int, default=4000); p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--micro", type=int, default=32); p.add_argument("--accum", type=int, default=2); p.add_argument("--T", type=int, default=32)
p.add_argument("--resident", type=int, default=24); p.add_argument("--ckpt_every", type=int, default=250); p.add_argument("--val_shards", type=int, default=2)
p.add_argument("--warmup", type=int, default=300); p.add_argument("--vscale", type=float, default=10.0); p.add_argument("--minctx", type=int, default=8)
a = p.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); dev = "cuda"; T = a.T
torch.manual_seed(0); random.seed(0)

# ---------------- data
dirs = []
for spec in a.data:
    d, _, w = spec.partition(":"); files = [Path(d) / json.loads(l)["file"] for l in open(Path(d) / "index.jsonl")]
    dirs.append(dict(files=files, w=float(w or 1)))
val_files = dirs[0]["files"][:a.val_shards]
for d in dirs:
    d["files"] = [f for f in d["files"] if f not in val_files]


def load_shard(f):
    d = pickle.load(open(f, "rb")); d.pop("gray", None)
    n = np.diff(d["seg_off"]); d["ok"] = np.flatnonzero(n >= T); d["len"] = n; return d


def decode(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)   # stored as RGB bytes via imencode -> same order back


class Loader:
    def __init__(self, dirs, resident, seed):
        self.dirs, self.rng = dirs, random.Random(seed); self.ex = cf.ThreadPoolExecutor(6)
        self.res = []
        per = [max(1, round(resident * d["w"] / sum(x["w"] for x in dirs))) for d in dirs]
        for d, k in zip(dirs, per):
            for f in self.rng.sample(d["files"], min(k, len(d["files"]))):
                self.res.append((d, load_shard(f)))
        self.q = queue.Queue(4); self.n = 0; threading.Thread(target=self.run, daemon=True).start()

    def batch(self, B):
        idx = []
        for _ in range(B):
            d, s = self.res[self.rng.randrange(len(self.res))]
            if not len(s["ok"]):
                continue
            g = int(s["ok"][self.rng.randrange(len(s["ok"]))]); n = int(s["len"][g]); st = self.rng.randrange(n - T + 1)
            idx.append((s, int(s["seg_off"][g]) + st, bool(s["seg_start"][g]) and st == 0))
        fr = list(self.ex.map(lambda x: np.stack([decode(b) for b in x[0]["col"][x[1]:x[1] + T]]), idx))
        tl = np.stack([x[0]["tl"][x[1]:x[1] + T] for x in idx]).astype(np.float32); tv = np.stack([x[0]["tv"][x[1]:x[1] + T] for x in idx])
        m = np.ones((len(idx), T), np.float32)
        for j, x in enumerate(idx):
            if not x[2]:
                m[j, :a.minctx] = 0
        return torch.from_numpy(np.stack(fr)), torch.from_numpy(tl), torch.from_numpy(tv), torch.from_numpy(m)

    def rotate(self):                                    # swap one resident shard for a random one of the same dir
        j = self.rng.randrange(len(self.res)); d = self.res[j][0]
        self.res[j] = (d, load_shard(self.rng.choice(d["files"])))

    def run(self):
        while True:
            self.q.put(self.batch(a.micro)); self.n += 1
            if self.n % 40 == 0:
                self.rotate()


# ---------------- models
pi = build(a.student, ckpt=True).to(dev); vnet = build("value", ckpt=True).to(dev)
opt = torch.optim.AdamW([dict(params=pi.parameters()), dict(params=vnet.parameters())], lr=a.lr, weight_decay=0.05, betas=(0.9, 0.95))
step = 0; ck_path = OUT / "ckpt.pt"; best = None
if ck_path.exists():
    c = torch.load(ck_path, map_location="cpu", weights_only=False)
    pi.load_state_dict(c["pi"]); vnet.load_state_dict(c["vnet"]); opt.load_state_dict(c["opt"]); step = c["step"]; best = c.get("best")
    print(f"RESUME {ck_path} step={step}", flush=True)
elif a.init:
    c = torch.load(a.init, map_location="cpu", weights_only=False); pi.load_state_dict(c["pi"]); vnet.load_state_dict(c["vnet"])
    print(f"INIT from {a.init}", flush=True)
else:                                                    # input normalisation from ~4k frames of 8 shards
    fs = dirs[0]["files"]; px = []
    for f in fs[:: max(1, len(fs) // 8)][:8]:
        d = pickle.load(open(f, "rb")); px += [decode(b) for b in d["col"][::32][:500]]
    px = torch.from_numpy(np.stack(px)).float() / 255.0; mu, sd = px.mean(0), px.std(0) + 0.05
    for net in (pi, vnet):
        net.mu.copy_(mu.to(dev)); net.sd.copy_(sd.to(dev))
    print(f"NORM from {len(px)} frames: mean {mu.mean():.3f} sd {sd.mean():.3f}", flush=True)
print(f"TRAIN {a.student} params pi={sum(x.numel() for x in pi.parameters()):,} vnet={sum(x.numel() for x in vnet.parameters()):,} "
      f"data={[ (str(d['files'][0].parent), len(d['files']), d['w']) for d in dirs]} val={len(val_files)} steps={a.steps}", flush=True)
arch = dict(pi=pi.arch, vnet=vnet.arch)


def lr_at(s):
    if s < a.warmup:
        return a.lr * (s + 1) / a.warmup
    return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, a.steps - a.warmup)))))


def losses(fr, tl, tv, m, train=True):
    fr, tl, tv, m = fr.to(dev, non_blocking=True), tl.to(dev), tv.to(dev), m.to(dev)
    with torch.autocast("cuda", torch.bfloat16):
        sl = pi(fr).float(); sv = vnet(fr).float()[..., 0]
    tp = F.log_softmax(tl, -1); sp = F.log_softmax(sl, -1)
    kl = (tp.exp() * (tp - sp)).sum(-1); ms = m.sum().clamp(min=1)
    kl = (kl * m).sum() / ms; vl = (F.smooth_l1_loss(sv, tv / a.vscale, reduction="none") * m).sum() / ms
    agree = (((sl.argmax(-1) == tl.argmax(-1)).float() * m).sum() / ms)
    verr = ((sv * a.vscale - tv).abs() * m).sum() / ms
    return kl, vl, agree, verr


def save(path, extra={}):
    tmp = str(path) + ".tmp"
    torch.save(dict(pi=pi.state_dict(), vnet=vnet.state_dict(), opt=opt.state_dict(), step=step, arch=arch, best=best, **extra), tmp)
    os.replace(tmp, path)


val = None


def validate():
    global val
    if val is None:
        vl_ = Loader([dict(files=val_files, w=1)], len(val_files), 123); val = [vl_.batch(a.micro) for _ in range(12)]
    pi.eval(); vnet.eval(); r = []
    with torch.no_grad():
        for b in val:
            r.append([x.item() for x in losses(*b, train=False)])
    pi.train(); vnet.train(); r = np.mean(r, 0); return dict(val_kl=r[0], val_vloss=r[1], val_agree=r[2], val_verr=r[3])


L = Loader(dirs, a.resident, 1 + step); mlog = open(OUT / "metrics.jsonl", "a"); pi.train(); vnet.train()
t0 = time.time(); s0 = step; acc = []
while step < a.steps:
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    opt.zero_grad(set_to_none=True)
    for _ in range(a.accum):
        kl, vl, ag, ve = losses(*L.q.get()); ((kl + vl) / a.accum).backward(); acc.append([kl.item(), vl.item(), ag.item(), ve.item()])
    gn = torch.nn.utils.clip_grad_norm_(list(pi.parameters()) + list(vnet.parameters()), 1.0)
    opt.step(); step += 1
    if step % 25 == 0:
        r = np.mean(acc, 0); acc = []; el = time.time() - t0; rate = (step - s0) / el; eta = (a.steps - step) / max(rate, 1e-9)
        m = dict(type="train", step=step, kl=float(r[0]), vloss=float(r[1]), agree=float(r[2]), verr=float(r[3]), lr=lr_at(step),
                 gnorm=float(gn), steps_per_s=rate, frames_per_s=rate * a.micro * a.accum * T, mem_gb=torch.cuda.max_memory_allocated() / 2**30, t=time.time())
        print(f"STEP {step}/{a.steps} kl={r[0]:.4f} agree={r[2]:.3f} vloss={r[1]:.4f} verr={r[3]:.2f} lr={m['lr']:.2e} "
              f"{m['frames_per_s']:.0f} fr/s mem={m['mem_gb']:.2f}GB ETA={eta / 60:.1f}min", flush=True)
        mlog.write(json.dumps(m) + "\n"); mlog.flush()
    if step % a.ckpt_every == 0 or step == a.steps:
        v = validate(); v.update(type="val", step=step, t=time.time()); mlog.write(json.dumps(v) + "\n"); mlog.flush()
        print(f"VAL {step} kl={v['val_kl']:.4f} agree={v['val_agree']:.3f} verr={v['val_verr']:.2f}", flush=True)
        if best is None or v["val_kl"] < best:
            best = v["val_kl"]; save(OUT / "best.pt")
        save(ck_path)
(OUT / "DONE").write_text(str(step)); print("TRAIN DONE", flush=True)
