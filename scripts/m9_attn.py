"""M9 attention analysis of runs/m8/best.pt (analysis only).

1. Collect token windows (8 x 512, the Student cache) from greedy M8 games on seeds 1000.. (not eval seeds).
2. Attention stats per layer/head: entropy, mass on the newest token, mass by age, top-1/top-2 mass.
3. Ablations scored offline on those windows (greedy-action agreement with the unablated net, mean KL at tau 0.25):
   head masking (each of 16 heads), self-only attention per layer, drop a layer, window truncation,
   pooled old tokens (temporal summary), top-k attention, low-rank (SVD) projections/MLP.
4. A few ablations as real 100-game greedy evals (seeds 0..99, sticky 0.25, M6/M7/M8 protocol).
Writes runs/m9/attn.json. Env: GAMES (24), EVAL_CFGS (comma list), EVAL_N (100), WORKERS (16).
"""
from __future__ import annotations

import collections
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CK = os.environ.get("CKPT", "runs/m8/best.pt")
OUT = Path("runs/m9"); OUT.mkdir(parents=True, exist_ok=True)
E = os.environ.get


def load(dev="cpu"):
    from msclear.swa_net import SWAv2
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    net = SWAv2(**ck["arch"]); net.load_state_dict(ck["model"]); net.eval()
    return net.to(dev)


# ------------------------------------------------------------------ ablatable trunk
def attn(a, x, cfg, layer, want=None):
    """RecencyCausalAttention with ablations. x: (B, t, d) (already LN'd). Returns (y, probs)."""
    b, t, d = x.shape
    q, k, v = a.qkv(x).view(b, t, 3, a.n_heads, a.d_head).unbind(dim=2)
    q, k, v = (z.transpose(1, 2) for z in (q, k, v))
    s = q @ k.transpose(-2, -1) / math.sqrt(a.d_head)
    idx = torch.arange(t, device=x.device)
    age = idx[:, None] - idx[None, :]
    s = s - a.lam * age.clamp(min=0).to(s.dtype)
    mask = age < 0
    if cfg.get("self_only") and layer in cfg["self_only"]:
        mask = age != 0
    s = s.masked_fill(mask, torch.finfo(s.dtype).min)
    if cfg.get("topk"):
        kk = min(cfg["topk"], t)
        thr = s.topk(kk, -1).values[..., -1:]
        s = s.masked_fill(s < thr, torch.finfo(s.dtype).min)
    p = torch.softmax(s, -1)
    y = p @ v                                                 # (b, H, t, dh)
    hm = cfg.get("head_mask")
    if hm is not None and hm[0] == layer:
        y = y.clone(); y[:, hm[1]] = 0
    y = a.out(y.transpose(1, 2).reshape(b, t, d))
    if want is not None:
        want.append(p.detach())
    return y


def lowrank(net, r):
    """Copy of net with every Linear in the blocks replaced by its rank-r SVD (proxy for a slimmer width)."""
    import copy
    n2 = copy.deepcopy(net)
    for blk in n2.blocks:
        for lin in (blk.attn.qkv, blk.attn.out, blk.mlp[0], blk.mlp[2]):
            U, S, Vh = torch.linalg.svd(lin.weight.data, full_matrices=False)
            lin.weight.data = (U[:, :r] * S[:r]) @ Vh[:r]
    return n2


def trunk_ab(net, z, cfg, want=None):
    W = net.window
    keep = cfg.get("window", W)
    if cfg.get("pool_old"):                                   # tokens older than the newest pool_old -> their mean
        n = cfg["pool_old"]
        old = z[:, : W - n].mean(1, keepdim=True).expand(-1, W - n, -1)
        z = torch.cat([old, z[:, W - n:]], 1)
    z = z[:, W - keep:] + net.pe_age(torch.arange(W - keep, W, device=z.device))[None]
    for li, blk in enumerate(net.blocks):
        if cfg.get("drop_layer") == li:
            continue
        z = z + attn(blk.attn, blk.ln1(z), cfg, li, want)
        z = z + blk.mlp(blk.ln2(z))
    return net.ln_f(z[:, -1])


def logits(net, z, cfg):
    return net.adv(trunk_ab(net, z, cfg))


# ------------------------------------------------------------------ data collection (CPU workers)
_N = {}


def _init():
    torch.set_num_threads(1)


def collect(seed):
    from msclear.demos import Student
    from msclear.pixenv import PixelGame
    net = _N.setdefault("net", load())
    st = Student(net); g = PixelGame(sticky=0.25); stack = g.reset(seed); st.reset()
    out, items = [], []
    while not g.done:
        a = st.act(stack)
        if g.steps % 3 == 0:
            out.append(torch.stack(list(st.z)).half().numpy()); items.append(g.items_left())
        stack, *_ = g.step(a)
    return np.stack(out), np.array(items, np.int16), g.result(seed)


def play(args):
    cfg, seed = args
    from msclear.pixenv import PixelGame
    key = json.dumps(cfg, sort_keys=True)
    if _N.get("key") != key:
        net = load()
        if cfg.get("rank"):
            net = lowrank(net, cfg["rank"])
        _N.update(key=key, anet=net)
    net = _N["anet"]
    g = PixelGame(sticky=0.25); stack = g.reset(seed); zq = None
    with torch.no_grad():
        while not g.done:
            z = net.embed(torch.from_numpy(np.ascontiguousarray(stack))[None])[0]
            zq = collections.deque([z] * net.window, maxlen=net.window) if zq is None else (zq.append(z) or zq)
            a = int(logits(net, torch.stack(list(zq))[None], cfg)[0].argmax())
            stack, *_ = g.step(a)
    return g.result(seed)


def main():
    t0 = time.time()
    res = {}
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(int(E("WORKERS", "16")), initializer=_init)
    games = int(E("GAMES", "24"))
    data = pool.map(collect, range(1000, 1000 + games))
    Z = torch.from_numpy(np.concatenate([d[0] for d in data])).float()
    ITEMS = np.concatenate([d[1] for d in data])
    res["collect"] = dict(games=games, windows=len(Z), clears=int(sum(d[2]["cleared"] for d in data)),
                          secs=round(time.time() - t0))
    print("collected", res["collect"], flush=True)
    dev = torch.device("cuda")
    net = load(dev); Z = Z.to(dev)
    with torch.no_grad():
        # ---------------- attention stats
        ps = []
        base = []
        for k in range(0, len(Z), 2048):
            want = []
            base.append(F.log_softmax(net.adv(trunk_ab(net, Z[k:k + 2048], {}, want)) / 0.25, -1))
            ps.append([w for w in want])
        base = torch.cat(base); a0 = base.argmax(1)
        stats = {}
        for li in range(len(net.blocks)):
            P = torch.cat([p[li] for p in ps])                 # (N, H, t, t)
            Pn = P[:, :, -1]                                   # newest query: (N, H, t), oldest first
            ent = -(Pn.clamp_min(1e-12).log() * Pn).sum(-1)    # nats, max ln 8 = 2.08
            srt = Pn.sort(-1, descending=True).values
            stats[f"layer{li}"] = dict(
                entropy_per_head=ent.mean(0).tolist(), entropy_mean=float(ent.mean()), entropy_max=math.log(Pn.shape[-1]),
                mass_newest_per_head=Pn[..., -1].mean(0).tolist(),
                mass_by_age_newest_query=Pn.mean((0, 1)).flip(-1).tolist(),     # index = age 0..7
                top1_mass=float(srt[..., 0].mean()), top2_mass=float(srt[..., :2].sum(-1).mean()),
                top3_mass=float(srt[..., :3].sum(-1).mean()),
                frac_weights_below_0p02=float((Pn < 0.02).float().mean()),
                endgame_mass_by_age=Pn[torch.from_numpy(ITEMS <= 30).to(dev)].mean((0, 1)).flip(-1).tolist(),
            )
        res["attention"] = stats
        # ---------------- offline ablations
        def score(cfg, n2=None):
            n_ = n2 or net
            lp = torch.cat([F.log_softmax(n_.adv(trunk_ab(n_, Z[k:k + 4096], cfg)) / 0.25, -1) for k in range(0, len(Z), 4096)])
            agree = (lp.argmax(1) == a0).float()
            kl = (base.exp() * (base - lp)).sum(1)
            eg = torch.from_numpy(ITEMS <= 30).to(dev)
            return dict(agree=float(agree.mean()), agree_endgame=float(agree[eg].mean()), kl=float(kl.mean()))
        ab = {}
        for li in range(2):
            for h in range(8):
                ab[f"mask_L{li}H{h}"] = score(dict(head_mask=(li, h)))
        ab["self_only_L0"] = score(dict(self_only=[0]))
        ab["self_only_L1"] = score(dict(self_only=[1]))
        ab["self_only_both (no temporal mixing)"] = score(dict(self_only=[0, 1]))
        ab["drop_layer0"] = score(dict(drop_layer=0))
        ab["drop_layer1"] = score(dict(drop_layer=1))
        for w in (6, 4, 3, 2, 1):
            ab[f"window{w}"] = score(dict(window=w))
        for n in (4, 2):
            ab[f"pool_old_keep{n}"] = score(dict(pool_old=n))
        for k in (1, 2, 3, 4):
            ab[f"topk{k}"] = score(dict(topk=k))
        for r in (256, 128, 64):
            ab[f"rank{r}"] = score({}, lowrank(net, r))
        res["offline_ablations"] = ab
        print(json.dumps(ab, indent=0), flush=True)
    # ---------------- real games for selected ablations
    cfgs = dict(base={}, window4=dict(window=4), window2=dict(window=2), window1=dict(window=1),
                pool_old_keep4=dict(pool_old=4), topk2=dict(topk=2), self_only_L0=dict(self_only=[0]),
                rank128=dict(rank=128))
    pick = E("EVAL_CFGS", ",".join(cfgs)).split(",")
    n = int(E("EVAL_N", "100"))
    from msclear.hra import summarize
    ev = {}
    for name in pick:
        t1 = time.time()
        r = pool.map(play, [(cfgs[name], s) for s in range(n)])
        s = summarize(r); s["secs"] = round(time.time() - t1); ev[name] = s
        print(f"EVAL {name}: clear {s['clear_rate']:.2f} ± {s['clear_se']:.2f} items {s['eaten']:.1f} deaths {s['deaths']:.2f} ({s['secs']}s)", flush=True)
        res["game_evals"] = ev
        (OUT / "attn.json").write_text(json.dumps(res, indent=1))
    pool.close(); pool.join()
    res["secs"] = round(time.time() - t0)
    (OUT / "attn.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
