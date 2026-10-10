"""M9 profile of the SWA-v2 student (analysis only; no default code path changed).

Times every component of the net on the RTX 3060 Laptop (GPU) and the i7-12700H (CPU, 1 thread as in evals),
at the batch shapes PPO/eval actually use, plus env stepping, and an exact "last-token-only" trunk variant.
Writes runs/m9/profile.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msclear.swa_net import SWAv2  # noqa: E402

OUT = Path("runs/m9"); OUT.mkdir(parents=True, exist_ok=True)
CK = "runs/m8/best.pt"


def trunk(net, z):
    z = z + net.pe_age(torch.arange(z.shape[1], device=z.device))[None]
    for blk in net.blocks:
        z = blk(z)
    return net.ln_f(z[:, -1])


def trunk_last(net, z):
    """Exact: the last block only needs its newest query (causal attn, only h[-1] is read)."""
    z = z + net.pe_age(torch.arange(z.shape[1], device=z.device))[None]
    for blk in net.blocks[:-1]:
        z = blk(z)
    blk = net.blocks[-1]; a = blk.attn
    b, t, d = z.shape
    x = blk.ln1(z)
    W, B_ = a.qkv.weight, a.qkv.bias
    q = F.linear(x[:, -1:], W[:d], B_[:d]).view(b, 1, a.n_heads, a.d_head).transpose(1, 2)
    kv = F.linear(x, W[d:], B_[d:]).view(b, t, 2, a.n_heads, a.d_head)
    k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
    s = (q @ k.transpose(-2, -1)) / a.d_head ** 0.5
    age = torch.arange(t - 1, -1, -1, device=z.device, dtype=s.dtype)
    s = s - a.lam * age
    y = (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(b, 1, d)
    h = z[:, -1:] + a.out(y)
    h = h + blk.mlp(blk.ln2(h))
    return net.ln_f(h[:, 0])


def tm(fn, n=50, warm=10, cuda=True):
    for _ in range(warm):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3     # ms


def macs():
    d, W = 512, 8
    stem = 400 * 32 * 4 * 64 + 81 * 64 * 32 * 16 + 49 * 64 * 64 * 9 + 3136 * d
    per_tok_layer = d * 3 * d + d * d + 2 * d * 4 * d
    attn_core = 2 * W * W * d                      # QK^T and AV, per layer, whole window
    return dict(stem=stem, layer_per_token=per_tok_layer, attn_scores_per_layer_window=attn_core,
                trunk_full=2 * (W * per_tok_layer + attn_core),
                trunk_last=(W * per_tok_layer + attn_core) + (W * 2 * d * d + d * d + d * d + 2 * d * 4 * d + 2 * W * d),
                trunk_kvcache_relpos=2 * (per_tok_layer + 2 * W * d))


def main():
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    res = dict(macs=macs())
    net = SWAv2(**ck["arch"]); net.load_state_dict(ck["model"]); net.eval()
    res["params"] = {k: sum(p.numel() for n, p in net.named_parameters() if n.startswith(k))
                     for k in ("conv", "fc", "pe_age", "blocks.0.attn", "blocks.0.mlp", "blocks.1.attn", "blocks.1.mlp",
                               "blocks.0.ln", "blocks.1.ln", "ln_f", "val", "adv")}
    res["params"]["total"] = sum(p.numel() for p in net.parameters())
    # exactness of the last-token trunk
    z = torch.randn(64, 8, 512)
    with torch.no_grad():
        res["trunk_last_maxdiff"] = float((trunk(net, z) - trunk_last(net, z)).abs().max())
    # ---------------- CPU, 1 thread (eval / video Student: B=1, one stem pass + trunk over the cached window)
    torch.set_num_threads(1)
    cpu = {}
    with torch.no_grad():
        x1 = torch.randint(0, 255, (1, 4, 84, 84), dtype=torch.uint8); z1 = torch.randn(1, 8, 512)
        cpu["stem_B1"] = tm(lambda: net.embed(x1), 200, 20, False)
        cpu["trunk_B1"] = tm(lambda: net.head(z1), 200, 20, False)
        cpu["trunk_last_B1"] = tm(lambda: trunk_last(net, z1), 200, 20, False)
        blk = net.blocks[0]
        cpu["block_attn_B1"] = tm(lambda: blk.attn(blk.ln1(z1)), 200, 20, False)
        cpu["block_mlp_B1"] = tm(lambda: blk.mlp(blk.ln2(z1)), 200, 20, False)
        from msclear.demos import Student
        stud = Student(net); stud.reset(); stk = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
        cpu["student_act_B1"] = tm(lambda: stud.act(stk), 200, 20, False)
    res["cpu_1thread_ms"] = cpu
    # ---------------- env stepping (1 core)
    from msclear.pixenv import PixelGame
    g = PixelGame(); g.reset(0); rng = np.random.default_rng(0)
    t = time.perf_counter(); n = 0
    while n < 3000:
        if g.done:
            g.reset(n)
        g.step(int(rng.integers(4))); n += 1
    res["env_step_ms_1core"] = (time.perf_counter() - t) / n * 1e3
    # ---------------- GPU
    torch.set_num_threads(4)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    net = net.to(dev)
    gpu = {}
    for B in (42, 512):
        x = torch.randint(0, 255, (B, 4, 84, 84), dtype=torch.uint8, device=dev); z = torch.randn(B, 8, 512, device=dev)
        with torch.no_grad():
            gpu[f"stem_B{B}"] = tm(lambda: net.embed(x))
            gpu[f"trunk_B{B}"] = tm(lambda: trunk(net, z))
            gpu[f"trunk_last_B{B}"] = tm(lambda: trunk_last(net, z))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gpu[f"stem_bf16_B{B}"] = tm(lambda: net.embed(x))
                gpu[f"trunk_bf16_B{B}"] = tm(lambda: trunk(net, z))
    # rollout step as in m7/m8_ppo (pi + ref: 2 stems, 2 trunks, value, sample, .cpu())
    from scripts.m7_ppo import ValueHead
    vh = ValueHead(512).to(dev); ref = SWAv2(**ck["arch"]).to(dev); ref.load_state_dict(ck["model"])
    N = 42
    stack = torch.randint(0, 255, (N, 4, 84, 84), dtype=torch.uint8, device=dev)
    tok = torch.randn(N, 8, 512, device=dev); rtok = tok.clone(); new = torch.zeros(N, dtype=torch.bool, device=dev)

    def roll_step():
        with torch.no_grad():
            zz = net.embed(stack); rz = ref.embed(stack)
            t1 = torch.where(new[:, None, None], zz[:, None].expand(-1, 8, -1), torch.cat([tok[:, 1:], zz[:, None]], 1))
            t2 = torch.where(new[:, None, None], rz[:, None].expand(-1, 8, -1), torch.cat([rtok[:, 1:], rz[:, None]], 1))
            h = trunk(net, t1); lg = net.adv(h) / 0.25; v = vh(h)
            rl = F.log_softmax(ref.adv(trunk(ref, t2)) / 0.25, -1)
            a = torch.distributions.Categorical(logits=lg).sample()
            return a.cpu().numpy()
    gpu["rollout_step_N42"] = tm(roll_step, 100, 20)
    # training minibatch: nb=14 envs, T=128, W=8 -> 14*135 stems + 1792 windows, fwd+bwd, 3 epochs x 3 mb per update
    nb, T = 14, 128
    H = torch.randint(0, 255, (nb, 7 + T, 4, 84, 84), dtype=torch.uint8, device=dev)
    pos = (7 + torch.arange(T, device=dev))[:, None] - 7 + torch.arange(8, device=dev)[None]
    IDX = pos[None].expand(nb, -1, -1)
    opt = torch.optim.Adam(list(net.parameters()) + list(vh.parameters()), lr=1e-9)
    net.train()

    def train_mb(fn):
        def f():
            Z = net.embed(H.reshape(-1, 4, 84, 84)).view(nb, 7 + T, -1)
            win = Z[torch.arange(nb, device=dev)[:, None, None], IDX]
            h = fn(net, win.reshape(nb * T, 8, -1))
            loss = net.adv(h).logsumexp(-1).mean() + vh(h).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return f
    gpu["train_mb_fwdbwd"] = tm(train_mb(trunk), 20, 5)
    gpu["train_mb_fwdbwd_lasttok"] = tm(train_mb(trunk_last), 20, 5)

    def stem_only():
        Z = net.embed(H.reshape(-1, 4, 84, 84)); loss = Z.mean(); opt.zero_grad(); loss.backward()
    gpu["train_mb_stem_only_fwdbwd"] = tm(stem_only, 20, 5)
    net.eval()
    # torch.compile of the rollout trunk (reduce-overhead = CUDA graphs)
    try:
        ct = torch.compile(lambda zz: trunk(net, zz), mode="reduce-overhead")
        with torch.no_grad():
            gpu["trunk_B42_compiled"] = tm(lambda: ct(tok), 100, 30)
    except Exception as e:  # noqa: BLE001
        gpu["trunk_B42_compiled"] = f"failed: {type(e).__name__}: {str(e)[:120]}"
    res["gpu_ms"] = gpu
    (OUT / "profile.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
