"""M10Net: incremental KV-cache path == full-window recompute (fp32, long rollouts with episode resets)."""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msclear.m10_net import M10Net, from_swav2  # noqa: E402


def _strict():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run_parity(net, B=8, steps=3000, p_reset=0.004, dev="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    net = net.to(dev).eval()
    hist = torch.relu(torch.randn(B, steps, net.d, generator=g)).to(dev) * 2
    resets = torch.rand(B, steps, generator=g) < p_reset
    resets[:, 0] = True
    cache = net.new_cache(B, dev)
    es = torch.zeros(B, dtype=torch.long)
    worst = 0.0
    global stats
    stats = dict(n=0, agree=0, qabs=0.0, abs=0.0)
    with torch.no_grad():
        for t in range(steps):
            r = resets[:, t]
            es = torch.where(r, torch.full_like(es, t), es)
            z = hist[:, t]
            net.prime(cache, r.to(dev), z)
            h_inc = net.step(cache, z)
            if t % 7 == 0 or r.any():
                idx = torch.clamp(torch.arange(t - net.ctx + 1, t + 1)[None] , min=0).expand(B, -1)
                idx = torch.maximum(idx, es[:, None])
                ctx = hist[torch.arange(B)[:, None], idx.to(dev)]
                h_full = net.trunk_last(ctx)
                qi, qf = net.q_of(h_inc), net.q_of(h_full)
                # relative to scale (Q of M9 is O(10)); LayerNormed h is O(1)
                worst = max(worst, ((qi - qf).abs() / qf.abs().clamp(min=1)).max().item(), (h_inc - h_full).abs().max().item())
                stats["n"] += qi.shape[0]; stats["agree"] += (qi.argmax(1) == qf.argmax(1)).sum().item()
                stats["qabs"] = max(stats["qabs"], qf.abs().max().item()); stats["abs"] = max(stats["abs"], (qi - qf).abs().max().item())
    return worst


def test_parity_random_cpu():
    torch.manual_seed(0)
    w = run_parity(M10Net(), steps=1500)
    assert w < 1e-4, w


def test_parity_m9_weights_gpu():
    if not torch.cuda.is_available() or not Path("runs/m9run/best.pt").exists():
        return
    _strict()
    ck = torch.load("runs/m9run/best.pt", map_location="cpu", weights_only=False)
    net, _ = from_swav2(ck["model"], ck["arch"])
    w = run_parity(net, B=32, steps=int(os.environ.get("PARITY_STEPS", "5000")), dev="cuda", seed=1)
    print("max rel/h diff", w, stats)
    assert w < 1e-4, w


if __name__ == "__main__":
    print("cpu random", run_parity(M10Net(), steps=1500))
    test_parity_m9_weights_gpu()
