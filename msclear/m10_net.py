"""M10 net: SWAv2 with relative (ALiBi) positions instead of the absolute age embedding, plus an exact
incremental KV-cache inference path.

Semantics (streaming sliding-window attention): every layer, at every time step, attends to the last W
positions (itself + W-1 previous) of the layer below, with an ALiBi bias -slope_h * age (per-head learnable
slope, init = SWAv2's recency lam so the M9 weights transfer). No absolute positions anywhere.
Receptive field of the top layer = L*(W-1)+1 tokens (15 for 2 layers, W=8).

Episode start (normal reset or pool/snapshot start) = the first token repeated into the past, exactly as SWAv2's
window of W copies. Life loss does NOT reset the window (same as SWAv2 in fasteval / PPO).

Full path   trunk_seq(z (B, S, d)) -> h (B, S, d), banded causal mask; for the newest output, S >= 15 tokens of
            history clamped at the episode start reproduces the streaming value exactly.
Incremental KVCache: per env, per layer, K/V of the last W-1 positions (rolling, index = age). step(z) costs one
            CNN pass + one query per layer; reset(mask, z0) primes the caches with the repeated first token.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlibiAttention(nn.Module):
    def __init__(self, d, n_heads, window, lam=0.05):
        super().__init__()
        self.h, self.dh, self.W = n_heads, d // n_heads, window
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.slope = nn.Parameter(torch.full((n_heads,), float(lam)))

    def forward(self, x):
        """(B, S, d) -> (B, S, d) banded causal (age 0..W-1)."""
        b, s, d = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, self.h, self.dh).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        idx = torch.arange(s, device=x.device)
        age = (idx[:, None] - idx[None, :]).to(x.dtype)
        bias = -self.slope.view(1, -1, 1, 1) * age.clamp(min=0)
        bias = bias.masked_fill(((age < 0) | (age >= self.W)), float("-inf"))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return self.out(y.transpose(1, 2).reshape(b, s, d))

    def forward_q(self, x, nq):
        b, s, d = x.shape
        w, bb = self.qkv.weight, self.qkv.bias
        q = F.linear(x[:, -nq:], w[:d], bb[:d]).view(b, nq, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(x)
        qi = torch.arange(s - nq, s, device=x.device); ki = torch.arange(s, device=x.device)
        age = (qi[:, None] - ki[None, :]).to(x.dtype)
        bias = -self.slope.view(1, -1, 1, 1) * age.clamp(min=0)
        bias = bias.masked_fill(((age < 0) | (age >= self.W)), float("-inf"))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return self.out(y.transpose(1, 2).reshape(b, nq, d))

    def kv(self, x):
        """(B, S, d) normed -> k, v (B, H, S, dh)."""
        b, s, d = x.shape
        w, bb = self.qkv.weight, self.qkv.bias
        k, v = F.linear(x, w[d:], bb[d:]).view(b, s, 2, self.h, self.dh).unbind(2)
        return k.transpose(1, 2), v.transpose(1, 2)

    def step(self, x, ck, cv):
        """x (B, d) normed newest; ck, cv (B, H, W-1, dh) oldest first -> y (B, d), new k, v (B, H, 1, dh)."""
        b, d = x.shape
        q, k, v = self.qkv(x).view(b, 3, self.h, self.dh).unbind(1)
        k, v = k[:, :, None], v[:, :, None]
        K, V = torch.cat([ck, k], 2), torch.cat([cv, v], 2)                     # (B, H, W, dh)
        age = torch.arange(self.W - 1, -1, -1, device=x.device, dtype=x.dtype)
        sc = (q[:, :, None] @ K.transpose(-2, -1))[:, :, 0] / math.sqrt(self.dh) - self.slope[:, None] * age
        y = (torch.softmax(sc, -1)[:, :, None] @ V)[:, :, 0]                    # (B, H, dh)
        return self.out(y.reshape(b, d)), k, v


class Block(nn.Module):
    def __init__(self, d, n_heads, window, lam):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = AlibiAttention(d, n_heads, window, lam)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

    def forward_q(self, x, nq):
        """(B, S, d) -> (B, nq, d): outputs for the nq newest positions only (keys/values over all S)."""
        h = self.ln1(x)
        y = x[:, -nq:] + self.attn.forward_q(h, nq)
        return y + self.mlp(self.ln2(y))


class M10Net(nn.Module):
    def __init__(self, n_actions=4, d=512, n_heads=8, n_layers=2, window=8, lam=0.05):
        super().__init__()
        self.d, self.window, self.n_actions, self.n_layers = d, window, n_actions, n_layers
        self.ctx = n_layers * (window - 1) + 1
        self.conv = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=1), nn.ReLU())
        self.fc = nn.Linear(3136, d)
        self.blocks = nn.ModuleList(Block(d, n_heads, window, lam) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.val = nn.Linear(d, 1)
        self.adv = nn.Linear(d, n_actions)

    def embed(self, stacks):
        return torch.relu(self.fc(self.conv(stacks.float() / 255.0).flatten(1)))

    def trunk_seq(self, z):
        for blk in self.blocks:
            z = blk(z)
        return self.ln_f(z)

    def trunk_last(self, z):
        """(B, S, d) history (S >= ctx for exactness, clamped at episode start) -> h (B, d) newest.
        Pruned: layer l only computes the (L-1-l)*(W-1)+1 newest positions the layers above need (exact)."""
        x = z[:, -self.ctx:]
        L = self.n_layers
        for l, blk in enumerate(self.blocks):
            nq = (L - 1 - l) * (self.window - 1) + 1
            x = blk.forward_q(x, nq)
        return self.ln_f(x[:, -1])

    def q_of(self, h):
        a = self.adv(h)
        return self.val(h) + a - a.mean(1, keepdim=True)

    # ---- incremental
    def new_cache(self, B, device, dtype=torch.float32):
        H, dh = self.blocks[0].attn.h, self.blocks[0].attn.dh
        return [torch.zeros(2, B, H, self.window - 1, dh, device=device, dtype=dtype) for _ in self.blocks]

    def prime(self, cache, mask, z0):
        """Episode start for envs `mask` (bool (B,)): history = z0 repeated. Run the full path on ctx-1 copies,
        store each layer's K/V of the last W-1 positions."""
        idx = mask.nonzero()[:, 0]
        if len(idx) == 0:
            return
        x = z0[idx][:, None].expand(-1, self.ctx - 1, -1)
        for li, blk in enumerate(self.blocks):
            k, v = blk.attn.kv(blk.ln1(x))
            cache[li][0, idx] = k[:, :, -(self.window - 1):]
            cache[li][1, idx] = v[:, :, -(self.window - 1):]
            x = blk(x)

    def step(self, cache, z):
        """z (B, d) newest tokens -> h (B, d); cache updated in place (rolled by one)."""
        x = z
        for li, blk in enumerate(self.blocks):
            y, k, v = blk.attn.step(blk.ln1(x), cache[li][0], cache[li][1])
            cache[li][0] = torch.cat([cache[li][0][:, :, 1:], k], 2)
            cache[li][1] = torch.cat([cache[li][1][:, :, 1:], v], 2)
            x = x + y
            x = x + blk.mlp(blk.ln2(x))
        return self.ln_f(x)

    def select(self, cache, keep):
        """Drop finished envs: keep = index tensor."""
        return [c[:, keep] for c in cache]


def from_swav2(sd: dict, arch: dict) -> tuple[M10Net, list[str]]:
    """M10Net initialised from a SWAv2 state dict (everything except pe_age; slopes = lam)."""
    net = M10Net(**arch)
    own = net.state_dict()
    ok = [k for k, v in sd.items() if k in own and own[k].shape == v.shape]
    own.update({k: sd[k] for k in ok}); net.load_state_dict(own)
    return net, ok
