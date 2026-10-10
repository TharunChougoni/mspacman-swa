"""SWA-v2 student net, ported from mspacman-rainbow (cleanrl_utils/swa_net.py, archived 2026-10-08).

SWA = sliding-window attention: Nature-CNN stem turns each 4x84x84 stack into ONE token,
a causal window of W tokens (newest last) goes through recency-biased causal attention
blocks, and a dueling head reads Q from the newest token.

Changes vs the original: n_actions defaults to 4 (UP/RIGHT/LEFT/DOWN, HRA's action set);
the tokens of past steps can be cached (`embed` + `head`) so acting costs one CNN pass per
step; `load_stem` initialises from the archived RTCNet checkpoint (all shape-matching
tensors; goal and 9-way advantage weights are skipped). The ViT/RTC variants are not ported.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Last block emits only the newest token (causal; the head reads that one). MSCLEAR_FAST_LAST=0 disables it.
FAST_LAST = os.environ.get("MSCLEAR_FAST_LAST", "1") != "0"


class RecencyCausalAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, lam: float = 0.05):
        super().__init__()
        if d % n_heads:
            raise ValueError(f"d={d} not divisible by n_heads={n_heads}")
        self.n_heads, self.d_head, self.lam = n_heads, d // n_heads, lam
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, self.d_head).unbind(dim=2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        idx = torch.arange(t, device=x.device)
        age = idx[:, None] - idx[None, :]
        scores = scores - self.lam * age.clamp(min=0).to(scores.dtype)
        scores = scores.masked_fill(age < 0, torch.finfo(scores.dtype).min)
        y = torch.matmul(torch.softmax(scores, dim=-1), v)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, d))

    def forward_last(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, T, d), already layer-normed -> (B, 1, d) for the newest query. K and V still cover all T."""
        b, t, d = x.shape
        w, bias = self.qkv.weight, self.qkv.bias
        # qkv rows are [0:d]=q, [d:2d]=k, [2d:3d]=v, matching .view(..., 3, n_heads, d_head).
        q = F.linear(x[:, -1:], w[:d], bias[:d]).view(b, 1, self.n_heads, self.d_head)
        kv = F.linear(x, w[d:], bias[d:]).view(b, t, 2, self.n_heads, self.d_head)
        k, v = kv.unbind(dim=2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        age = (t - 1) - torch.arange(t, device=x.device)          # newest query: -lam * (T-1-j), no future keys
        scores = scores - self.lam * age.to(scores.dtype)
        y = torch.matmul(torch.softmax(scores, dim=-1), v)
        return self.out(y.transpose(1, 2).contiguous().view(b, 1, d))


class SWABlock(nn.Module):
    def __init__(self, d: int, n_heads: int, lam: float):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = RecencyCausalAttention(d, n_heads, lam)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

    def forward_last(self, x):
        """(B, T, d) -> (B, 1, d): newest token only. This block's K,V still see all T."""
        h = x[:, -1:] + self.attn.forward_last(self.ln1(x))
        return h + self.mlp(self.ln2(h))


class SWAv2(nn.Module):
    """Conv stem -> 1 token per 4-stack -> causal window of W tokens -> dueling Q (newest token)."""

    def __init__(self, n_actions: int = 4, d: int = 512, n_heads: int = 8, n_layers: int = 2,
                 window: int = 16, lam: float = 0.05):
        super().__init__()
        self.d, self.window, self.n_actions = d, window, n_actions
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Linear(3136, d)
        self.pe_age = nn.Embedding(window, d)
        self.blocks = nn.ModuleList(SWABlock(d, n_heads, lam) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.val = nn.Linear(d, 1)
        self.adv = nn.Linear(d, n_actions)

    def embed(self, stacks: torch.Tensor) -> torch.Tensor:
        """(N, 4, 84, 84) uint8 -> (N, d) tokens."""
        x = stacks.float() / 255.0
        return torch.relu(self.fc(self.conv(x).flatten(1)))

    def trunk(self, z: torch.Tensor, last_only: bool | None = None) -> torch.Tensor:
        """(B, T, d) tokens, oldest first -> (B, d) ln_f at the newest token (what head feeds to val/adv)."""
        if last_only is None:
            last_only = FAST_LAST
        t = z.shape[1]
        if t > self.window:
            raise ValueError(f"T={t} > window={self.window}")
        z = z + self.pe_age(torch.arange(t, device=z.device))[None]
        if last_only and len(self.blocks):
            for blk in self.blocks[:-1]:
                z = blk(z)
            h = self.blocks[-1].forward_last(z)[:, 0]
        else:
            for blk in self.blocks:
                z = blk(z)
            h = z[:, -1]
        return self.ln_f(h)

    def head(self, z: torch.Tensor) -> torch.Tensor:
        """(B, T, d) tokens, oldest first, T <= window -> Q (B, n_actions) at the newest token."""
        h = self.trunk(z)
        a = self.adv(h)
        return self.val(h) + a - a.mean(dim=1, keepdim=True)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, 4, 84, 84) or (B, 4, 84, 84) uint8 -> Q (B, n_actions)."""
        if frames.ndim == 4:
            frames = frames.unsqueeze(1)
        b, t = frames.shape[:2]
        z = self.embed(frames.reshape(b * t, *frames.shape[2:])).view(b, t, self.d)
        return self.head(z)

    def forward_seq(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, T+3, 84, 84) consecutive single frames -> Q, using the T overlapping 4-stacks."""
        st = seq.unfold(1, 4, 1).permute(0, 1, 4, 2, 3)        # B, T, 4, 84, 84
        return self.forward(st)


def load_stem(net: nn.Module, path: str) -> list[str]:
    """Copy every shape-matching tensor from an archived SWAv2/RTCNet checkpoint. Returns loaded keys."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        for key in ("model", "q_network", "state_dict", "net"):
            if key in ck and isinstance(ck[key], dict):
                ck = ck[key]
                break
    own = net.state_dict()
    ok = {k: v for k, v in ck.items() if k in own and hasattr(v, "shape") and v.shape == own[k].shape}
    own.update(ok)
    net.load_state_dict(own)
    return sorted(ok)
