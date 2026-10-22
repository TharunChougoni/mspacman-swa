"""M23 students: per-frame spatial tokens + factorized attention (spatial within a frame, causal temporal per grid
cell over W frames, RoPE on time), fused multi-head SDPA (one call for all heads; Flash in bf16 on Ampere,
memory-efficient kernel in fp32). Input: (.., 128, 128, 3) uint8 HWC colour playfield frames (msclear.m23_env).

Stems: 'cnn' = 4 convs (128 -> 16x16) + adaptive pool to grid x grid, 1x1 to d; 'vit' = linear 16x16 patches (8x8).
Head: learned-query attention pool over the newest frame's tokens -> n_out (4 policy logits, or 1 value).
Acting: act(frame, cache, t) runs one frame; each layer caches the temporal K/V (RoPE applied) of the last W-1
frames per grid cell. (Cached layer-2 keys carry their own sliding context, so acting can see slightly further back
than a W-frame training chunk; training never sees more than W frames.)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def rope(x, pos):
    dh = x.shape[-1]
    f = 10000 ** (-torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh)
    a = pos.float()[:, None] * f[None]; c, s = a.cos().to(x.dtype), a.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).flatten(-2)


class MHA(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.h, self.dh = h, d // h; self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)

    def forward(self, x, causal=False, pos=None, kv=None):
        N, L, d = x.shape
        q, k, v = self.qkv(x).view(N, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if pos is not None:
            q, k = rope(q, pos[-L:]), rope(k, pos[-L:])
        if kv is not None:
            k, v = torch.cat([kv[0].to(k.dtype), k], 2), torch.cat([kv[1].to(v.dtype), v], 2); causal = False
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.o(y.transpose(1, 2).reshape(N, L, d)), (k, v)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sp, self.tm = MHA(d, h), MHA(d, h)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, pos, kv=None):            # x (B, T, P, d)
        B, T, P, d = x.shape
        x = x + self.sp(self.l1(x).reshape(B * T, P, d))[0].view(B, T, P, d)
        xt = self.l2(x).permute(0, 2, 1, 3).reshape(B * P, T, d)
        y, nkv = self.tm(xt, causal=True, pos=pos, kv=kv)
        x = x + y.view(B, P, T, d).permute(0, 2, 1, 3)
        return x + self.mlp(self.l3(x)), nkv


class TokNet(nn.Module):
    def __init__(self, stem="cnn", d=512, h=8, L=2, grid=4, patch=16, n_out=4, window=32, ckpt=False):
        super().__init__()
        self.arch = dict(stem=stem, d=d, h=h, L=L, grid=grid, patch=patch, n_out=n_out, window=window)
        self.d, self.W, self.ckpt = d, window, ckpt
        if stem == "cnn":
            self.stem = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
                                      nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(), nn.Conv2d(128, 128, 3, 1, 1), nn.GELU(),
                                      nn.AdaptiveAvgPool2d(grid), nn.Conv2d(128, d, 1))
            P = grid * grid
        else:
            self.stem = nn.Sequential(nn.Conv2d(3, d, patch, patch)); P = (128 // patch) ** 2
        self.ln_in = nn.LayerNorm(d)
        self.pos2d = nn.Parameter(torch.randn(1, 1, P, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(L))
        self.lnf = nn.LayerNorm(d); self.pool_q = nn.Parameter(torch.randn(d) * 0.02); self.out = nn.Linear(d, n_out)
        self.P = P
        # fixed input normalisation (dataset per-pixel mean/std, set by scripts/m23_train.py): the static maze dominates
        # raw pixels and an un-normalised student sits on the action prior for thousands of steps
        self.register_buffer("mu", torch.zeros(128, 128, 3)); self.register_buffer("sd", torch.ones(128, 128, 3))

    def embed(self, fr):                           # (N, 128, 128, 3) uint8 -> (N, P, d)
        x = ((fr.float() / 255.0 - self.mu) / self.sd).permute(0, 3, 1, 2)
        return self.ln_in(self.stem(x).flatten(2).transpose(1, 2))

    def head(self, x):                             # (..., P, d) -> (..., n_out)
        x = self.lnf(x); w = torch.softmax(x @ self.pool_q / math.sqrt(self.d), -1)
        return self.out((w.unsqueeze(-1) * x).sum(-2))

    def forward(self, frames):                     # (B, T, 128, 128, 3) -> (B, T, n_out), causal in T
        B, T = frames.shape[:2]
        flat = frames.reshape(B * T, *frames.shape[2:])
        z = checkpoint(self.embed, flat, use_reentrant=False) if self.ckpt and self.training else self.embed(flat)
        x = z.view(B, T, self.P, self.d) + self.pos2d; pos = torch.arange(T, device=frames.device)
        for b in self.blocks:
            x = checkpoint(lambda x_, b=b: b(x_, pos)[0], x, use_reentrant=False) if self.ckpt and self.training else b(x, pos)[0]
        return self.head(x)

    @torch.no_grad()
    def act(self, frame, cache, t):                # frame (B,128,128,3); cache None at episode start (t = 0)
        B = frame.shape[0]; x = self.embed(frame).view(B, 1, self.P, self.d) + self.pos2d
        pos = torch.arange(t + 1, device=frame.device); new = []
        for i, b in enumerate(self.blocks):
            x, kv = b(x, pos, kv=None if cache is None else cache[i])
            new.append((kv[0][:, :, -(self.W - 1):], kv[1][:, :, -(self.W - 1):]))
        return self.head(x[:, -1]), new

    def select(self, cache, keep):                 # keep: index tensor over the batch
        out = []
        for k, v in cache:
            BP = k.shape[0]; B = BP // self.P
            out.append(tuple(z.view(B, self.P, *z.shape[1:])[keep].reshape(-1, *z.shape[1:]) for z in (k, v)))
        return out


ARCHS = {
    "cnn": dict(stem="cnn", d=512, h=8, L=2, grid=4),
    "vit": dict(stem="vit", d=384, h=6, L=2, patch=16),
    "value": dict(stem="cnn", d=256, h=4, L=2, grid=4, n_out=1),
}


def build(name_or_arch, **kw):
    a = dict(ARCHS[name_or_arch]) if isinstance(name_or_arch, str) else dict(name_or_arch)
    a.update(kw); ck = a.pop("ckpt", False)
    return TokNet(**a, ckpt=ck)


def load_student(path, device, which="pi"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build(ck["arch"][which]); net.load_state_dict(ck[which]); return net.eval().to(device)
