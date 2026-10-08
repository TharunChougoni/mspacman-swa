"""M6 training data: pack demo shards into one uint8 memmap per source dir, sample windows.

pack(src) -> src/pack/{frames.u8, meta.npz, packed.json}. Idempotent: skipped when packed.json
lists exactly the shards present; rebuilt otherwise (a cut mid-pack leaves only *.tmp files).

A training row t = one decision. Its input is the W+3 consecutive frames ending at t
(clamped at the episode start, which reproduces the repeat-padding of the live student),
its label the HRA action. Rows where Ms. Pac-Man did not move during the step (death and
respawn freezes, ghost-eaten pauses, the end-of-board flash) are kept as context but not trained on.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from msclear.demos import read_shard

META = ("label", "act", "items", "lives", "lost", "moved", "done", "q")


def shard_files(src: Path) -> list[Path]:
    return sorted(p for p in Path(src).glob("shard_*.npz") if not p.name.endswith(".tmp.npz"))


def pack(src: Path, val_every: int = 20, log=print) -> Path:
    src = Path(src); out = src / "pack"; out.mkdir(exist_ok=True)
    files = shard_files(src)
    names = [p.name for p in files]
    pj = out / "packed.json"
    if pj.exists() and json.loads(pj.read_text()).get("shards") == names and (out / "frames.u8").exists():
        log(f"pack {src}: up to date ({len(names)} shards)")
        return out
    for p in (pj, out / "frames.u8", out / "meta.npz"):
        p.unlink(missing_ok=True)
    sizes = []
    for p in files:
        with np.load(p) as z:
            sizes.append(int(z["label"].shape[0]))
    n = int(sum(sizes))
    tmp = out / "frames.u8.tmp"
    mm = \
        np.memmap(tmp, dtype=np.uint8, mode="w+", shape=(n, 84, 84))
    cols = {k: [] for k in META}
    ep_start, cleared, shard_id, ep_gid, val, sticky, eps_noise = [], [], [], [], [], [], []
    episodes = []
    o = 0
    for si, p in enumerate(files):
        eps, a = read_shard(p)
        m = len(a["label"])
        mm[o:o + m] = a["frames"]
        for k in META:
            cols[k].append(a[k])
        sid = int(p.stem.split("_")[1])
        is_val = (sid % val_every) == val_every - 1
        for e in eps:
            s0 = o + e["start"]
            ep_start.append(np.full(e["len"], s0, np.int64))
            cleared.append(np.full(e["len"], e["cleared"], bool))
            ep_gid.append(np.full(e["len"], len(episodes), np.int32))
            sticky.append(np.full(e["len"], e.get("sticky", 0.25), np.float16))
            eps_noise.append(np.full(e["len"], e.get("eps", 0.0), np.float16))
            episodes.append(dict(e, shard=p.name, val=is_val, gstart=s0))
        shard_id.append(np.full(m, sid, np.int32)); val.append(np.full(m, is_val, bool))
        o += m
        if si % 20 == 0:
            log(f"  pack {src.name}: {si + 1}/{len(files)} shards, {o:,}/{n:,} rows")
    mm.flush(); del mm
    os.replace(tmp, out / "frames.u8")
    meta = {k: np.concatenate(v) for k, v in cols.items()}
    meta.update(ep_start=np.concatenate(ep_start), cleared=np.concatenate(cleared), shard=np.concatenate(shard_id),
                ep=np.concatenate(ep_gid), val=np.concatenate(val), sticky=np.concatenate(sticky),
                eps=np.concatenate(eps_noise))
    np.savez(out / "meta.tmp.npz", **meta)
    os.replace(out / "meta.tmp.npz", out / "meta.npz")
    (out / "episodes.json").write_text(json.dumps(episodes))
    pj.write_text(json.dumps(dict(shards=names, rows=n)))
    log(f"pack {src}: {len(names)} shards, {n:,} rows -> {out}")
    return out


class Pack:
    def __init__(self, d: Path):
        d = Path(d)
        self.meta = dict(np.load(d / "meta.npz"))
        self.n = len(self.meta["label"])
        self.frames = np.memmap(d / "frames.u8", dtype=np.uint8, mode="r", shape=(self.n, 84, 84))
        m = self.meta
        self.trainable = (m["moved"] == 1) & (m["label"] >= 0)


class Windows:
    """Batches of (frames (B, W+3, 84, 84) uint8, label (B,), q (B, 4)) from one or more packs.

    Batch k is a pure function of (seed, k), so training resumes onto the same sample stream.
    """

    def __init__(self, packs: list[Path], window: int, batch: int, split: str = "train", seed: int = 0,
                 weights: list[float] | None = None):
        self.packs = [Pack(p) for p in packs]
        self.W, self.B, self.seed = window, batch, seed
        self.rows = []
        for pk in self.packs:
            sel = pk.trainable & (pk.meta["val"] if split == "val" else ~pk.meta["val"])
            self.rows.append(np.flatnonzero(sel))
        w = np.array(weights if weights else [len(r) for r in self.rows], float)
        self.p = w / w.sum()

    def __len__(self):
        return 10**12

    def gather(self, pi: int, t: np.ndarray):
        pk = self.packs[pi]
        idx = t[:, None] + np.arange(-(self.W + 2), 1)[None]
        idx = np.maximum(idx, pk.meta["ep_start"][t][:, None])
        return pk.frames[idx], pk.meta["label"][t].astype(np.int64), pk.meta["q"][t].astype(np.float32)

    def sample(self, k: int):
        rng = np.random.default_rng([self.seed, k])
        which = rng.choice(len(self.packs), size=self.B, p=self.p)
        fr, lab, q = [], [], []
        for pi in np.unique(which):
            m = int((which == pi).sum())
            t = np.sort(self.rows[pi][rng.integers(0, len(self.rows[pi]), m)])
            f, l, qq = self.gather(pi, t)
            fr.append(f); lab.append(l); q.append(qq)
        return np.concatenate(fr), np.concatenate(lab), np.concatenate(q)

    def __getitem__(self, k: int):
        import torch
        f, l, q = self.sample(int(k))
        return torch.from_numpy(f), torch.from_numpy(l), torch.from_numpy(q)

    def fixed(self, n: int, seed: int = 123):
        """A fixed evaluation set: list of (pack, rows) covering n rows."""
        rng = np.random.default_rng(seed)
        tot = sum(len(r) for r in self.rows)
        out = []
        for pi, r in enumerate(self.rows):
            m = int(round(n * len(r) / tot))
            if m:
                out.append((pi, np.sort(rng.choice(r, size=min(m, len(r)), replace=False))))
        return out
