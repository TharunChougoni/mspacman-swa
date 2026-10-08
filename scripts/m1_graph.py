"""M1: maze graph + GVF tables, with coverage, eat-zone, convergence and BFS cross-checks.

Env: WORKERS (default 12), N_DEC decisions per worker (default 300000).
Writes runs/m1/model.pkl (model + dot GVFs + cell GVFs) and runs/m1/m1.json.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msclear.maze import Model, collect, teleported  # noqa: E402

OUT = Path(os.environ.get("OUT", "runs/m1")); OUT.mkdir(parents=True, exist_ok=True)
W = int(os.environ.get("WORKERS", "12")); N = int(os.environ.get("N_DEC", "300000"))
log = lambda *a: print(*a, flush=True)  # noqa: E731

t0 = time.time()
log(f"M1 start: {W} workers x {N:,} decisions (+ validation {W} x {N // 4:,})")
trans, flips = collect(workers=W, n_dec=N, seed0=0, progress=log)
log(f"collected {W}x{N} decisions in {time.time() - t0:.0f}s: {len(trans)} distinct transitions, {sum(flips.values())} dot flips, {collect.teleports} teleport steps dropped")
t1 = time.time()
vtrans, vflips = collect(workers=W, n_dec=max(N // 4, 1000), seed0=1000, progress=log)   # independent validation run
log(f"validation run {W}x{N // 4} in {time.time() - t1:.0f}s")
m = Model(trans, flips)
log(f"model: {m.nS} states (x,y,heading), {len(m.positions)} positions, {len(m.cells)} cells, {len(m.o_sa)} outcomes; eat offsets {dict(m.offsets)}")

# coverage: decision positions seen in the validation run that the model has with all 4 actions sampled
full = {(s[0], s[1]) for s, c in zip(m.states, m.sa_count) if (c > 0).all()}
seen_any = {(s[0], s[1]) for s in m.states}
vpos = {(k[0][0], k[0][1]) for k in vtrans}
cov_full = len(vpos & full) / len(vpos); cov_any = len(vpos & seen_any) / len(vpos)
sa_frac = float((m.sa_count > 0).mean())
log(f"coverage: {cov_any:.4f} of validation positions known, {cov_full:.4f} with all 4 actions sampled; (s,a) pairs sampled {sa_frac:.4f}")

# eat-zone check on validation flips
dot_t = {b: t for t, b in enumerate(m.dot_bits)}
ok = tot = 0
for (i, p), c in vflips.items():
    if i in dot_t:
        tot += c; ok += c * (p in m.zones[dot_t[i]])
log(f"eat zones: {ok}/{tot} validation dot flips inside the predicted zone ({ok / max(tot, 1):.4f}); zone size {min(map(len, m.zones))}-{max(map(len, m.zones))}")

# GVFs
nd = len(m.zones)      # 150 dots + 4 power pellets
log(f"power pellet zones: sizes {[len(z) for z in m.zones[m.n_dots:]]}")
t2 = time.time(); Gd, hd = m.solve(targets=range(nd), log=log); log(f"dot+power GVFs: {len(hd)} sweeps, final max change {hd[-1]:.2e}, {time.time() - t2:.0f}s")
t3 = time.time(); Gc, hc = m.solve(targets=range(nd, m.nT), log=log); log(f"cell GVFs: {len(hc)} sweeps, final max change {hc[-1]:.2e}, {time.time() - t3:.0f}s")

# BFS cross-check on the frame-level position graph
adj = collections.defaultdict(set)
for (s, a, s2, vis), c in trans.items():
    seq = [(s[0], s[1])] + list(vis)
    for p, q in zip(seq[:-1], seq[1:]):
        if p != q and not teleported([p, q]):
            adj[p].add(q); adj[q].add(p)


def bfs(srcs):
    d = {p: 0 for p in srcs}; dq = collections.deque(srcs)
    while dq:
        p = dq.popleft()
        for q in adj[p]:
            if q not in d:
                d[q] = d[p] + 1; dq.append(q)
    return d


agree = n = 0; xs, ys = [], []
rng = np.random.default_rng(0)
idx_sa = {}
for k in range(len(m.starts)):
    st = m.starts[k]; en = m.starts[k + 1] if k + 1 < len(m.starts) else len(m.o_sa)
    idx_sa[int(m.o_sa[st])] = (st, en)
full_states = [i for i in range(m.nS) if (m.sa_count[i] > 0).all()]
log("BFS cross-check ...")
for t in range(m.n_dots):
    d = bfs(list(m.zones[t]))
    for si in rng.choice(full_states, size=min(300, len(full_states)), replace=False):
        exp = []
        for a in range(4):
            st, en = idx_sa[si * 4 + a]; e = 0.0
            for o in range(st, en):
                hit = t in m.o_rew[o]
                p2 = m.states[m.o_next[o]]
                e += m.o_p[o] * (0 if hit else d.get((p2[0], p2[1]), 10**6))
            exp.append(e)
        exp = np.array(exp); g = Gd[si, :, t]
        if g.max() > 0:
            xs.append(np.log(g.max()) / np.log(m.gamma)); s = m.states[si]; ys.append(d.get((s[0], s[1]), np.nan))
        o = np.sort(exp)
        if o[1] - o[0] >= 2 and np.sort(g)[-1] - np.sort(g)[-2] > 1e-6:
            n += 1; agree += int(np.argmax(g) == np.argmin(exp))
xs, ys = np.array(xs), np.array(ys); ok_ = np.isfinite(ys) & (ys < 10**5)
r = float(np.corrcoef(xs[ok_], ys[ok_])[0, 1]); slope = float(np.polyfit(ys[ok_], xs[ok_], 1)[0])
log(f"BFS cross-check: greedy action agreement {agree}/{n} = {agree / max(n, 1):.4f} (cases with a clear BFS winner); "
    f"corr(GVF steps, BFS frames) = {r:.4f}, slope {slope:.3f} steps/frame")

m.save(OUT / "model.pkl", Gd=Gd.astype(np.float32), Gc=Gc.astype(np.float16))
json.dump(dict(workers=W, n_dec=N, nS=m.nS, positions=len(m.positions), cells=len(m.cells), cov_any=cov_any, cov_full=cov_full,
               sa_frac=sa_frac, zone_recall=ok / max(tot, 1), dot_sweeps=len(hd), dot_final=hd[-1], cell_sweeps=len(hc), cell_final=hc[-1],
               bfs_agree=agree / max(n, 1), bfs_n=n, bfs_corr=r, seconds=time.time() - t0), open(OUT / "m1.json", "w"), indent=1)
log(f"saved {OUT}/model.pkl; total {time.time() - t0:.0f}s")
