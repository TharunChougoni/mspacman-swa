"""M9 option-A verification on REAL M8 states (written by the orchestrating agent, not Grok Build).

Plays G games of runs/m8/best.pt in lockstep in this process (fp32 GPU, full trunk, greedy = the reference
policy that drives the games) and at every step compares on the same stacks/token history:
  * fast-last vs full trunk, fp32 GPU          (exactness of item 1)
  * fast-last vs full trunk, fp32 CPU (every 4th step; the eval/video path)
  * bf16 autocast (embed + head, own bf16 token history) vs fp32  (item 2)
Reports max |logit diff| and greedy-action agreement. Env: G (8), SEED0 (3000), OUT (runs/m9/verify.json).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msclear.pixenv import PixelGame  # noqa: E402
from msclear.swa_net import SWAv2  # noqa: E402

E = os.environ.get


def main():
    G, S0 = int(E("G", "8")), int(E("SEED0", "3000"))
    ck = torch.load(E("CKPT", "runs/m8/best.pt"), map_location="cpu", weights_only=False)
    net = SWAv2(**ck["arch"]); net.load_state_dict(ck["model"]); net.eval()
    cpu = SWAv2(**ck["arch"]); cpu.load_state_dict(ck["model"]); cpu.eval()
    dev = torch.device("cuda"); net = net.to(dev); W = net.window
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # strict fp32 reference
    torch.set_num_threads(4)
    games = [PixelGame(sticky=0.25) for _ in range(G)]
    stacks = np.stack([g.reset(S0 + i) for i, g in enumerate(games)])
    tok = tok16 = None
    acc = dict(n=0, fast_agree=0, fast_maxdiff=0.0, bf16_agree=0, bf16_maxdiff=0.0, n_cpu=0, cpu_agree=0, cpu_maxdiff=0.0)
    t0 = time.time(); step = 0
    alive = np.ones(G, bool)
    with torch.no_grad():
        while alive.any():
            x = torch.from_numpy(stacks).to(dev)
            z = net.embed(x)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z16 = net.embed(x)
            if tok is None:
                tok = z[:, None].repeat(1, W, 1); tok16 = z16[:, None].repeat(1, W, 1)
            else:
                tok = torch.cat([tok[:, 1:], z[:, None]], 1); tok16 = torch.cat([tok16[:, 1:], z16[:, None]], 1)
            idx = torch.from_numpy(np.flatnonzero(alive)).to(dev)
            T, T16 = tok[idx], tok16[idx]
            full = net.adv(net.trunk(T, last_only=False)); fast = net.adv(net.trunk(T, last_only=True))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q16 = net.adv(net.trunk(T16)).float()
            a = full.argmax(1)
            acc["n"] += len(idx)
            acc["fast_agree"] += int((fast.argmax(1) == a).sum()); acc["fast_maxdiff"] = max(acc["fast_maxdiff"], float((fast - full).abs().max()))
            acc["bf16_agree"] += int((q16.argmax(1) == a).sum()); acc["bf16_maxdiff"] = max(acc["bf16_maxdiff"], float((q16 - full).abs().max()))
            if step % 4 == 0:
                Tc = T.cpu()
                fc, sc = cpu.adv(cpu.trunk(Tc, last_only=True)), cpu.adv(cpu.trunk(Tc, last_only=False))
                acc["n_cpu"] += len(idx); acc["cpu_agree"] += int((fc.argmax(1) == sc.argmax(1)).sum())
                acc["cpu_maxdiff"] = max(acc["cpu_maxdiff"], float((fc - sc).abs().max()))
            for j, i in enumerate(idx.tolist()):
                s, _, d, _ = games[i].step(int(a[j]))
                stacks[i] = s
                if d:
                    alive[i] = False
            step += 1
    res = {k: v for k, v in acc.items()}
    res.update(fast_agree_rate=acc["fast_agree"] / acc["n"], bf16_agree_rate=acc["bf16_agree"] / acc["n"],
               cpu_agree_rate=acc["cpu_agree"] / max(acc["n_cpu"], 1), secs=round(time.time() - t0),
               games=[g.result(S0 + i) for i, g in enumerate(games)])
    Path(E("OUT", "runs/m9/verify.json")).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "games"}, indent=1))
    print("clears", sum(r["cleared"] for r in res["games"]), "/", G)


if __name__ == "__main__":
    main()
