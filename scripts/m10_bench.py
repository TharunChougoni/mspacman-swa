"""Bench: M9 SWAv2 fasteval vs M10 KV-cache eval (fp32), same 100 seeds; plus pure net step/s for rollout batch N.
Env: N (100), SEED0 (0), CK9, CK10 (default: M10 init from M9 weights)."""
import json, os, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msclear.fasteval import FastEvaluator, _strict_fp32
from msclear.m10_fasteval import M10Evaluator
from msclear.m10_net import from_swav2
from msclear.swa_net import SWAv2

ck = torch.load(os.environ.get("CK9", "runs/m9run/best.pt"), map_location="cpu", weights_only=False)
m9 = SWAv2(**ck["arch"]); m9.load_state_dict(ck["model"]); m9 = m9.cuda().eval()
m10 = from_swav2(ck["model"], ck["arch"])[0].cuda().eval()
if os.environ.get("CK10"):
    c10 = torch.load(os.environ["CK10"], map_location="cpu", weights_only=False); m10.load_state_dict(c10["model"])
out = {}
# pure net throughput, batch B, strict fp32 (eval) and TF32 (training-time rollout)
for B in (42, 100, 256):
    for mode in ("fp32", "tf32"):
        ctx = _strict_fp32() if mode == "fp32" else torch.backends.cudnn.flags(enabled=True)
        torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"; torch.backends.cudnn.allow_tf32 = mode == "tf32"
        with torch.no_grad(), ctx:
            x = torch.randint(0, 255, (B, 4, 84, 84), dtype=torch.uint8, device="cuda")
            tok = m9.embed(x)[:, None].repeat(1, 8, 1); cache = m10.new_cache(B, "cuda")
            def s9():
                global tok
                z = m9.embed(x); tok = torch.cat([tok[:, 1:], z[:, None]], 1); return m9.head(tok).argmax(-1)
            def s10():
                return m10.adv(m10.step(cache, m10.embed(x))).argmax(-1)
            for name, f in (("m9", s9), ("m10", s10)):
                for _ in range(20): f()
                torch.cuda.synchronize(); t = time.time(); n = 300
                for _ in range(n): f()
                torch.cuda.synchronize(); out[f"net_{name}_B{B}_{mode}_steps_per_s"] = round(n * B / (time.time() - t))
print(json.dumps(out, indent=1), flush=True)
seeds = list(range(int(os.environ.get("SEED0", "0")), int(os.environ.get("SEED0", "0")) + int(os.environ.get("N", "100"))))
for name, cls, net in (("m9_fasteval", FastEvaluator, m9), ("m10_kv", M10Evaluator, m10)):
    with cls(workers=14) as ev:
        t = time.time(); s, res = ev.evaluate(net, seeds); dt = time.time() - t
    steps = sum(r["steps"] for r in res)
    out[name] = dict(wall=round(dt, 1), env_steps_per_s=round(steps / dt), clear_rate=s["clear_rate"], eaten=s["eaten"])
    print(name, out[name], flush=True)
Path("runs/m10").mkdir(parents=True, exist_ok=True)
Path(os.environ.get("OUT", "runs/m10/bench.json")).write_text(json.dumps(out, indent=1))
