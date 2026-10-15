"""Minimal TensorBoard: baselines m11/m13 + M17 -> runs/tb_simple/<run>, x = env steps. Regenerates event files every --every s.
Tags: b2_clear (M17: fixed 200 held-out board-2 snapshots; m11/m13: reach3/reach2 from multiboard evals), b1_clear, reach3,
spare_lives_at_b3, deaths_b2 (multiboard), entropy."""
import json, shutil, sys, time
from pathlib import Path
from tensorboardX import SummaryWriter
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "runs/tb_simple"
RUNS = ["m11", "m13", "m17"]


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def sync():
    tmp = OUT.with_name("tb_simple_tmp"); shutil.rmtree(tmp, ignore_errors=True)
    for r in RUNS:
        p = ROOT / "runs" / r / "metrics.jsonl"
        if not p.exists():
            continue
        w = SummaryWriter(str(tmp / r))
        for line in open(p):
            try:
                d = json.loads(line)
            except Exception:
                continue
            x = int(d.get("env_steps") or 0)
            if d.get("type") == "eval":
                b2 = d.get("b2_clear")
                if not num(b2):
                    r2 = d.get("mb_reach2")
                    if not num(r2) and num(d.get("mb_clears")) and num(d.get("mb_reach3")):
                        r2 = d["mb_clears"] - d["mb_reach3"] - (d.get("mb_reach4") or 0.0)
                    b2 = d["mb_reach3"] / r2 if num(r2) and r2 > 0 and num(d.get("mb_reach3")) else None
                for t, v in (("b2_clear", b2), ("b1_clear", d.get("b1_clear")), ("reach3", d.get("mb_reach3")),
                             ("spare_lives_at_b3", d.get("mb_lives_b3")), ("deaths_b2", d.get("mb_deaths_b2"))):
                    if num(v):
                        w.add_scalar(t, v, x)
            elif d.get("type") == "train" and num(d.get("ent")):
                w.add_scalar("entropy", d["ent"], x)
        w.close()
    shutil.rmtree(OUT, ignore_errors=True); tmp.rename(OUT)


every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 120
while True:
    sync(); print(time.strftime("%H:%M:%S"), "synced", flush=True); time.sleep(every)
