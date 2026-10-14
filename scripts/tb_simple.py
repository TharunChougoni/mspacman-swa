"""Minimal TensorBoard: runs m11/m13/m14/m15 -> runs/tb_simple/<run>, 6 shared tags, x = env steps (m14: samples).
Regenerates all event files every --every seconds."""
import json, shutil, sys, time
from pathlib import Path
from tensorboardX import SummaryWriter
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "runs/tb_simple"
RUNS = ["m11", "m13", "m14", "m15", "m16"]
EVAL = {"b1_clear": "b1_clear", "snap3_items": "snap3_items", "snap3_clear": "snap3_clear",
        "late3_items_frac": "late3_ate", "mb_lives_b3": "spare_lives_at_b3", "cur_level": "curriculum_level", "late3_clear": "late3_clear"}

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
                for k, t in EVAL.items():
                    if isinstance(d.get(k), (int, float)):
                        w.add_scalar(t, d[k], x)
                r2 = d.get("mb_reach2")
                if not isinstance(r2, (int, float)) and isinstance(d.get("mb_clears"), (int, float)) and isinstance(d.get("mb_reach3"), (int, float)):
                    r2 = d["mb_clears"] - d["mb_reach3"] - (d.get("mb_reach4") or 0.0)    # clears = reach2 + reach3 + reach4
                if isinstance(r2, (int, float)) and r2 > 0 and isinstance(d.get("mb_reach3"), (int, float)):
                    w.add_scalar("b2_clear", d["mb_reach3"] / r2, x)                     # P(clear board 2 | reached board 2)
            elif d.get("type") == "train" and isinstance(d.get("ent"), (int, float)):
                w.add_scalar("entropy", d["ent"], x)
        w.close()
    shutil.rmtree(OUT, ignore_errors=True); tmp.rename(OUT)

every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 120
while True:
    sync(); print(time.strftime("%H:%M:%S"), "synced", flush=True); time.sleep(every)
