"""Rebuild TensorBoard event files for M23 from the jsonl metrics every 30 s (run with the .tbvenv python):
runs/tb_simple/m23_collect, m23_cnn, m23_vit (train/val scalars; eval tags clean_b1 / clean_b2 at the step the eval
belongs to)."""
import json, shutil, sys, time
from pathlib import Path
from tensorboardX import SummaryWriter
ROOT = Path(__file__).resolve().parent.parent; R = ROOT / "runs/m23"; OUT = ROOT / "runs/tb_simple"


def rows(p):
    if not p.exists():
        return []
    out = []
    for l in open(p):
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


def sync():
    for run, srcs in {"m23_collect": [R / "data/m22/metrics.jsonl", R / "data/dagger_cnn/metrics.jsonl", R / "data/dagger_vit/metrics.jsonl"],
                      "m23_cnn": [R / "cnn/metrics.jsonl", R / "cnn_dagger/metrics.jsonl"],
                      "m23_vit": [R / "vit/metrics.jsonl", R / "vit_dagger/metrics.jsonl"]}.items():
        tmp = OUT / (run + "_tmp"); shutil.rmtree(tmp, ignore_errors=True); w = SummaryWriter(str(tmp)); off = 0
        for src in srcs:
            last = 0
            for d in rows(src):
                tag = src.parent.name
                if d.get("type") == "collect":
                    x = int(d["total"])
                    for k in ("sps", "b1_clean", "b2_clean", "clear", "deaths", "agree", "eps"):
                        if isinstance(d.get(k), (int, float)) and d[k] == d[k]:
                            w.add_scalar(f"{tag}/{k}", d[k], x)
                elif d.get("type") in ("train", "val"):
                    x = off + int(d["step"]); last = x
                    for k, v in d.items():
                        if k not in ("type", "step", "t") and isinstance(v, (int, float)):
                            w.add_scalar(f"{d['type']}/{k}", v, x)
            off = last
        for d in rows(R / "eval/metrics.jsonl"):
            nm = d["name"]
            if nm.split("_")[0] == run.split("_")[1] or (run == "m23_cnn" and nm == "m22"):
                x = 0 if nm == "m22" else (1 if "dagger" not in nm else 2)
                for k in ("clean_b1", "clean_b2", "clear_b1", "clear_b2", "deaths_b1", "deaths_b2", "agree_b1", "agree_b2"):
                    if isinstance(d.get(k), (int, float)):
                        w.add_scalar(k if nm != "m22" else f"m22_ref/{k}", d[k], x)
        w.close(); dst = OUT / run; shutil.rmtree(dst, ignore_errors=True); tmp.rename(dst)


while True:
    try:
        sync()
    except Exception as e:
        print("sync error", e, flush=True)
    if len(sys.argv) > 1:
        break
    time.sleep(30)
