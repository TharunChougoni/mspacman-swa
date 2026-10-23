"""Rebuild TensorBoard event files for M23 from the jsonl metrics every 30 s (run with the .tbvenv python).

Runs under runs/tb_simple: m23_cnn, m23_vit (distill steps 0-4000, DAgger fine-tune continues after),
m23_collect (data collection / DAgger progress), m22_ref (teacher eval reference).
Tag groups: key/ (most important, sorts first), train/, val/, eval/, collect/, sys/.
A custom_scalars layout (Custom Scalars tab) overlays CNN vs ViT on the key charts.
"""
import json, shutil, sys, time
from pathlib import Path
from tensorboardX import SummaryWriter
ROOT = Path(__file__).resolve().parent.parent; R = ROOT / "runs/m23"; OUT = ROOT / "runs/tb_simple"

TRAIN = {"agree": "key/train_agree", "kl": "train/kl", "vloss": "train/vloss", "lr": "sys/lr",
         "frames_per_s": "sys/frames_per_s"}
VAL = {"val_agree": "key/val_agree", "val_kl": "key/val_kl", "val_vloss": "key/val_vloss", "val_verr": "val/verr"}
EVAL = {"clean_b1": "key/eval_clean_b1", "clean_b2": "key/eval_clean_b2", "clear_b1": "eval/clear_b1",
        "clear_b2": "eval/clear_b2", "deaths_b1": "eval/deaths_b1", "deaths_b2": "eval/deaths_b2"}
COLLECT = ("total", "sps", "clear", "deaths", "agree", "b1_clean", "b2_clean", "eps")
LAYOUT = {
    "1 key (CNN vs ViT)": {
        "val agree": ["Multiline", [r"key/val_agree"]],
        "val KL": ["Multiline", [r"key/val_kl"]],
        "val vloss": ["Multiline", [r"key/val_vloss"]],
        "train agree": ["Multiline", [r"key/train_agree"]],
        "eval clean clears board 1": ["Multiline", [r"key/eval_clean_b1"]],
        "eval clean clears board 2": ["Multiline", [r"key/eval_clean_b2"]],
    },
    "2 collect / DAgger": {
        "collected steps": ["Multiline", [r"collect/.*/total"]],
        "student agree while collecting": ["Multiline", [r"collect/.*/agree"]],
        "collect sps": ["Multiline", [r"collect/.*/sps"]],
    },
}


def rows(p):
    out = []
    if p.exists():
        for l in open(p):
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def write(run, fill):
    tmp = OUT.parent / "tb_simple_tmp" / run; shutil.rmtree(tmp, ignore_errors=True)
    w = SummaryWriter(str(tmp)); fill(w); w.close()
    dst = OUT / run; shutil.rmtree(dst, ignore_errors=True); tmp.rename(dst)


def student(name, w):
    ends = []; off = 0
    for src in (R / name / "metrics.jsonl", R / f"{name}_dagger/metrics.jsonl"):
        last = off; seen = set()
        for d in rows(src):
            t = d.get("type")
            if t not in ("train", "val") or (t, d.get("step")) in seen:
                continue
            seen.add((t, d.get("step"))); x = off + int(d["step"]); last = max(last, x)
            for k, tag in (TRAIN if t == "train" else VAL).items():
                if num(d.get(k)):
                    w.add_scalar(tag, d[k], x)
        ends.append(last); off = last
    for d in rows(R / "eval/metrics.jsonl"):
        nm = d.get("name", "")
        if nm.split("_")[0] == name:
            x = ends[1] if "dagger" in nm or "ft" in nm else ends[0]
            for k, tag in EVAL.items():
                if num(d.get(k)):
                    w.add_scalar(tag, d[k], x)
    if name == "cnn":
        w.add_custom_scalars(LAYOUT)
    return ends


def sync():
    ends = {}
    write("m23_cnn", lambda w: ends.__setitem__("cnn", student("cnn", w)))
    write("m23_vit", lambda w: ends.__setitem__("vit", student("vit", w)))

    def collect(w):
        for stage in ("m22", "dagger_cnn", "dagger_vit"):
            for d in rows(R / f"data/{stage}/metrics.jsonl"):
                if d.get("type") == "collect" and num(d.get("total")):
                    for k in COLLECT:
                        if num(d.get(k)):
                            w.add_scalar(f"collect/{stage}/{k}", d[k], int(d["total"]))
    write("m23_collect", collect)

    def ref(w):
        m = [d for d in rows(R / "eval/metrics.jsonl") if d.get("name") == "m22"]
        xmax = max([0] + [e[-1] for e in ends.values()])
        if m:
            for x in sorted({0, xmax}):
                for k, tag in EVAL.items():
                    if num(m[-1].get(k)):
                        w.add_scalar(tag, m[-1][k], x)
    write("m22_ref", ref)


while True:
    try:
        sync()
    except Exception as e:
        print("sync error", e, flush=True)
    if len(sys.argv) > 1:
        break
    time.sleep(30)
