"""Multi-seed fp32 eval: each checkpoint on the fixed board-1 set (B1_DIR) and board-2 set (B2_DIR), REP times with different
sticky-action seeds (offsets 1000k; offset 0 = the training eval). Usage: multiseed_eval.py OUT.json ckpt [ckpt ...]
(ckpt may be a ckpt_last.pt: its raw 'pi' and, with suffix ':ema', its EMA weights are evaluated)."""
import glob, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, ".")
from msclear import fasteval, m13_env
from msclear.m10_fasteval import M10Evaluator
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
fasteval._MAX_STEPS = 20001
os.environ["MB_CAP"] = "8000"; os.environ["SNAP_STOP_CLEAR"] = "1"; fasteval._worker = m13_env.snap_worker
W, REP, DEV = int(os.environ.get("WORKERS", "4")), int(os.environ.get("REP", "3")), os.environ.get("DEVICE", "cuda")
b1 = sorted(glob.glob(os.environ.get("B1_DIR", "runs/m19/b1_test") + "/*.pkl")); b2 = sorted(glob.glob(os.environ.get("B2_DIR", "runs/m17/b2_test") + "/*.pkl"))[:200]
os.environ["M11_SNAPS"] = ",".join(b2); ev2 = M10Evaluator(workers=W, device=DEV)
os.environ["M11_SNAPS"] = ",".join(b1); ev1 = M10Evaluator(workers=W, device=DEV)
def load(spec):
    p, _, kind = spec.partition(":"); c = torch.load(p, map_location="cpu", weights_only=False)
    sd = c["model"] if "model" in c else (c["ema"] if kind == "ema" else c["pi"])
    tmp = f"/tmp/mse_{abs(hash(spec))}.pt"; torch.save(dict(model=sd, arch=c["arch"], kind="m10"), tmp); return tmp
out = {}
for spec in sys.argv[2:]:
    t = time.time(); ck = load(spec); r = {}
    for nm, ev, files in (("b1", ev1, b1), ("b2", ev2, b2)):
        cl, cr, dd = [], [], []
        for k in range(1, REP + 1):
            _, res = ev.evaluate(ck, list(range(k * 1000, k * 1000 + len(files))))
            cl.append(float(np.mean([g["cleared"] and g["deaths"] == 0 for g in res]))); cr.append(float(np.mean([g["cleared"] for g in res]))); dd.append(float(np.mean([g["deaths"] for g in res])))
        r[nm] = dict(n=len(files), clean=[round(x, 3) for x in cl], clean_mean=round(float(np.mean(cl)), 3), clear_mean=round(float(np.mean(cr)), 3), deaths_mean=round(float(np.mean(dd)), 3))
    r["mean_clean"] = round((r["b1"]["clean_mean"] + r["b2"]["clean_mean"]) / 2, 3); out[spec] = r
    print("MULTISEED", spec, json.dumps(r), f"({time.time() - t:.0f}s)", flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
