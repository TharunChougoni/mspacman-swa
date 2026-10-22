"""Re-evaluate checkpoints on fresh sticky-action seeds: b2 = fixed 200 states x REP seed offsets, b1 = fresh seeds."""
import glob, json, os, sys, time
import numpy as np
sys.path.insert(0, ".")
from msclear import fasteval, m13_env
from msclear.m10_fasteval import M10Evaluator
import torch
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
fasteval._MAX_STEPS = 20001
os.environ["MB_CAP"] = "8000"; os.environ["SNAP_STOP_CLEAR"] = "1"; fasteval._worker = m13_env.snap_worker
b2 = sorted(glob.glob("runs/m17/b2_test/*.pkl")); os.environ["M11_SNAPS"] = ",".join(b2); ev2 = M10Evaluator(workers=8)
b1 = sorted(glob.glob("runs/m19/b1_test/*.pkl")); os.environ["M11_SNAPS"] = ",".join(b1); ev1 = M10Evaluator(workers=8)
REP = int(os.environ.get("REP", "3")); out = {}
for ck in sys.argv[1:]:
    t = time.time(); r = {}
    for nm, ev, files in (("b1", ev1, b1), ("b2", ev2, b2)):
        cl = []; cr = []
        for k in range(1, REP + 1):                   # offsets 1..REP: same start states, different sticky RNG (offset 0 = training eval)
            _, res = ev.evaluate(ck, list(range(k * 1000, k * 1000 + len(files))))
            cl.append(np.mean([g["cleared"] and g["deaths"] == 0 for g in res])); cr.append(np.mean([g["cleared"] for g in res]))
        r[nm] = dict(clean=[round(x, 3) for x in cl], clean_mean=round(float(np.mean(cl)), 3), clear_mean=round(float(np.mean(cr)), 3))
    r["mean_clean"] = round((r["b1"]["clean_mean"] + r["b2"]["clean_mean"]) / 2, 3); out[ck] = r
    print(ck, json.dumps(r), f"({time.time() - t:.0f}s)", flush=True)
json.dump(out, open("runs/m19/reeval.json", "w"), indent=1)
