"""M23 fp32 multiseed eval (same protocol as scripts/multiseed_eval.py: board-1 set runs/m19/b1_test, first 200 of the
board-2 set runs/m17/b2_test, REP=3 sticky seeds k*1000+i, snapshot = file i, until the start board is cleared,
game over, or 8000 decisions). The M22 teacher runs alongside every student for argmax agreement.
Resumable: one json per (model, board, rep) in OUT. Usage: m23_eval.py OUT name=ckpt|m22 [...]"""
import glob, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, ".")
from msclear import m23_env as E
from msclear.m23_net import load_student
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True); REP = int(os.environ.get("REP", "3")); dev = "cuda"
sets = dict(b1=sorted(glob.glob("runs/m19/b1_test/*.pkl")), b2=sorted(glob.glob("runs/m17/b2_test/*.pkl"))[:200])
teacher = E.load_teacher(os.environ.get("TEACHER", "runs/m23/m22_teacher.pt"), dev); pool = E.Pool(int(os.environ.get("WORKERS", "14")))
summary = json.load(open(OUT / "summary.json")) if (OUT / "summary.json").exists() else {}
for spec in sys.argv[2:]:
    name, _, path = spec.partition("=")
    student = None if path == "m22" else load_student(path, dev)
    r = {}
    for b, files in sets.items():
        reps = []
        for k in range(1, REP + 1):
            f = OUT / f"{name}_{b}_{k}.json"
            if not f.exists():
                t0 = time.time()
                res = E.rollout(pool, [(k * 1000 + i, files[i]) for i in range(len(files))], teacher, student,
                                actor="teacher" if student is None else "student", cap=8000,
                                progress=lambda t, n: print(f"  {name} {b} rep{k} step {t} alive {n}", flush=True))
                json.dump(res, open(f, "w")); print(f"EVAL {name} {b} rep{k} clean={np.mean([x['clean'] for x in res]):.3f} ({time.time() - t0:.0f}s)", flush=True)
            reps.append(json.load(open(f)))
        cl = [float(np.mean([x["clean"] for x in rs])) for rs in reps]
        r[b] = dict(n=len(files), clean=[round(x, 3) for x in cl], clean_mean=round(float(np.mean(cl)), 3),
                    clear_mean=round(float(np.mean([np.mean([x["cleared"] for x in rs]) for rs in reps])), 3),
                    deaths_mean=round(float(np.mean([np.mean([x["deaths"] for x in rs]) for rs in reps])), 3),
                    agree_m22=None if student is None else round(float(np.mean([np.mean([x["agree"] for x in rs]) for rs in reps])), 3))
    r["mean_clean"] = round((r["b1"]["clean_mean"] + r["b2"]["clean_mean"]) / 2, 3); summary[name] = r
    print("SUMMARY", name, json.dumps(r), flush=True)
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    with open(OUT / "metrics.jsonl", "a") as fh:
        fh.write(json.dumps(dict(type="eval", name=name, clean_b1=r["b1"]["clean_mean"], clean_b2=r["b2"]["clean_mean"],
                                 clear_b1=r["b1"]["clear_mean"], clear_b2=r["b2"]["clear_mean"], deaths_b1=r["b1"]["deaths_mean"],
                                 deaths_b2=r["b2"]["deaths_mean"], agree_b1=r["b1"]["agree_m22"], agree_b2=r["b2"]["agree_m22"], t=time.time())) + "\n")
pool.close()
