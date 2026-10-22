"""M23 data collection: M22 (actor=teacher) or a student (actor=student, DAgger) plays waves of games from a
board-1 fresh / board-2 snapshot start mix with eps-random actions; every step is labelled with the M22 EMA
policy logits (adv/TAU) and the M22 separate-value-net value. Saved per game in overlapping segments (SEG new
frames + up to CTX frames of preceding context), PNG colour 128x128 + PNG gray 84x84, in pickled shards.

Resumable: shards are written atomically and listed in index.jsonl; state.json holds the last finished wave.
Usage: m23_collect.py OUT_DIR TARGET_STEPS [--actor student --student ckpt.pt] [--eps 0.03]
"""
import argparse, concurrent.futures as cf, glob, json, os, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, ".")
from msclear import m23_env as E

p = argparse.ArgumentParser()
p.add_argument("out"); p.add_argument("target", type=int)
p.add_argument("--teacher", default="runs/m23/m22_teacher.pt"); p.add_argument("--actor", default="teacher")
p.add_argument("--student", default=None); p.add_argument("--eps", type=float, nargs="+", default=[0.02, 0.05])
p.add_argument("--games", type=int, default=96); p.add_argument("--workers", type=int, default=14)
p.add_argument("--cap", type=int, default=3000); p.add_argument("--p_b1", type=float, default=0.5)
p.add_argument("--seed_base", type=int, default=6_000_000); p.add_argument("--metrics", default=None)
a = p.parse_args()
SEG, CTX, SHARD = 256, 31, 16384
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); dev = "cuda"
b2 = sorted(glob.glob("runs/multiboard/states/*_board2.pkl") + glob.glob("runs/multiboard_m10/states/*_board2.pkl"))
teacher = E.load_teacher(a.teacher, dev)
student = None
if a.actor == "student":
    from msclear.m23_net import load_student
    student = load_student(a.student, dev)
st = json.load(open(OUT / "state.json")) if (OUT / "state.json").exists() else dict(wave=-1)
idx = [json.loads(l) for l in open(OUT / "index.jsonl")] if (OUT / "index.jsonl").exists() else []
total = sum(r["n"] for r in idx); nshard = len(idx)
for f in OUT.glob("*.tmp"):
    f.unlink()
mlog = open(a.metrics or OUT / "metrics.jsonl", "a")
print(f"COLLECT out={OUT} actor={a.actor} have={total} target={a.target} resume_wave={st['wave'] + 1} b2_files={len(b2)}", flush=True)
ex = cf.ThreadPoolExecutor(8)
import cv2
enc = lambda im: cv2.imencode(".png", im, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()
pending = []; shard = None


def new_shard():
    return dict(col=[], gray=[], tl=[], tv=[], act=[], seg_off=[0], seg_ctx=[], seg_start=[], seg_meta=[])


def encode_seg(seg):
    seg["col"] = [enc(c) for c in seg["col"]]; seg["gray"] = [enc(g) for g in seg["gray"]]; return seg


def write_shard(force=False):
    global shard, nshard, total
    if shard is None or (len(shard["col"]) < SHARD and not force) or not shard["col"]:
        return
    n = len(shard["col"]); new = n - int(sum(shard["seg_ctx"]))
    d = dict(shard); d["tl"] = np.concatenate(d["tl"]).astype(np.float16); d["tv"] = np.concatenate(d["tv"]).astype(np.float32)
    d["act"] = np.concatenate(d["act"]).astype(np.uint8); d["seg_off"] = np.asarray(d["seg_off"], np.int64)
    d["seg_ctx"] = np.asarray(d["seg_ctx"], np.int16); d["seg_start"] = np.asarray(d["seg_start"], bool)
    name = f"shard_{nshard:05d}.pkl"; tmp = OUT / (name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(d, f, protocol=5)
    os.replace(tmp, OUT / name)
    rec = dict(file=name, n=new, frames=n, segs=len(d["seg_ctx"]), bytes=os.path.getsize(OUT / name), t=time.time())
    with open(OUT / "index.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    nshard += 1; total += new; shard = new_shard()


def drain(block=False):
    global shard
    keep = []
    for fu in pending:
        if block or fu.done():
            seg = fu.result(); s = shard
            s["col"] += seg["col"]; s["gray"] += seg["gray"]; s["tl"].append(seg["tl"]); s["tv"].append(seg["tv"]); s["act"].append(seg["act"])
            s["seg_off"].append(s["seg_off"][-1] + len(seg["col"])); s["seg_ctx"].append(seg["ctx"]); s["seg_start"].append(seg["start"]); s["seg_meta"].append(seg["meta"])
            write_shard()
        else:
            keep.append(fu)
    pending[:] = keep


shard = new_shard(); pool = E.Pool(a.workers); wave = st["wave"] + 1; t_start = time.time(); s_start = total
while total < a.target:
    rng = np.random.default_rng(a.seed_base + wave); eps = a.eps[wave % len(a.eps)]
    specs = [(a.seed_base + wave * 1000 + i, None if rng.random() < a.p_b1 else b2[int(rng.integers(len(b2)))]) for i in range(a.games)]
    buf = {i: dict(col=[], gray=[], tl=[], tv=[], act=[], ctx=0, t0=0) for i in range(a.games)}

    def flush(i, final=False):
        b = buf[i]; nnew = len(b["col"]) - b["ctx"]
        if nnew <= 0:
            return
        seg = dict(col=b["col"], gray=b["gray"], tl=np.stack(b["tl"]), tv=np.asarray(b["tv"]), act=np.asarray(b["act"]),
                   ctx=b["ctx"], start=b["t0"] == 0, meta=(wave, i, specs[i][0], os.path.basename(specs[i][1] or "fresh"), b["t0"]))
        pending.append(ex.submit(encode_seg, seg))
        k = min(CTX, len(b["col"])); t0 = b["t0"] + len(b["col"]) - k
        buf[i] = dict(col=b["col"][-k:], gray=b["gray"][-k:], tl=b["tl"][-k:], tv=b["tv"][-k:], act=b["act"][-k:], ctx=k, t0=t0)

    def on_step(gids, col, gray, tl, tv, act):
        for j, i in enumerate(gids.tolist()):
            b = buf[i]; b["col"].append(col[j].copy()); b["gray"].append(gray[j].copy()); b["tl"].append(tl[j]); b["tv"].append(float(tv[j])); b["act"].append(int(act[j]))
            if len(b["col"]) - b["ctx"] >= SEG:
                flush(i)
        drain()

    t0 = time.time()
    res = E.rollout(pool, specs, teacher, student, actor=a.actor, eps=eps, cap=a.cap, rng=rng, on_step=on_step)
    for i in range(a.games):
        flush(i, final=True)
    drain(block=True)
    steps = sum(r["steps"] for r in res); dt = time.time() - t0
    m = dict(type="collect", wave=wave, eps=eps, steps=steps, total=total, sps=round(steps / dt, 1),
             b1_clean=float(np.mean([r["clean"] for r in res if r["b0"] == 1] or [np.nan])),
             b2_clean=float(np.mean([r["clean"] for r in res if r["b0"] == 2] or [np.nan])),
             clear=float(np.mean([r["cleared"] for r in res])), deaths=float(np.mean([r["deaths"] for r in res])),
             agree=float(np.mean([r["agree"] for r in res])) if student is not None else None, t=time.time())
    mlog.write(json.dumps(m) + "\n"); mlog.flush()
    json.dump(dict(wave=wave), open(OUT / "state.json.tmp", "w")); os.replace(OUT / "state.json.tmp", OUT / "state.json")
    rate = (total - s_start) / max(1e-6, time.time() - t_start)
    eta = (a.target - total) / max(rate, 1e-6)
    print(f"WAVE {wave} eps={eps} steps={steps} ({m['sps']}/s) saved_total={total}/{a.target} b1_clean={m['b1_clean']:.3f} "
          f"b2_clean={m['b2_clean']:.3f} deaths={m['deaths']:.2f} agree={m['agree']} disk={sum(r['bytes'] for r in [json.loads(l) for l in open(OUT / 'index.jsonl')]) / 2**30 if (OUT / 'index.jsonl').exists() else 0:.2f}GB ETA={eta / 60:.1f}min", flush=True)
    wave += 1
write_shard(force=True)
pool.close(); (OUT / "DONE").write_text(str(total)); print(f"COLLECT DONE {total} steps", flush=True)
