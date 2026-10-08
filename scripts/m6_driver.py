"""M6 driver: one command that runs (or resumes) the whole pipeline.

  scripts/tmux_job.sh m6 runs/m6/driver.log '.venv/bin/python scripts/m6_driver.py'

Stages (marker runs/m6/.done/<stage> once finished; finished stages are skipped on rerun):
  demos -> pack_demos -> bc -> [dagger_rK -> pack_rK -> train_rK] x ROUNDS -> confirm -> video
Inside a stage, work resumes too: collection skips shards on disk, training resumes from
ckpt_last.pt. After each stage a summary goes to docs/m6/<stage>.json and is committed.
The wall-clock deadline is fixed at the first start (runs/m6/driver_state.json): training stops
at DEADLINE - RESERVE_MIN, later DAgger rounds are skipped if they cannot fit, confirm/video always run.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
M6 = Path("runs/m6"); DONE = M6 / ".done"; DOCS = Path("docs/m6")
PY = str(ROOT / ".venv/bin/python")
E = os.environ.get
HOURS = float(E("HOURS", "8"))
RESERVE_MIN = float(E("RESERVE_MIN", "25"))
ROUNDS = int(E("ROUNDS", "3"))
BETAS = [float(b) for b in E("BETAS", "0.3,0.1,0.0").split(",")]
DEMO_SHARDS = int(E("DEMO_SHARDS", "160"))
DAGGER_SHARDS = int(E("DAGGER_SHARDS", "40"))
BC_STEPS = int(E("BC_STEPS", "60000"))
R_STEPS = int(E("R_STEPS", "20000"))
R_MIN = float(E("R_MIN", "60"))           # do not start a DAgger round with less than this many minutes left
CONFIRM_N = int(E("CONFIRM_N", "300"))
GATE_AGREE = float(E("GATE_AGREE", "0.6"))


def log(m):
    print(f"[driver {time.strftime('%F %T')}] {m}", flush=True)


def state():
    p = M6 / "driver_state.json"
    if p.exists():
        return json.loads(p.read_text())
    s = dict(started=time.time(), deadline=time.time() + HOURS * 3600)
    M6.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(s))
    return s


def done(stage):
    return (DONE / stage).exists()


def mark(stage, summary=None):
    DONE.mkdir(parents=True, exist_ok=True)
    if summary is not None:
        DOCS.mkdir(parents=True, exist_ok=True)
        (DOCS / f"{stage}.json").write_text(json.dumps(summary, indent=1))
        subprocess.run(["git", "add", str(DOCS)], check=False)
        subprocess.run(["git", "commit", "-q", "-m", f"M6 stage {stage}: {short(summary)}"], check=False)
    (DONE / stage).write_text(time.strftime("%F %T"))
    log(f"stage {stage} done")
    if E("STOP_AFTER") == stage:
        log(f"STOP_AFTER={stage}: stopping here"); sys.exit(0)


def short(s):
    if "best" in s and s["best"]:
        b = s["best"]
        return f"best eval clear {b['clear_rate']:.2f} items {b['eaten']:.1f} @ step {b['step']}"
    if "transitions" in s:
        return f"{s['games']} games, {s['transitions']:,} transitions, clear {s['clear_rate']:.2f}"
    if "clear_rate" in s:
        return f"clear {s['clear_rate']:.3f} ± {s['clear_se']:.3f}, items {s['eaten']:.1f}"
    return "ok"


def run(stage, cmd, **env):
    log(f"stage {stage}: {cmd} {env}")
    r = subprocess.run(cmd, shell=True, env={**os.environ, **{k: str(v) for k, v in env.items()}})
    if r.returncode:
        log(f"stage {stage} FAILED (exit {r.returncode}); stopping. Rerun the driver to resume.")
        sys.exit(r.returncode)


def best_of(run_dir):
    f = json.loads((Path(run_dir) / "final.json").read_text())
    return f


def main():
    st = state()
    deadline = st["deadline"]; train_deadline = deadline - RESERVE_MIN * 60
    log(f"started {time.ctime(st['started'])}, deadline {time.ctime(deadline)}")
    if not done("demos"):
        run("demos", f"{PY} scripts/m6_collect.py", MODE="demo", OUT=M6 / "demos", N_SHARDS=DEMO_SHARDS, GPS=8,
            SEED0=100000, WORKERS=16)
        mark("demos", json.loads((M6 / "demos/summary.json").read_text()))
    if not done("pack_demos"):
        run("pack_demos", f"{PY} -c \"from msclear.m6data import pack; pack('{M6 / 'demos'}')\"")
        mark("pack_demos")
    packs = [str(M6 / "demos/pack")]
    if not done("bc"):
        run("bc", f"{PY} scripts/m6_train.py", RUN=M6 / "bc", PACKS=",".join(packs), STEPS=BC_STEPS,
            INIT=E("BC_INIT", "archive"), DEADLINE=train_deadline)
        mark("bc", best_of(M6 / "bc"))
    prev = M6 / "bc"
    fb = best_of(prev)
    if fb["last_val"] and fb["last_val"]["ema"]["agree"] < GATE_AGREE:
        log(f"GATE: BC validation agreement {fb['last_val']['ema']['agree']:.3f} < {GATE_AGREE}; not starting DAgger")
        rounds = 0
    else:
        rounds = ROUNDS
    for k in range(1, rounds + 1):
        rd, tr = M6 / f"dagger_r{k}", M6 / f"train_r{k}"
        if done(f"train_r{k}"):
            prev = tr; packs.append(str(rd / "pack")); continue
        left = (train_deadline - time.time()) / 60
        if not done(f"dagger_r{k}") and left < R_MIN:
            log(f"round {k}: only {left:.0f} min left before the training deadline, skipping the remaining rounds")
            break
        student = prev / "best.pt"
        if not done(f"dagger_r{k}"):
            run(f"dagger_r{k}", f"{PY} scripts/m6_collect.py", MODE="dagger", OUT=rd, N_SHARDS=DAGGER_SHARDS, GPS=8,
                SEED0=200000 + 10000 * k, WORKERS=14, STUDENT=student, BETA=BETAS[min(k - 1, len(BETAS) - 1)])
            s = json.loads((rd / "summary.json").read_text())
            s["student"] = str(student)
            mark(f"dagger_r{k}", s)
        if not done(f"pack_r{k}"):
            run(f"pack_r{k}", f"{PY} -c \"from msclear.m6data import pack; pack('{rd}')\"")
            mark(f"pack_r{k}")
        packs.append(str(rd / "pack"))
        run(f"train_r{k}", f"{PY} scripts/m6_train.py", RUN=tr, PACKS=",".join(packs), STEPS=R_STEPS, INIT=student,
            LR=E("R_LR", "1.5e-4"), WARMUP=500, EVAL_EVERY=E("R_EVAL_EVERY", "5000"), DEADLINE=train_deadline,
            SEED=k)
        mark(f"train_r{k}", best_of(tr))
        prev = tr
    # overall best checkpoint across phases (by the periodic 100-game eval on seeds 0-99)
    phases = [M6 / "bc"] + [M6 / f"train_r{k}" for k in range(1, ROUNDS + 1) if (M6 / f"train_r{k}/final.json").exists()]
    cands = [(best_of(p)["best"], p) for p in phases if best_of(p).get("best")]
    b, bp = max(cands, key=lambda c: (c[0]["clear_rate"], c[0]["eaten"]))
    shutil.copyfile(bp / "best.pt", M6 / "best.pt")
    if not done("confirm"):
        log(f"overall best: {bp}/best.pt (clear {b['clear_rate']:.2f} on seeds 0-99 @ step {b['step']}) -> runs/m6/best.pt")
        run("confirm", f"{PY} scripts/m6_eval.py", CKPT=M6 / "best.pt", SEED0=50000, N=CONFIRM_N, OUT=M6 / "confirm.json",
            WORKERS=16)
        s = json.loads((M6 / "confirm.json").read_text())
        s.update(source=str(bp / "best.pt"), seeds_0_99=b)
        mark("confirm", s)
    s = json.loads((DOCS / "confirm.json").read_text())
    if not done("video"):
        if max(s["clear_rate"], b["clear_rate"]) >= 0.10:
            run("video", f"{PY} scripts/m6_video.py", CKPT=M6 / "best.pt")
            mark("video", json.loads((M6 / "video.json").read_text()))
        else:
            log(f"no video: clear rate {s['clear_rate']:.3f} < 0.10")
            mark("video")
    log("all stages done")


if __name__ == "__main__":
    main()
