"""M18 on Kaggle: extract bundle + M18 code, resume from a prior session's output (kernel_sources) if present, train until
DEADLINE, outputs in /kaggle/working/m18 (ckpt_last.pt, best.pt, metrics.jsonl, tb/, driver.log)."""
import glob, os, shutil, subprocess, sys, time
MODE = "full"
T0 = time.time(); HOURS = 11.5 if MODE == "full" else 0.1
def sh(c, env=None):
    print("$", c, flush=True); return subprocess.run(c, shell=True, text=True, env=env)
sh("nproc; nvidia-smi -L; df -h /kaggle/working | tail -1")
W = "/kaggle/working/ms"; RUN = "/kaggle/working/m18"; os.makedirs(W, exist_ok=True); os.makedirs(RUN, exist_ok=True)
bun = glob.glob("/kaggle/input/**/msclear_colab.tgz", recursive=True)[0]
code = glob.glob("/kaggle/input/**/msclear_code_m18.tgz", recursive=True)[0]
sh(f"tar xzf {bun} -C {W} && tar xzf {code} -C /tmp && cp -r /tmp/code/. {W}/")
prev = [p for p in glob.glob("/kaggle/input/**/m18/ckpt_last.pt", recursive=True) if not p.startswith(RUN)]
if prev and not os.path.exists(f"{RUN}/ckpt_last.pt"):
    d = os.path.dirname(sorted(prev)[-1]); print("RESUME from", d, flush=True)
    for f in ("ckpt_last.pt", "best.pt", "metrics.jsonl", "driver.log"):
        if os.path.exists(f"{d}/{f}"): shutil.copy(f"{d}/{f}", f"{RUN}/{f}")
    if os.path.isdir(f"{d}/tb"): shutil.copytree(f"{d}/tb", f"{RUN}/tb", dirs_exist_ok=True)
sh("pip install -q gymnasium==1.3.0 ale-py==0.12.0 2>&1 | tail -1")
env = dict(os.environ, RUN=RUN, PYTHONPATH=W, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
           DEADLINE=str(T0 + HOURS * 3600), ENV_WORKERS="5", PER_WORKER="14", EVAL_WORKERS="2", CKPT_S="180", EVAL_EVERY="150", VWARM="50")
if MODE != "full":
    env.update(EVAL_EVERY="15", N_B1="6", N_B2="6", CKPT_S="60", VWARM="5")
lim = int(T0 + HOURS * 3600 + 900 - time.time())   # hard stop 15 min after DEADLINE (eval/ckpt slack)
r = sh(f"cd {W} && timeout -k 60 {lim} python -u scripts/m18_ppo.py 2>&1 | tee -a {RUN}/driver.log | grep --line-buffered -vE '^\\s*$|A.L.E|Stella'; echo exit=$?", env=env)
sh(f"rm -rf {RUN}/harvest {RUN}/b1_test {W}; ls -la {RUN}; du -sh {RUN}")
