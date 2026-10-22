#!/bin/bash
# M11 multi-board PPO from runs/m10/best.pt (resumable). Start / resume with ONE command:
#   cd ~/HermesWorkspace/mspacman-clear && scripts/tmux_job.sh m11run runs/m11/driver.log 'bash scripts/m11_run.sh'
# Deadline fixed at first start: now + ${HOURS:-8} h in runs/m11/deadline (rm it + restart for a new one).
# Stop cleanly: touch runs/m11/STOP (saves ckpt_last.pt, exit 3, no retry, no post-eval). Resume: rm runs/m11/STOP, rerun.
# After the deadline: 100-game multiboard eval of runs/m11/best.pt (seeds 50000-50099) + video of the furthest game.
cd "$(dirname "$0")/.."
mkdir -p runs/m11
[ -f runs/m11/deadline ] || echo $(( $(date +%s) + ${HOURS:-8} * 3600 )) > runs/m11/deadline
export DEADLINE=$(cat runs/m11/deadline) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== M11 deadline $(date -d @$DEADLINE '+%F %T')"
PY=.venv/bin/python
if [ ! -f runs/m11/final.json ]; then
  for try in 1 2 3 4 5; do
    $PY scripts/m11_ppo.py && break
    rc=$?
    if [ "$rc" -eq 3 ]; then echo "=== clean STOP; not retrying"; exit 0; fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "past deadline"; break; fi
    echo "=== m11_ppo exit $rc (try $try); retrying in 120 s"; sleep 120
  done
fi
if [ -f runs/m11/best.pt ] && [ ! -f runs/multiboard_m11/multiboard.json ]; then
  CKPT=runs/m11/best.pt OUTDIR=runs/multiboard_m11 $PY scripts/m10_multiboard_eval.py
fi
if [ -f runs/multiboard_m11/multiboard.json ] && [ ! -f runs/m11/video.json ]; then
  CKPT=runs/m11/best.pt MB_JSON=runs/multiboard_m11/multiboard.json $PY scripts/m11_video.py
fi
echo "=== M11 finished"
