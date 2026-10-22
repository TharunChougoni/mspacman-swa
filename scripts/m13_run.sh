#!/bin/bash
# M13 multi-board PPO from runs/m10/best.pt (resumable). Start / resume with ONE command:
#   cd ~/HermesWorkspace/mspacman-clear && scripts/tmux_job.sh m13run runs/m13/driver.log 'bash scripts/m11_run.sh'
# Deadline written once to runs/m13/deadline: DEADLINE_AT="2026-10-13 18:00" or HOURS (default 8) from first start.
# Stop cleanly: touch runs/m13/STOP (saves ckpt_last.pt, exit 3, no retry, no post-eval). Resume: rm runs/m13/STOP, rerun.
# After the deadline: 100-game multiboard eval of runs/m13/best.pt (seeds 50000-50099) + video of the furthest game.
cd "$(dirname "$0")/.."
mkdir -p runs/m13
[ -f runs/m13/deadline ] || { if [ -n "$DEADLINE_AT" ]; then date -d "$DEADLINE_AT" +%s; else echo $(( $(date +%s) + ${HOURS:-8} * 3600 )); fi; } > runs/m13/deadline
# NOTE: env vars given before scripts/tmux_job.sh do not reach the job (tmux server env); write runs/m13/deadline yourself, e.g.: date -d "2026-10-13 18:00" +%s > runs/m13/deadline
export DEADLINE=$(cat runs/m13/deadline) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== M13 deadline $(date -d @$DEADLINE '+%F %T')"
PY=.venv/bin/python
if [ ! -f runs/m13/final.json ]; then
  for try in 1 2 3 4 5; do
    $PY scripts/m13_ppo.py && break
    rc=$?
    if [ "$rc" -eq 3 ]; then echo "=== clean STOP; not retrying"; exit 0; fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "past deadline"; break; fi
    echo "=== m13_ppo exit $rc (try $try); retrying in 120 s"; sleep 120
  done
fi
if [ -f runs/m13/best.pt ] && [ ! -f runs/multiboard_m13/multiboard.json ]; then
  CKPT=runs/m13/best.pt OUTDIR=runs/multiboard_m13 $PY scripts/m10_multiboard_eval.py
fi
if [ -f runs/multiboard_m13/multiboard.json ] && [ ! -f runs/m13/video.json ]; then
  CKPT=runs/m13/best.pt MB_JSON=runs/multiboard_m13/multiboard.json PREFIX=multiboard_m13 VIDEO_JSON=runs/m13/video.json $PY scripts/m11_video.py
fi
echo "=== M13 finished"
