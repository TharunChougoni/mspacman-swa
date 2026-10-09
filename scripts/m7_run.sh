#!/bin/bash
# M7 PPO fine-tune (resumable). Start, or resume after a power cut / reboot, with ONE command:
#   cd ~/HermesWorkspace/mspacman-clear && scripts/tmux_job.sh ppo runs/m7/driver.log 'bash scripts/m7_run.sh'
# Training resumes from runs/m7/ckpt_last.pt; the wall-clock deadline is fixed at first start (runs/m7/deadline).
# After training: 300-game confirm of runs/m7/best.pt (seeds 50000-50299, same as M6), video if clear >= 10%.
cd "$(dirname "$0")/.."
mkdir -p runs/m7
HOURS=${HOURS:-8}
[ -f runs/m7/deadline ] || echo $(( $(date +%s) + ${HOURS%.*} * 3600 )) > runs/m7/deadline
export DEADLINE=$(cat runs/m7/deadline)
echo "=== M7 deadline $(date -d @$DEADLINE '+%F %T')"
PY=.venv/bin/python
if [ ! -f runs/m7/final.json ]; then
  for try in 1 2 3 4 5; do
    $PY scripts/m7_ppo.py && break
    rc=$?
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "past deadline, not retrying"; break; fi
    echo "=== m7_ppo exit $rc (try $try); retrying in 120 s"; sleep 120
  done
fi
if [ -f runs/m7/best.pt ] && [ ! -f runs/m7/confirm.json ]; then
  echo "=== confirm: 300 held-out games (seeds 50000-50299) of runs/m7/best.pt"
  CKPT=runs/m7/best.pt SEED0=50000 N=300 OUT=runs/m7/confirm.json WORKERS=16 $PY scripts/m6_eval.py
fi
if [ -f runs/m7/confirm.json ] && [ ! -f runs/m7/video.json ]; then
  CR=$($PY -c "import json;print(json.load(open('runs/m7/confirm.json'))['clear_rate'])")
  if $PY -c "import sys;sys.exit(0 if $CR >= 0.10 else 1)"; then
    echo "=== confirm clear $CR >= 0.10: recording video"
    CKPT=runs/m7/best.pt PREFIX=board_clear_pixel_m7 VIDEO_JSON=runs/m7/video.json $PY scripts/m6_video.py
  else
    echo "=== confirm clear $CR < 0.10: no video"
  fi
fi
echo "=== M7 finished"
