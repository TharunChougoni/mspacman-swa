#!/bin/bash
# M8 endgame PPO fine-tune (resumable). Start, or resume after a power cut / reboot, with ONE command:
#   cd ~/HermesWorkspace/mspacman-clear && scripts/tmux_job.sh m8 runs/m8/driver.log 'bash scripts/m8_run.sh'
# Training resumes from runs/m8/ckpt_last.pt; the wall-clock deadline is fixed at first start (runs/m8/deadline).
# After training: 300-game confirm of runs/m8/best.pt (seeds 50000-50299, same as M6/M7), video if clear >= 10%.
cd "$(dirname "$0")/.."
mkdir -p runs/m8
HOURS=${HOURS:-8}
[ -f runs/m8/deadline ] || echo $(( $(date +%s) + ${HOURS%.*} * 3600 )) > runs/m8/deadline
export DEADLINE=$(cat runs/m8/deadline)
echo "=== M8 deadline $(date -d @$DEADLINE '+%F %T')"
PY=.venv/bin/python
if [ ! -f runs/m8/final.json ]; then
  for try in 1 2 3 4 5; do
    $PY scripts/m8_ppo.py && break
    rc=$?
    if [ "$rc" -eq 3 ]; then echo "=== clean STOP requested; not retrying"; exit 0; fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "past deadline, not retrying"; break; fi
    echo "=== m8_ppo exit $rc (try $try); retrying in 120 s"; sleep 120
  done
fi
if [ -f runs/m8/best.pt ] && [ ! -f runs/m8/confirm.json ]; then
  echo "=== confirm: 300 held-out games (seeds 50000-50299) of runs/m8/best.pt"
  CKPT=runs/m8/best.pt SEED0=50000 N=300 OUT=runs/m8/confirm.json WORKERS=16 $PY scripts/m6_eval.py
fi
if [ -f runs/m8/confirm.json ] && [ ! -f runs/m8/video.json ]; then
  CR=$($PY -c "import json;print(json.load(open('runs/m8/confirm.json'))['clear_rate'])")
  if $PY -c "import sys;sys.exit(0 if $CR >= 0.10 else 1)"; then
    echo "=== confirm clear $CR >= 0.10: recording video"
    CKPT=runs/m8/best.pt PREFIX=board_clear_pixel_m8 VIDEO_JSON=runs/m8/video.json $PY scripts/m6_video.py
  else
    echo "=== confirm clear $CR < 0.10: no video"
  fi
fi
echo "=== M8 finished"
