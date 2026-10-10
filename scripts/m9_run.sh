#!/bin/bash
# M9 PPO run from the M8 best (resumable). Start, or resume after a power cut / reboot / STOP, with ONE command:
#   cd ~/HermesWorkspace/mspacman-clear && scripts/tmux_job.sh m9run runs/m9run/driver.log 'bash scripts/m9_run.sh'
# Training resumes from runs/m9run/ckpt_last.pt. Deadline fixed at first start: min(now + 8 h, 21:00 today) in runs/m9run/deadline.
# Stop cleanly: touch runs/m9run/STOP (saves ckpt_last.pt, exit 3, no retry, no confirm).
# After training: 300-game confirm of runs/m9run/best.pt (seeds 50000-50299, fp32 GPU fasteval), video if clear > 22.3% (M8).
cd "$(dirname "$0")/.."
mkdir -p runs/m9run
if [ ! -f runs/m9run/deadline ]; then
  A=$(( $(date +%s) + ${HOURS:-8} * 3600 )); B=$(date -d "today 21:00" +%s)
  echo $(( A < B ? A : B )) > runs/m9run/deadline
fi
export DEADLINE=$(cat runs/m9run/deadline)
echo "=== M9 run deadline $(date -d @$DEADLINE '+%F %T')"
PY=.venv/bin/python
if [ ! -f runs/m9run/final.json ]; then
  for try in 1 2 3 4 5; do
    $PY scripts/m9_ppo.py && break
    rc=$?
    if [ "$rc" -eq 3 ]; then echo "=== clean STOP requested; not retrying"; exit 0; fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "past deadline, not retrying"; break; fi
    echo "=== m9_ppo exit $rc (try $try); retrying in 120 s"; sleep 120
  done
fi
if [ -f runs/m9run/best.pt ] && [ ! -f runs/m9run/confirm.json ]; then
  echo "=== confirm: 300 held-out games (seeds 50000-50299) of runs/m9run/best.pt (fp32 GPU fasteval)"
  CKPT=runs/m9run/best.pt SEED0=50000 N=300 OUT=runs/m9run/confirm.json WORKERS=14 BF16=0 $PY scripts/m9_fasteval.py
fi
if [ -f runs/m9run/confirm.json ] && [ ! -f runs/m9run/video.json ]; then
  CR=$($PY -c "import json;print(json.load(open('runs/m9run/confirm.json'))['clear_rate'])")
  if $PY -c "import sys;sys.exit(0 if $CR > 0.223 else 1)"; then
    echo "=== confirm clear $CR > 0.223 (M8): recording video"
    CKPT=runs/m9run/best.pt PREFIX=board_clear_pixel_m9 VIDEO_JSON=runs/m9run/video.json $PY scripts/m6_video.py
  else
    echo "=== confirm clear $CR <= 0.223 (M8): no video"
  fi
fi
echo "=== M9 run finished"
