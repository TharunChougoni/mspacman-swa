#!/bin/bash
# M12 multi-board PPO from runs/m10/best.pt (resumable). Start / resume with ONE command:
#   cd ~/HermesWorkspace/mspacman-clear && scripts/tmux_job.sh m12run runs/m12/driver.log 'bash scripts/m11_run.sh'
# Deadline: absolute time DEADLINE_AT (default 2026-10-13 05:05 IST) written once to runs/m12/deadline.
# Stop cleanly: touch runs/m12/STOP (saves ckpt_last.pt, exit 3, no retry, no post-eval). Resume: rm runs/m12/STOP, rerun.
# After the deadline: 100-game multiboard eval of runs/m12/best.pt (seeds 50000-50099) + video of the furthest game.
cd "$(dirname "$0")/.."
mkdir -p runs/m12
[ -f runs/m12/deadline ] || date -d "${DEADLINE_AT:-2026-10-13 05:05}" +%s > runs/m12/deadline   # training ends here; post-eval + video take ~2-3 min
export DEADLINE=$(cat runs/m12/deadline) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== M12 deadline $(date -d @$DEADLINE '+%F %T')"
PY=.venv/bin/python
if [ ! -f runs/m12/final.json ]; then
  for try in 1 2 3 4 5; do
    $PY scripts/m12_ppo.py && break
    rc=$?
    if [ "$rc" -eq 3 ]; then echo "=== clean STOP; not retrying"; exit 0; fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "past deadline"; break; fi
    echo "=== m12_ppo exit $rc (try $try); retrying in 120 s"; sleep 120
  done
fi
if [ -f runs/m12/best.pt ] && [ ! -f runs/multiboard_m12/multiboard.json ]; then
  CKPT=runs/m12/best.pt OUTDIR=runs/multiboard_m12 $PY scripts/m10_multiboard_eval.py
fi
if [ -f runs/multiboard_m12/multiboard.json ] && [ ! -f runs/m12/video.json ]; then
  CKPT=runs/m12/best.pt MB_JSON=runs/multiboard_m12/multiboard.json PREFIX=multiboard_m12 VIDEO_JSON=runs/m12/video.json $PY scripts/m11_video.py
fi
echo "=== M12 finished"
