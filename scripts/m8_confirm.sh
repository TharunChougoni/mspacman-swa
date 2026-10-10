set -u; cd ~/HermesWorkspace/mspacman-clear
PY=.venv/bin/python
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
echo "=== M8 confirm finished"
