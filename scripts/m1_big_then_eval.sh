#!/bin/bash
# Bigger M1 model (16 x 1M decisions), then evaluate the M4 best config with it on the confirm seeds
# (20000-20299, as in m4_tune) and on the fixed eval seeds 0..99 (new tmux window m4big).
cd "$(dirname "$0")/.."
OUT=runs/m1_big WORKERS=16 N_DEC=${N_DEC:-1000000} .venv/bin/python scripts/m1_graph.py || exit 1
scripts/tmux_job.sh m4big runs/m4/m4big_eval.log \
  'export CFG="$(cat runs/m4/best.json)" MODEL=runs/m1_big/model.pkl WORKERS=16; TAG=m4big_confirm SEED0=20000 N=300 .venv/bin/python scripts/m3_eval.py && TAG=m4big N=100 .venv/bin/python scripts/m3_eval.py'
