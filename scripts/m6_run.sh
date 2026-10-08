#!/bin/bash
# M6 overnight run (resumable). Start or restart after a power cut with:
#   scripts/tmux_job.sh m6 runs/m6/driver.log 'bash scripts/m6_run.sh'
cd "$(dirname "$0")/.."
export HOURS=7.6 RESERVE_MIN=25 ROUNDS=4 BETAS=0.3,0.1,0,0 DEMO_SHARDS=160 DAGGER_SHARDS=40 \
       BC_STEPS=150000 R_STEPS=60000 R_MIN=50 BC_INIT=${BC_INIT:-archive} EVAL_EVERY=10000 VAL_EVERY=5000 W=8
exec .venv/bin/python scripts/m6_driver.py
