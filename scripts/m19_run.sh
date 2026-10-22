#!/bin/bash
# M19 (PC, 2026-10-16): M18 + dot-distance shaping on boards 1-2 endgame (<=60 left, 0.01/px, no jump penalty on eating),
# 30/70 b1/b2 starts with 70% of b2 starts from mid/late harvested states. Init M18 s1 best.pt (upd 3450) + s1 ckpt_last value head.
cd "$(dirname "$0")/.."
export DEADLINE=$(cat runs/m19/deadline) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$PWD
export RUN=runs/m19 INIT=runs/m18_s1/best.pt INIT_VH=runs/m18_s1/ckpt_last.pt START_UPDATE=3450 START_STEPS=30912000
export ENV_WORKERS=18 PER_WORKER=6 EVAL_WORKERS=8 EVAL_EVERY=100 CKPT_S=120 N_B1=200 VWARM=30 BETA=0.1 LR_PI=3e-5
export P_B1=0.30 P_B2=0.70 P_B2H=0.7 B2_EVERY=40 SHAPE_K=0.01 SHAPE_LEFT=60 SHAPE_MIN_BOARD=1 SHAPE_SKIP_EAT=1
.venv/bin/python scripts/m18_ppo.py; rc=$?; echo "=== exit $rc"
