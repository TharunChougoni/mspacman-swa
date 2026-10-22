#!/bin/bash
# M20 (PC 2026-10-16 15:4x): M18 s1 recipe from M18 best (upd 3450) but evaluate/save an EMA (0.99/update) of the policy:
# re-eval showed iterates fluctuate 0.32-0.39 true clean rate around best.pt; averaging should keep the good ones.
cd "$(dirname "$0")/.."
export DEADLINE=$(cat runs/m20/deadline) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$PWD
export RUN=runs/m20 INIT=runs/m18_s1/best.pt INIT_VH=runs/m18_s1/ckpt_last.pt START_UPDATE=3450 START_STEPS=30912000
export ENV_WORKERS=18 PER_WORKER=6 EVAL_WORKERS=8 EVAL_EVERY=60 CKPT_S=120 N_B1=200 VWARM=30 BETA=0.1 LR_PI=3e-5 EMA=0.99
export P_B1=0.50 P_B2=0.50 P_B2H=0.5
.venv/bin/python scripts/m18_ppo.py; echo "=== exit $?"
