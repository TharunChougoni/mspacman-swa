#!/bin/bash
# M22 (run on Kaggle 2026-10-17, here as a local launcher): M18 recipe from M18 best (upd 3450) with a SEPARATE value network
# (SEPV=1, own CNN+trunk initialised from INIT_VH) and EMA 0.99 of the policy for eval/best. Value warm-up counts from START_UPDATE.
cd "$(dirname "$0")/.."
mkdir -p runs/m22
export DEADLINE=${DEADLINE:-$(( $(date +%s) + 11*3600 ))} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$PWD
export RUN=runs/m22 INIT=runs/m18_s1/best.pt INIT_VH=runs/m18_s1/ckpt_last.pt START_UPDATE=3450 START_STEPS=30912000
export SEPV=1 EMA=0.99 LR_PI=3e-5 LR_V=3e-4 VWARM=50 BETA=0.1 N_B1=200 EVAL_EVERY=140 P_B1=0.5 P_B2=0.5 P_B2H=0.5
.venv/bin/python scripts/m18_ppo.py; echo "=== exit $?"
