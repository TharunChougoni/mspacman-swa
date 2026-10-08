#!/bin/bash
# M5 pipeline: ablation -> failure analysis of the best config (runs/analysis_m5) -> new video if it clearly beats baseline.
cd "$(dirname "$0")/.."
WORKERS=16 BUDGET_S=5400 .venv/bin/python scripts/m5_ablation.py || exit 1
scripts/tmux_job.sh m5fa runs/analysis_m5/failure_analysis.log \
  'CFG="$(cat runs/m5/best.json)" OUT=runs/analysis_m5 N=320 SEED0=40000 WORKERS=14 .venv/bin/python scripts/failure_analysis.py'
