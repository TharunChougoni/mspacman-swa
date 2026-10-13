#!/bin/bash
# Resumable demo generation: rerun anytime (skips jobs whose npz exists). Usage: bash scripts/demos_run.sh
cd ~/HermesWorkspace/mspacman-clear
scripts/tmux_job.sh demos runs/teacher/demos.log 'export PYTHONPATH=$PWD; DEMO_DIR=runs/teacher/demos TOPUP=2 MIXMAX=1 HYST=0.5 CATCH=6 RANDDEF=0.15 REPLAN=2 COMMIT=2 H=40 R=6 WORKERS=18 CAP=4000 .venv/bin/python scripts/teacher_search.py runs/teacher/demos.json "@runs/teacher/demo_jobs.txt"'
