#!/bin/bash
# Resumable M14 distill: rerun anytime (resumes from runs/m14/ckpt_last.pt; data packed once in runs/m14/data).
cd ~/HermesWorkspace/mspacman-clear
export PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RUN=runs/m14
.venv/bin/python -u scripts/m14_distill.py; rc=$?
echo "m14_distill exit $rc"
