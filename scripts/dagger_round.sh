#!/bin/bash
# DAgger round N (resumable; rerun anytime): label with the student ckpt, then retrain on demos + all DAgger labels so far.
# Usage: ROUND=1 CKPT=runs/m14/dagger_init.pt bash scripts/dagger_round.sh
cd ~/HermesWorkspace/mspacman-clear
export PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROUND=${ROUND:-1}; CKPT=${CKPT:-runs/m14/dagger_init.pt}; D=runs/dagger$ROUND
mkdir -p $D
if [ ! -f $D/labels_done ]; then
  CKPT=$CKPT LABEL_DIR=$D/labels P_LABEL=${P_LABEL:-0.3} TARGET_LABELS=${TARGET_LABELS:-100000} WORKERS=${WORKERS:-18} \
    MIXMAX=1 HYST=0.5 CATCH=6 RANDDEF=0.15 H=40 R=6 COMMIT=2 TOPUP=2 CAP=3000 \
    .venv/bin/python -u scripts/dagger_label.py $D/label_summary.json @runs/teacher/demo_jobs.txt || exit 1
  touch $D/labels_done
fi
DIRS=runs/teacher/demos; for r in $(seq 1 $ROUND); do DIRS=$DIRS,runs/dagger$r/labels; done
RUN=runs/m14d$ROUND INIT=$CKPT DEMO_DIR=$DIRS W_PD=0.1 WD=0.05 P_DAG=0.5 MAX_STEPS=${TRAIN_STEPS:-8000} EVAL_EVERY=1000 \
  .venv/bin/python -u scripts/m14_distill.py
echo "dagger round $ROUND train exit $?"
