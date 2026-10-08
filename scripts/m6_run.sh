#!/bin/bash
# M6 overnight run (resumable). Start, or resume after a power cut / reboot, with ONE command:
#   cd ~/HermesWorkspace/mspacman-clear && scripts/tmux_job.sh m6 runs/m6/driver.log 'bash scripts/m6_run.sh'
# Finished stages are skipped (runs/m6/.done/), collection skips shards on disk, training resumes from ckpt_last.pt.
# Auto-stop: DEADLINE_ISO. A failed driver is retried up to 3 times (2 min apart) for transient errors.
cd "$(dirname "$0")/.."
export DEADLINE_ISO=${DEADLINE_ISO:-2026-10-09T05:00:00+05:30} RESERVE_MIN=25 \
       ROUNDS=5 BETAS=0.3,0.1,0,0,0 DEMO_SHARDS=160 DAGGER_SHARDS=60 DAGGER_WEIGHT=0.5 \
       BC_STEPS=200000 R_STEPS=70000 R_MIN=50 BC_INIT=${BC_INIT:-archive} \
       W=8 EVAL_EVERY=10000 EVAL_N=100 VAL_EVERY=5000 VIDEO_AT=0.10
for try in 1 2 3 4; do
  .venv/bin/python scripts/m6_driver.py && exit 0
  rc=$?
  if [ "$(date +%s)" -ge "$(date -d "$DEADLINE_ISO" +%s)" ]; then echo "past deadline, not retrying"; exit $rc; fi
  echo "=== driver exit $rc (try $try); retrying in 120 s"; sleep 120
done
exit 1
