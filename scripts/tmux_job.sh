#!/bin/bash
# Usage: scripts/tmux_job.sh <window-name> <log-path> '<command string>'
# Runs the command in tmux session "msclear" (one window per job), output tee'd (-a) to the log,
# and points runs/LATEST.log at that log.  Watch: tmux attach -t msclear   or   tail -F runs/LATEST.log
set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"; WIN="$1"; LOG="$ROOT/$2"; CMD="$3"
mkdir -p "$(dirname "$LOG")" runs/.jobs
JOB="runs/.jobs/$WIN.sh"
cat > "$JOB" <<EOS
#!/bin/bash
cd "$ROOT"
export PYTHONUNBUFFERED=1
ln -sfn "$LOG" "$ROOT/runs/LATEST.log"
echo "=== \$(date '+%F %T') start [$WIN]" | tee -a "$LOG"
( $CMD ) 2>&1 | grep --line-buffered -v -E 'A\.L\.E|Stella' | tee -a "$LOG"
echo "=== \$(date '+%F %T') finished [$WIN] exit \${PIPESTATUS[0]}" | tee -a "$LOG"
exec bash
EOS
chmod +x "$JOB"
if tmux has-session -t msclear 2>/dev/null; then
  tmux new-window -d -t msclear -n "$WIN" "bash $ROOT/$JOB"
else
  tmux new-session -d -s msclear -n "$WIN" "bash $ROOT/$JOB"
fi
echo "launched tmux window msclear:$WIN -> $LOG"
