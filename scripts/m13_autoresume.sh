#!/bin/bash
# Boot-time auto-resume for M13 (NOT enabled by default). Resumes training iff runs/m13/deadline is in the future,
# no runs/m13/STOP, no runs/m13/final.json, and m13_ppo is not already running. Also restarts the tensorboard/tbsync
# windows if missing. Log: runs/m13/autoresume.log.
# Enable (pick one):
#   cron:    (crontab -l 2>/dev/null; echo "@reboot $HOME/HermesWorkspace/mspacman-clear/scripts/m13_autoresume.sh") | crontab -
#   systemd: cp scripts/m13_autoresume.service ~/.config/systemd/user/ && systemctl --user enable m13_autoresume.service
#            (+ `loginctl enable-linger $USER` so it runs at boot before login)
# Disable: remove the crontab line / systemctl --user disable m13_autoresume.service
sleep ${AUTORESUME_DELAY:-90}                      # let the GPU driver / network come up
cd "$HOME/HermesWorkspace/mspacman-clear" || exit 1
exec >> runs/m13/autoresume.log 2>&1
echo "=== $(date '+%F %T') autoresume check"
[ -f runs/m13/deadline ] || { echo "no deadline file; not started yet"; exit 0; }
[ "$(date +%s)" -lt "$(cat runs/m13/deadline)" ] || { echo "deadline passed"; exit 0; }
[ -f runs/m13/STOP ] && { echo "STOP present"; exit 0; }
[ -f runs/m13/final.json ] && { echo "final.json present"; exit 0; }
pgrep -f "scripts/m13_ppo.py" >/dev/null && { echo "already running"; exit 0; }
tmux has-session -t msclear 2>/dev/null || tmux new-session -d -s msclear -n shell
tmux list-windows -t msclear -F '#W' | grep -qx tensorboard || scripts/tmux_job.sh tensorboard runs/tb/server.log "$HOME/HermesWorkspace/.tbvenv/bin/tensorboard --logdir runs/tb --port 6006 --bind_all"
pgrep -f "scripts/tb_sync.py" >/dev/null || scripts/tmux_job.sh tbsync runs/tb/sync.log "$HOME/HermesWorkspace/.tbvenv/bin/python scripts/tb_sync.py --every 60"
scripts/tmux_job.sh m13run runs/m13/driver.log 'bash scripts/m13_run.sh'
echo "resumed m13run"
