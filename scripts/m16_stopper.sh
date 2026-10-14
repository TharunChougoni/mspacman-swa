#!/bin/bash
# 16:48 IST: clean stop of M16 PPO (STOP file) and the Q-labeler (STOP file + SIGINT; finished chunk files kept).
cd ~/HermesWorkspace/mspacman-clear
now=$(date +%s); t=$(date -d "16:48" +%s); [ $t -gt $now ] && sleep $((t-now))
echo "[$(date +%T)] stopping"; touch runs/m16/STOP runs/m16/qlabels/STOP; pkill -INT -f m16_qlabel.py
for i in $(seq 60); do pgrep -f "m16_ppo.py|m16_qlabel.py" >/dev/null || break; sleep 2; done
pkill -INT -f m16_qlabel.py; sleep 3; pkill -f m16_qlabel.py
pgrep -af "m16_ppo.py|m16_qlabel.py" || echo "[$(date +%T)] both exited"
rm -f runs/m16/qlabels/STOP; ls runs/m16/qlabels/c*.npz | wc -l
