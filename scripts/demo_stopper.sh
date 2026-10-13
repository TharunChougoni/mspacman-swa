#!/bin/bash
# Sleep until 16:45, gracefully stop the demo job, summarise runs/teacher/demos.log -> runs/teacher/demos_stop.txt
cd ~/HermesWorkspace/mspacman-clear
T=$(date -d "16:45" +%s); now=$(date +%s); echo "armed: stopping demo job at 16:45 ($(( (T-now)/60 )) min)"; sleep $(( T>now ? T-now : 0 ))
echo "$(date) sending Ctrl-C"; pkill -INT -f "teacher_search.py runs/teacher/demos.json"
for i in $(seq 60); do pgrep -f teacher_search >/dev/null || break; sleep 2; done
pkill -f teacher_search; sleep 2; pkill -9 -f teacher_search
python3 - <<'P' | tee runs/teacher/demos_stop.txt
import json,glob,datetime
g=[json.loads(l) for l in open('runs/teacher/demos.log') if l.startswith('{"src"')]
b3=[x for x in g if 'board3' in x['src'] and '_l' not in x['src']]
print('stopped', datetime.datetime.now().isoformat(timespec='minutes'))
print('games_done', len(g), 'cleared', sum(x['cleared'] for x in g), 'total_decisions', sum(x['decisions'] for x in g))
print('deaths_mean', round(sum(x['deaths'] for x in g)/max(1,len(g)),2))
print('board3_starts', len(b3), 'cleared', sum(x['cleared'] for x in b3), 'clear_rate', round(sum(x['cleared'] for x in b3)/max(1,len(b3)),3))
print('npz_files', len(glob.glob('runs/teacher/demos/*.npz')))
P
