#!/bin/bash
# M23 pipeline: collect (M22) -> distill CNN -> distill ViT -> DAgger round for each -> fp32 multiseed eval.
# Every stage runs in its own window of tmux session "m23", tee'd to runs/m23/<stage>.log, and is resumable:
# a stage is skipped when its DONE marker exists, otherwise it resumes from its own checkpoint/state.
# Start or resume after a power cut:  scripts/m23_pipeline.sh   (watch: tmux attach -t m23)
cd "$(dirname "$0")/.."; ROOT="$PWD"; PY="$ROOT/.venv/bin/python"; D=runs/m23
mkdir -p $D
if [ -z "$M23_INNER" ]; then
  tmux has-session -t m23 2>/dev/null || tmux new-session -d -s m23 -n pipeline "M23_INNER=1 bash $ROOT/scripts/m23_pipeline.sh; exec bash"
  tmux list-windows -t m23 -F '#W' | grep -qx tbsync || tmux new-window -d -t m23 -n tbsync "cd $ROOT; $HOME/HermesWorkspace/.tbvenv/bin/python scripts/m23_tbsync.py 2>&1 | tee -a $D/tbsync.log"
  tmux has-session -t tb 2>/dev/null || tmux new-session -d -s tb "cd $ROOT; $HOME/HermesWorkspace/.tbvenv/bin/tensorboard --logdir runs/tb_simple --port 6006 --bind_all --reload_interval 30 2>&1 | tee -a $D/tb.log; exec bash"
  echo "m23 pipeline running: tmux attach -t m23 ; logs in $D/*.log ; TensorBoard http://localhost:6006"; exit 0
fi
stage() {   # stage <name> <done-marker> <command...>
  local name=$1 marker=$2; shift 2
  if [ -e "$marker" ]; then echo "[$(date '+%F %T')] skip $name (done)"; return 0; fi
  echo "[$(date '+%F %T')] start $name"
  tmux kill-window -t "m23:$name" 2>/dev/null
  tmux new-window -d -t m23 -n "$name" "cd $ROOT; export PYTHONUNBUFFERED=1; echo '=== '\$(date '+%F %T')' start $name' >> $D/$name.log; $* 2>&1 | grep --line-buffered -v -E 'A\.L\.E|Stella' | tee -a $D/$name.log; tmux wait-for -S m23_$name; exec bash"
  tmux wait-for "m23_$name"
  if [ -e "$marker" ]; then echo "[$(date '+%F %T')] done $name"; else echo "[$(date '+%F %T')] FAILED $name (see $D/$name.log)"; exit 1; fi
}
stage collect     $D/data/m22/DONE        $PY scripts/m23_collect.py $D/data/m22 ${M23_STEPS:-2000000}
stage train_cnn   $D/cnn/DONE             $PY scripts/m23_train.py $D/cnn --student cnn --data $D/data/m22 --steps ${M23_TRAIN:-4000}
stage train_vit   $D/vit/DONE             $PY scripts/m23_train.py $D/vit --student vit --data $D/data/m22 --steps ${M23_TRAIN:-4000}
stage dagger_cnn  $D/data/dagger_cnn/DONE $PY scripts/m23_collect.py $D/data/dagger_cnn ${M23_DAGGER:-400000} --actor student --student $D/cnn/best.pt --eps 0.03 --seed_base 7000000
stage ft_cnn      $D/cnn_dagger/DONE      $PY scripts/m23_train.py $D/cnn_dagger --student cnn --data $D/data/m22:1 $D/data/dagger_cnn:1 --init $D/cnn/best.pt --steps ${M23_FT:-1500} --lr 1e-4 --warmup 100
stage dagger_vit  $D/data/dagger_vit/DONE $PY scripts/m23_collect.py $D/data/dagger_vit ${M23_DAGGER:-400000} --actor student --student $D/vit/best.pt --eps 0.03 --seed_base 8000000
stage ft_vit      $D/vit_dagger/DONE      $PY scripts/m23_train.py $D/vit_dagger --student vit --data $D/data/m22:1 $D/data/dagger_vit:1 --init $D/vit/best.pt --steps ${M23_FT:-1500} --lr 1e-4 --warmup 100
stage eval        $D/eval/DONE            "$PY scripts/m23_eval.py $D/eval m22=m22 cnn=$D/cnn/best.pt vit=$D/vit/best.pt cnn_dagger=$D/cnn_dagger/best.pt vit_dagger=$D/vit_dagger/best.pt && touch $D/eval/DONE"
echo "[$(date '+%F %T')] M23 pipeline finished"; cat $D/eval/summary.json
