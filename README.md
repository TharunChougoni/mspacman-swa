# mspacman-clear

Goal: an agent that clears board 1 of Ms. Pac-Man (all dots eaten, board 2 loads).
Approach: Hybrid Reward Architecture (van Seijen et al. 2017) on decoded RAM objects. Tabular per-position GVFs
plus a normalised aggregator. Fallback: RAM Go-Explore → backward PPO. Full plan: `docs/plan.md`.

## Setup
```bash
cd ~/HermesWorkspace/mspacman-clear
~/.hermes/bin/uv venv --python 3.11 .venv
~/.hermes/bin/uv pip install --python .venv/bin/python gymnasium==1.3.0 ale-py==0.12.0 numpy pytest
.venv/bin/python -m pytest -q
```

## Layout
| path | what |
|---|---|
| `msclear/ram.py` | verified RAM map: positions, lives, dot bitmap + 150-dot mask, power pellets, frightened bits, level byte |
| `msclear/env.py` | `make_env()` (ALE v5, frameskip 4, sticky 0.25 by default) |
| `msclear/clear.py` | `ClearDetector`: board clear = dots ≤ 3 → refill ≥ 100 and level byte up |
| `msclear/explore.py` | RAM-cell Go-Explore (deterministic) used to obtain real clears |
| `msclear/maze.py` | M1: `PinnedGame` (ghosts held in pen), transition collection, empirical model, GVF solver |
| `msclear/hra.py` | M3: HRA agent (pellet head + ghost heads + blue-ghost heads + count term), episode runner, evaluator |
| `msclear/data/` | `maze1_dots.json` (bit → screen position for 150 dots), `clear_traj.json` (real clears of boards 1 and 2) |
| `scripts/` | one script per milestone check |
| `tests/` | pytest |
| `seeds/`, `legacy/` | kept from mspacman-rainbow (GE archives, old helpers) |
| `runs/` | outputs and logs (gitignored) |

## Milestones
| | run | output |
|---|---|---|
| M0 pellet map | `.venv/bin/python scripts/m0_pellet_map.py` | `msclear/data/maze1_dots.json`, `runs/m0/pellet_overlay.png` |
| M0 frightened | `.venv/bin/python scripts/m0_fright.py` | `runs/m0/m0_fright.json` |
| M0 clear/level | `WORKERS=16 FP_EPISODES=1600 .venv/bin/python scripts/m0_clear.py` | `runs/m0/m0_clear.json` |
| M1 graph + GVFs | `WORKERS=16 N_DEC=300000 .venv/bin/python scripts/m1_graph.py` | `runs/m1/model.pkl`, `runs/m1/m1.json` |
| M2 pellets only | `WORKERS=16 .venv/bin/python scripts/m2_pellets.py` (needs M1 model) | `runs/m2/m2.json` |
| M3 full HRA eval | `WORKERS=16 N=100 .venv/bin/python scripts/m3_eval.py` | `runs/m3/m3_eval.json` |
| M4 tuning | `WORKERS=16 .venv/bin/python scripts/m4_tune.py`, then `CFG="$(cat runs/m4/best.json)" TAG=m4 .venv/bin/python scripts/m3_eval.py` | `runs/m4/trials.jsonl`, `best.json`, `m4_eval.json` |

| M4 refine | `WORKERS=16 .venv/bin/python scripts/m4_refine.py` (extends the w_g grid around `best.json`) | `runs/m4/refine.log` |
| bigger M1 model + eval | `scripts/m1_big_then_eval.sh` | `runs/m1_big/`, `runs/m4/m4big_eval.log` |

Long runs go in tmux session `msclear`, one window per job, output `tee -a` to the log and
`runs/LATEST.log` pointing at the current job's log:
```bash
scripts/tmux_job.sh <window> runs/<milestone>/<job>.log '<command>'
tmux attach -t msclear        # Ctrl-b n / p to switch windows, Ctrl-b d to detach
tail -F runs/LATEST.log
```

## Key facts (M0)
- Bitmap `ram[59:101]`: 242 bits set at start, only **150** are dots (8 of them in `ram[100]`, the bottom-row "dashes").
  Count with `ram.dots_left()`, never the raw bit sum. Power pellets: `ram[117]` bits 2–5. A board ends when all
  154 items are eaten (`ram[119]` counts them).
- Level byte `ram[101]`: 20, 42, 64 on boards 1, 2, 3.
- Frightened: `ram[1..4]` bit 7 per ghost (99.8 % vs sprite colour); `ram[116]` is the global frightened timer
  (counts down 1 per 8 frames).
- M1 tables are the only learned artifact: transition model + GVFs from ~5M decisions of random play with ghosts
  pinned (6 min on 16 cores). The M3 agent combines them online; no emulator lookahead at play time.
