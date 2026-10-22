# M12: why the policy stalls at 60 items left in maze 2 (board 3)

Data: `scripts/m12_diag.py` on runs/m12/ckpt_last.pt (update 800; policy + value head), strict fp32 GPU, from all 122
held-out + train 60-left board-3 states (runs/m12/late_test, late_train), 1 greedy + 2 sampled (tau 0.25) rollouts each
(366 rollouts) until the first life loss, a board change, or 1500 decisions. Figures in runs/m12/diag/.

## Findings
1. **The same dots are always left: the upper-central block of maze 2 around the ghost house** (`maze2_heatmaps.png`,
   panel b). The agent clears the outer ring, the bottom and the side corridors (panel a: visited cells), then never
   commits to the interior lanes above / beside the ghost pen. Mean items eaten from 60 left: 8.4; 0 clears in 366.
2. **It dies wandering, not eating.** Deaths 289/366 (77 hit the 1500 cap). Median 69 decisions since the last item at
   death (p75 159); last-60-decision trajectories cover only ~19 distinct 4-px cells (circling the outer loop).
   Blinky is the nearest ghost in 127/289 deaths (44%); the others ~18% each.
3. **The policy knows it is lost:** entropy rises from 0.62 (first 20 decisions) to 0.89 (last 20) and the value drops
   from -1.2 to -3.9, i.e. it expects to die and has no preferred direction toward the remaining dots.
4. **The dots ARE visible to the net** (`visibility.png`, `visibility.json`): maze-2 dots share the wall colour, grey
   121 vs 146 in maze 1 on background 32; after the 84x84 INTER_AREA resize the per-dot signal is 54 (min 13) vs 73
   (min 9) in maze 1, so about 75% of the maze-1 contrast and no dot vanishes. A preprocessing change is not needed.
5. Context: ~92% of board-3 states have 0 spare lives (ram[123] & 7 == 0), so the first death there ends the game.

## Conclusion
This is an exploration / credit-assignment problem, not a perception problem. Once the outer ring is eaten, the
remaining reward lies in the ghost-dense centre, many decisions away; the item reward gives no gradient toward it
and dying costs little more than circling (value already about -4), and with 0 spare lives every attempt is the last.
Fixes, for M13: (a) potential-based shaping toward the nearest remaining dot on boards >= 3 with <= 60 left (gives
direction, does not change the optimal policy); (b) lives banking (bigger death penalty on boards 1-2, bonus per spare
life on arriving at board 3) so board 3 is played with spare lives; (c) keep the late-state curriculum
(the late pool is exactly these centre-left states).
