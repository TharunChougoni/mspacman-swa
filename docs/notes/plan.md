# Ms. Pac-Man board clear — cleanup + new approach (prepared 2026-10-08, nothing on the PC changed)

## 1. Hardware (PC "AM")
| | |
|---|---|
| GPU | RTX 3060 Laptop, 6 GB VRAM, driver 610.57, fan dead (85–89 °C accepted) |
| CPU | i7-12700H, 14 cores / 20 threads (6P+8E), 4.7 GHz max |
| RAM | 15 GiB + 15 GiB zram swap (~8.5 GiB free at check) |
| Disk | btrfs `/dev/nvme0n1p7` 198 G, **62 G free**, `compress=zstd:1`; project on subvol `@home` |
| Snapshots | snapper installed, only config `root` (subvol `@`). `@home` not in it ⇒ home deletions free space immediately. Confirm with `sudo snapper -c root list; sudo btrfs subvolume list /` (needs root, not run) |
| Project venv | Python 3.11.15, torch 2.11.0+cu128 (CUDA 12.8, cuDNN 9.19, cuda available), gymnasium 1.3.0, ale-py 0.12.0, numpy 2.4.6, pygame 2.6.1, wandb 0.28.2 |
| Other big dirs (not touched) | `mf-tracker` 9.0 G, `~/.cache/codex-runtimes` 4.1 G, `~/.cache/uv` 3.9 G (keep: new venv hardlinks from it), browser caches ~3.7 G |

## 2. Cleanup — mspacman-rainbow (11 G)
Every pixel checkpoint and pixel tape here was made with **16 frames/decision** (v5 frameskip 4 × MaxAndSkip 4) + sticky 0.25, from policies that never cleared (0 `maze_changed` across all probes). None feed the new RAM-based approach. ALE emulator-state archives are frameskip-independent and stay.

| item | size | action | reason |
|---|---|---|---|
| `.venv` | 7.0 G | DELETE | rebuild fresh with uv (hardlinks from uv cache, ~0 extra disk) |
| `experiments/non_vit/runs/ALE/*swa_rl*` (4) | 586 M | DELETE | SWA-RL collapse runs |
| `experiments/non_vit/runs/ALE/*rtc_pellet_rl*` (2) | 190 M | DELETE | pellet-TD collapse (917) |
| `experiments/non_vit/runs/ALE/*swa-vit-rl-smoke*` | 104 M | DELETE | smoke |
| `experiments/non_vit/runs/ALE/*rainbow_atari__1786867268/867323/867862/868032` | ~92 M | DELETE | Rainbow smoke runs |
| `experiments/non_vit/runs/ALE/*rainbow_atari__1786874028` | 26 M | DELETE | first half of 2M run, superseded |
| `experiments/non_vit/runs/ALE/*rainbow_atari__1786882832` | 33 M | ARCHIVE → `_archive/.../ckpts/rainbow2m_final` | 2M Rainbow teacher (eval 2006), only reference baseline |
| `experiments/non_vit/runs/{bc_teacher,swa_vit_pellet,boardclear_bc,swa_selfim}` | 91 M | DELETE | dead BC/self-imitation heads |
| `experiments/non_vit/data/*.npz` tapes (dagger_c1-3, rtc_score_c4-6, rtc_vit_*, pellet_c1, rtc_pellet_os, boardclear_bc, swa_selfim, rtc_v2_3800) | ~780 M | DELETE | 16-frame pixel tapes from non-clearing teachers; zero positive labels |
| `experiments/non_vit/data/human_session/{go_explore_archive,ge_from_c3,c3_seed_archive}.pkl` | 50 M | KEEP → `mspacman-clear/seeds/` | ALE states at leftover 101–159; possible start states (verify they `restoreState` on ale-py 0.12) |
| rest of `human_session/` (logs, prompts, pngs, jsonl; excl. play_*/state.pkl) | ~2 M | ARCHIVE (tarball) | history of GE/LLM/human attempts |
| `experiments/non_vit/data/human_clears/*.npz` + `explorer_best.npz` | 2.5 M | ARCHIVE | none cleared (pellets_min 101–233), 16-frame pixels |
| `experiments/non_vit/data/maze1_bitmap*.npz` | 15 K | KEEP → `legacy/` | partial pellet-bit→pixel map (only observed bits) |
| `experiments/non_vit/dirs/dqfd_leftover` | 158 M | ARCHIVE code / DELETE ckpts | DQfD attempt |
| `experiments/non_vit/dirs/swa_vit_pellet_ppo` | 108 M | ARCHIVE code / DELETE ckpts | ViT-PPO attempt |
| `experiments/non_vit/scripts`, `dirs/viz3d` | 1.2 M | ARCHIVE | probes, GE, record_human, LLM play |
| `runs/swa_rtc` (21 ckpts) | 663 M | DELETE except `swa_rtc_attn_c3_300k.pt` (32 M ARCHIVE) | C3-300k = best pixel survivor (mean 3243) |
| `runs/swa_vit_rtc` (15 ckpts) | 318 M | DELETE except `swa_vit_rtc_score_s1.pt` (22 M ARCHIVE) | last head you picked |
| `runs/swa_bc`, `runs/swa_vit_bc` | 52 M | DELETE | superseded BC inits |
| `data/` (teacher_bc*, rtc_* tapes) | 572 M | DELETE | 16-frame pixel tapes |
| `cleanrl_utils`, `scripts`, `tests`, `docs`, `.hermes`, `.git`, README, requirements | ~1 M | ARCHIVE (tarball); copy `maze_watch, pellet_bfs, ram_pellet_goals, go_explore, human_keys` → `legacy/` | source history; git has no commits |
| `results/`, `videos/` | 3 M | ARCHIVE | Rainbow curve + clips |
| `archive/` | 0 | DELETE | empty |

**Kept/archived ≈ 0.15 G. Reclaimable ≈ 10.8 G** (7.0 G is the venv).

Script: `cleanup.sh` (dry run by default, `--apply` to execute; archives + verifies before any `rm`).

Target layout:
```
~/HermesWorkspace/
  _archive/mspacman-rainbow-2026-10-08/   source_docs_logs.tar.zst, small_demos_bitmaps.tar.zst, ckpts/
  mspacman-rainbow -> _archive/...        (symlink left behind)
  mspacman-clear/
    README.md  pyproject.toml  .gitignore (runs/ seeds/*.pkl *.npz *.pt)
    msclear/ram.py        # verified addresses, pellet map, clear detector
    msclear/env.py        # ALE v5, frameskip=4, sticky 0.25, RAM obs + info
    msclear/maze.py       # online tile graph (pos,heading) -> next
    msclear/hra.py        # GVF tables, heads, aggregator, exploration heads
    msclear/explore.py    # Go-Explore (fallback)
    msclear/robustify.py  # backward-algorithm PPO (fallback)
    msclear/eval.py       # 100-episode protocol, clear rate
    tests/  configs/  seeds/  legacy/  runs/
```

## 3. New approach

### What already failed (from Hermes notes + project logs) and the common cause
| tried | result | root cause |
|---|---|---|
| Rainbow-lite 2M (pixels) | ~2.2k plateau, 0/48 clears | ghost-farm optimum; 16-frame decisions |
| SWA / ViT / RTC-IM / BC / DAgger (pixels) | best C3-300k 3243 mean, leftover ~100 (≈ 8 drawn dots, see M0), 0/80 clears detected | imitation of teachers with **zero clear labels** |
| TD from clones (SWA-RL, frozen stem, C2 polish, pellet-TD) | 4 collapses (2158/1900/1777/917) | TD on BC-initialised net, no BC regulariser |
| Option A RAM BFS + C3 dodge (`bfs_lives.jsonl`) | lives eat 1–14 pellets | 16-frame steps overshoot junctions; pixel-grid BFS |
| Go-Explore, 16 workers × 8 h (`overnight_ge.json`) | best leftover 159; from C3 seeds 101 | cell = leftover + pixel xy only; held actions of 12×16 = 192 frames |
| LLM / vision / composer play | leftover ≥ 101 | — |

Common thread: everything ran at **16 frames per decision** and either learned a monolithic value from pixels or copied a policy that never cleared. HRA's paper shows object channels alone don't fix it (A3C pixels 2168 vs object channels 2423). The bottleneck is that a single value function can't represent "go to the remaining dots while avoiding 4 ghosts". On top of that, HRA's own ablation hit the same endgame wall: when few pellets are left, the ghost terms drown out the pellet terms. They fixed it by normalising the score heads.

### Primary: HRA-style agent on verified RAM objects (CPU only)
Hybrid Reward Architecture (van Seijen et al. 2017, arXiv:1706.04208): 25,304 fixed-start / 23,770 random-start on Ms. Pac-Man without memorisation (multiple boards per game), ≈150 M frames, mostly tabular GVFs. Nothing to clone, no deep TD to collapse.

**Env:** `gym.make("ALE/MsPacman-v5", frameskip=4, repeat_action_probability=0.25, obs_type="ram")`, actions {UP, RIGHT, LEFT, DOWN} (+NOOP), no reward clipping, life loss is not terminal (the ghost heads carry the penalty), `max_num_frames_per_episode=108000`.

**State decode (verified on ale-py 0.12, `ram_verify.py`):**
- Ms. Pac-Man: x = `ram[10]`, y = `ram[16]` (screen x ≈ ram−8.3, y ≈ ram+5.8; r = 0.998)
- Ghosts x = `ram[6..9]`, y = `ram[12..15]` (6 = Sue/orange, 7 = Inky, 8 = Pinky, 9 = Blinky; r = 1.000, same offsets)
- Lives `ram[123]`; pellet bitmap `ram[59:101]`: 242 bits set at start but **only 150 are dots** (142 in `ram[59:100]` plus 8 bottom-row dots in `ram[100]`, drawn as dashes); the other 92 never clear. One bit clears per dot (+10 reward). Power pellets: `ram[117]` bits 2–5. Items counter `ram[119]` +1 per dot or power pellet; the board ends at 154 = 150 + 4.
- Frightened (M0, verified against sprite colour on 42k ghost sightings): per ghost `ram[1..4]` bit 7 (same ghost order as x/y), **99.82 %** accuracy. `ram[116]` is the global frightened countdown (0 when none): 98.2 % as a flag, misses eaten ghosts.
- Clear detector (M0): dots (150-bit mask) ≤ 3, then refill to ≥ 100 **and** level byte `ram[101]` increased (20 → 42 → 64, +22 per board). Zeroing the bitmap via `setRAM` does not advance the level. The old `maze_watch` thresholds (≤ 8 bits, refill from ≤ 80) counted all 234 bits (floor 92) and could never fire, so every old "leftover ~101" was really ~9 + up to 8 bottom-row dots left.
- Pellet bit → maze position (M0, `msclear/data/maze1_dots.json`): 150/150 dots mapped by single-bit render diff, cross-checked online (player position at each bit flip: median 0.9 px, max 3.7 px), and every `ram[119]` increment is explained by a mapped dot or power pellet. Levels 1–2 share the red map.

**Agent:**
- Position $p$ = (x//4, y//4) (~400 reachable per map). State $s = (p, \text{heading})$.
- Maze graph learned from observed transitions.
- One GVF per position: $G_p(s,a)$ trained off-policy on every transition with $\alpha=1,\gamma=0.99$, pseudo-reward 1 on reaching $p$. On a deterministic graph this converges to $\gamma^{d(s,a\to p)}$, so it can also be computed by BFS on the learned graph as a cross-check. Table ≈ 400 × 950 × 5 floats, a few MB.
- Aggregator, normalised as in HRA §A:
  $$Q(s,a)=\frac{\sum_{i\in\text{pellets}} 10\,G_{p_i} + \sum_{j\in\text{power}} 50\,G_{p_j} + \sum_{k\in\text{blue}} 1000\,G_{p_k} + 200\,G_{\text{fruit}}}{\max_a(\cdot)} \;-\; w_g\sum_{g\in\text{ghosts}} G^{(\gamma_g)}_{p_g}(s,a) \;+\; Q_{\text{div}} + \kappa\sqrt{\tfrac{N^{1/4}}{n(s,a)}}$$
  - Score heads normalised to [0,1], so the dot pull doesn't fade near the end.
  - $w_g \approx 10$ after normalisation; grid $\gamma_g\in\{0.95,0.97,0.99\}$.
  - $Q_{\text{div}}\sim U(0,20)$ for the first 50 steps.
  - Count bonus for exploration.
- No executive-memory head (that would memorise a fixed trajectory, not learn a policy).

**Compute:** ALE ≈ 10 k frames/s per core (measured on the box: 2.5 k agent steps/s per core at frameskip 4, no rendering). 12 workers ≈ 100 k frames/s raw; with Python overhead, budget ~20–30 k frames/s. The HRA paper's 150 M frames ≈ 1.5–2 h; board clears should come well before that. Tabular HRA stages are CPU-only; the GPU (dead fan costs some speed, not a thermal risk) is used for the neural stages (M5, fallback PPO). Realistic schedule: 1–2 days of engineering, then hours of compute.

**Milestones (all evals: 100 episodes, sticky 0.25, fixed seeds, report mean ± SE):**
- **M0 (decode, ≤1 day):**
  - Unit tests for the RAM decode.
  - All 150 dot bits mapped. Overlay test: every mapped dot lands on a visible dot. ✅
  - Frightened flag verified. ✅ (99.82 %)
  - Clear detector fires on a real clear (from M2 or the fallback GE) and never in 1,000 random episodes. Record the level byte when it changes. ✅ (`ram[101]`)
- **M1 (graph + GVFs, <1 h CPU):** ≥ 95 % of reachable positions visited; GVF table max TD change < 1e-3; BFS cross-check agrees. Implementation note: actions are UP/RIGHT/LEFT/DOWN only (NOOP keeps the last joystick direction, which is hidden state); state = exact RAM (x, y, heading); ghosts pinned in the pen during collection; GVFs solved by sweeps on the empirical model (= α = 1 TD).
- **M2 (pellets only, ghosts ignored, deterministic):** clears board 1 when ghosts are made harmless by RAM poke, or reaches ≤ 5 dots left. This checks the navigation part on its own. Note: plain sum-of-GVFs gets stuck (HRA's known failure); nearest-target (max over GVFs) plus a per-episode count penalty $\kappa\,n(s,a)$ works.
- **M3 (full HRA):** mean dots eaten on board 1 ≥ 135/150; **clear rate ≥ 20 %**.
- **M4 (tuning $w_g,\gamma_g$, normalisation, count bonus):** **clear rate ≥ 50 %** over 100 episodes; mean score > 5,000.
- **M5 (optional, if you want a pixel/neural agent):** dump HRA clearing episodes at frameskip 4. These are the first positive "clear" labels you've had. Train a CNN with BC + DQfD margin loss (your Option B), with eval clear rate ≥ 10 %.

### Fallback: RAM Go-Explore → backward-algorithm PPO
1. **Explore:** deterministic ALE (frameskip 4, sticky 0), `cloneState/restoreState`. Cell = (pellets left, mask of which pellet groups remain, player tile x//12,y//12, lives). Select by $e^{-(\ell-\ell_{\min})/5}\,1.5^{\text{lives}}/\sqrt{\text{visits}+1}$. Explore 60 steps of random-repeat actions (switch with p = 0.15). 12 workers. Stop at the first verified clear (drawn dots ≤ 3, then refill + level byte). Box measurement: once cells count drawn dots only, a real board-1 clear is found in **8 s on 1 core** (1,677 agent steps) and a board-2 clear 10 s later (1,946 steps); see `msclear/explore.py`, `msclear/data/clear_traj.json`. The old 8 h GE runs were probably reaching clears that went undetected.
2. **Robustify** (Salimans & Chen 2018; Go-Explore Nature):
   - PPO on RAM-object channels: 6 × 40 × 40 (player, normal ghosts, blue ghosts, dots, power, fruit) + heading one-hot, with a small CNN.
   - 16 envs, sticky 0.25. Start each episode from demo state $T-k$, and step $k$ back when ≥ 20 % of rollouts match the demo's return from that point.
   - Reward = game score / 100 + 10 on clear − 1 per life lost.
   - Hyperparameters: lr 2.5e-4, n_steps 128, 4 epochs, 4 minibatches, γ 0.99, λ 0.95, clip 0.1, ent 0.01.
   - Expect ~1–3 days on the PC (backward stages are the cost). Milestone: from-reset clear rate ≥ 20 % at sticky 0.25.

### Reuse
- `maze_watch.py` (clear detector, replaced by `msclear/clear.py`), `pellet_bfs.py` (maze BFS), `go_explore.py` archive/sampling code, `human_keys.py`/`record_human.py` (if you want to record a human clear at frameskip 4).
- GE `.pkl` ALE states as optional start states.
- Rainbow 2M only as a baseline number.
- Pixel checkpoints/tapes: not reusable (16-frame, no clears).
