# M9: SWA-v2 attention profile and options (analysis only)

Date: 10 Oct 2026. Net: `runs/m8/best.pt` (M8 confirm: 22.3% ± 2.4% clears, 300 games).
Scripts: `scripts/m9_profile.py` (timings → `runs/m9/profile.json`), `scripts/m9_attn.py` (attention stats, ablations, game evals → `runs/m9/attn.json`). Copies of both JSONs are in `docs/m9/`.
No default code path was changed and no training was run.

## 1. What the network actually is

`msclear/swa_net.py` `SWAv2(n_actions=4, d=512, n_heads=8, n_layers=2, window=8, lam=0.05)`, **7,996,581 params**.

| part | what it does | params |
|---|---|---|
| Conv stem (Nature CNN) | 4×84×84 uint8 stack (4 max-pooled grayscale frames, 4 emulator frames per decision) → conv 8×8/4 (32) → 4×4/2 (64) → 3×3/1 (64) → 3136 → `fc` 3136→512 + ReLU | 77,984 + 1,606,144 |
| **Tokens** | **1 token per decision step** (one per 4-stack). There are no spatial or object tokens: the CNN flattens the whole screen into one 512-d vector. | |
| Window | The newest **W = 8 tokens** (current step + 7 previous) → `pe_age` = learned embedding **indexed by slot 0..7 in the window (oldest first)**, so it is an absolute slot embedding, not a relative one. | 4,096 |
| 2 × SWABlock (pre-LN) | `RecencyCausalAttention`: 8 heads × 64, causal over the 8 tokens, scores − 0.05·age (ALiBi-like recency bias); then MLP 512→2048→512 (GELU) | 2 × (1,050,624 attn + 2,099,712 MLP + 2,048 LN) |
| Head | `ln_f` on the **newest token only** → dueling `val` (1) + `adv` (4). The PPO policy is softmax(adv/0.25); `val` is unused by PPO. | 1,024 + 513 + 2,052 |

What "SWA" attends over: **time, not space.** At each step the newest token attends over the last 8 step-embeddings, so the receptive field is 8 decisions ≈ 32 frames ≈ 0.53 s. Acting (`Student`, rollouts) caches the 8 tokens, so the CNN runs once per step. But **both transformer layers are recomputed for all 8 tokens every step**, because `pe_age` re-labels every token's slot when the window shifts. That makes a standard KV cache invalid for these weights.

## 2. Profile

### MACs per decision (one env, one forward)

| component | MACs | share (current) |
|---|---|---|
| conv stem (3 convs) | 7.74 M | 13% |
| fc 3136→512 | 1.61 M | 3% |
| transformer, projections + MLP (2 layers × 8 tokens × 3.15 M) | 50.33 M | 84% |
| **attention core** (QKᵀ + AV, 2 layers, 8×8) | **0.13 M** | **0.2%** |
| total | 59.8 M | |
| exact variant: last layer computes only the newest query | 41.4 M | −31% |
| retrained, relative-position + KV cache (1 new token per layer per step) | 15.7 M | −74% |

PPO rollouts run this twice per step (policy + frozen KL reference).

### Wall time (RTX 3060 Laptop, i7-12700H)

| measurement | time |
|---|---|
| GPU stem, B=42 (fp32/TF32 → bf16) | 0.42 → 0.31 ms |
| GPU trunk, B=42: full / last-token exact / bf16 / torch.compile (CUDA graphs) | 1.43 / 0.83 / 0.68 / 0.97 ms |
| GPU **one PPO rollout step**, N=42 (policy + ref stems and trunks, value, sample, `.cpu()`) | **2.26 ms** |
| GPU **one PPO training minibatch** (14 envs × 128 steps, fwd+bwd+Adam) | **96.7 ms** (stem-only part 24.4 ms → the transformer is ~75%) |
| same with exact last-token trunk | 79.6 ms (−18%) |
| CPU 1 thread, `Student.act` (eval/video path, B=1) | **3.54 ms** (stem 0.43, trunk 2.82; per block attn 0.44, MLP 0.75) |
| CPU 1 thread, last-token trunk | 2.00 ms |
| env `PixelGame.step`, 1 core | **0.67 ms** |
| vector env, 14 workers × 3 games, one step for all 42 | 4.6 ms alone; 7.3 ms while the 6 eval workers run |

### Where M8's PPO time actually went

From `runs/m8/metrics.jsonl`, 8,498 updates: **3.35 s per update** = rollout 2.25 s (128 steps, so 17.6 ms per vector step) + training 1.10 s. Median throughput was 1,560 steps/s.

| part of an update | estimate | share |
|---|---|---|
| env stepping (128 × 4.6–7.3 ms) | 0.6–0.9 s | 20–28% |
| rollout net forwards (128 × 2.26 ms) | 0.29 s | 9% |
| rollout glue: per-step H2D copies, per-env start-stack loop, GPU↔CPU sync, Python; env and GPU never overlap | ~1.0 s (inferred) | ~30% |
| PPO training fwd/bwd (9 minibatches × ~97 ms + GAE) | 1.10 s | 33% |

In addition, the periodic eval was **CPU-network-bound**: 100 games × ~2,900 steps × 3.54 ms on 6 single-thread workers took ~614 s, and with an eval every 150 updates (~500 s) **an eval was always running.** Those 6 busy cores slowed env stepping from 4.6 to 7.3 ms.

**Is the network the bottleneck?** In evals, yes: the net is 84% of per-game CPU time (3.54 ms vs 0.67 ms env). In PPO training, partly: the net is ~42% of wall time (training 33% + rollout forwards 9%), env stepping only ~25%, and serial Python/sync glue the rest. "PPO is CPU/env-bound" was true for M7 (2.5k steps/s with a 1,450-step game). It is not the main limit for M8.

## 3. Attention statistics (27,573 windows from 24 greedy M8 games, seeds 1000+)

Query = the newest token (the only output the policy reads). Entropy maximum is ln 8 = 2.08 nats.

| | layer 0 | layer 1 |
|---|---|---|
| mean entropy | 1.12 nats (≈ 3.1 effective tokens) | 1.20 nats (≈ 3.3) |
| per-head entropy | 0.91–1.32 | 1.04–1.33 |
| top-1 / top-2 / top-3 mass | 0.58 / 0.78 / 0.88 | 0.55 / 0.76 / 0.87 |
| weights < 0.02 | 43% | 39% |
| mass by age 0 (newest) … 7 | .06 .13 .14 .11 .13 .11 .10 **.22** | .16 .16 .14 .13 .11 .08 .09 .13 |
| endgame (≤ 30 items left) | same profile | same profile |

Each query is fairly peaked (3 tokens carry ~88% of the mass), but *which* tokens varies from state to state. On average every age gets attention, and layer 0 uses the oldest slot as a sink or summary (22%).

### Ablations, no retraining

Agreement is greedy-action agreement with the unablated net on the 27.6k windows; KL is at τ = 0.25. Games are 100 greedy games on seeds 0–99, normal starts, sticky 0.25; the baseline there is **0.31 ± 0.05** (selection-biased; the held-out confirm is 22.3%).

| ablation | agree | KL | games: clear / items / deaths |
|---|---|---|---|
| baseline | 1 | 0 | **0.31 ± 0.05** / 150.9 / 1.80 |
| mask any single head | 0.89–0.94 (worst L1H2) | 0.025–0.073 | — |
| top-k = 4 / 3 / 2 / 1 per query | 0.976 / 0.960 / 0.932 / 0.862 | .004 / .011 / .030 / .121 | top-2: 0.23 ± 0.04 / 151.0 / 1.87 |
| pool tokens older than 4 into their mean (temporal summary) | 0.905 | 0.057 | 0.19 ± 0.04 / 150.5 / 1.81 |
| pool older than 2 | 0.836 | 0.141 | — |
| window 6 / 4 / 2 / 1 (truncate history) | .84 / .72 / .61 / .52 | .16 / .41 / .77 / .99 | w4: 0.02 / 142.4; w2: 0.00 / 114.7; w1: 0.00 / 96.6 |
| layer-0 attention → self only | 0.546 | 1.00 | 0.00 / 63.4 / 2.00 |
| layer-1 attention → self only | 0.671 | 0.44 | — |
| drop layer 0 / layer 1 | 0.56 / 0.51 | 0.81 / 1.03 | — |
| SVD rank 256 / 128 / 64 in every block matrix | .92 / .84 / .71 | .03 / .13 / .38 | rank 128: 0.18 ± 0.04 / 149.7 / 1.94 |

What the ablations show:
1. **Temporal context is essential.** Cutting the history to 4 steps takes clears from 31% to 2%; one step plays like a much weaker agent. Ghost dodging lives in the 8-step history, so shrinking the window is not an option.
2. **No dead heads or layers.** Each head costs 6–11% agreement when masked, and removing either layer breaks the policy. There is no free pruning.
3. Attention is **moderately sparse per query** (top-3 ≈ 88%), but at 8 tokens the softmax part is 0.2% of the compute, so sparsifying it saves nothing. Top-2 even costs about 8 points of clears.
4. The width is used: rank 128 (≈ a 4× slimmer projection) loses ~13 points of clears before any fine-tuning. A slim student needs distillation, not truncation.

## 4. Waste found (largest first)

1. **Recomputing 8 tokens × 2 layers every step.** 87.5% of transformer work per step is recomputation of tokens already processed. The cause is the slot-indexed `pe_age`; a relative position scheme (the recency bias is already relative) would allow a true KV cache. This needs retraining or distillation.
2. **The last layer computes 7 token outputs nobody reads.** This is an exact fix with no retraining: trunk MACs −36%, measured GPU trunk −42% at B=42, CPU trunk −29%, training minibatch −18%. Verified max |Δ| 1e-6.
3. **Evals on CPU, one thread per game, fp32, B=1.** That is 3.54 ms of net per step, against 0.67 ms of env. Running the 100 eval games batched on the GPU (one vector env) would make evals env-bound, take roughly 1–2 minutes instead of ~10, and free 6 cores for the PPO envs.
4. **Serial rollout.** The GPU waits for the envs and the envs wait for the GPU, plus ~8 ms per step of glue. Two half-batches in ping-pong, or async stepping, would hide most of the env time.
5. **The frozen KL reference runs a full forward every rollout step.** That is half the rollout net time. Its logits could be computed batched at training time, or cached the same way.
6. **fp32/TF32 everywhere.** bf16 autocast gives trunk −53% and stem −26% at B=42. Not yet checked for action agreement.
7. torch.compile (CUDA graphs) helps less than the last-token change (0.97 vs 0.83 ms), and Triton compile warm-up is slow on this laptop. It's low priority.

The attention softmax itself (QKᵀ/AV) is **not** waste worth chasing: 0.13 M of 59.8 M MACs.

## 5. Hybrid and sparse attention candidates

| candidate | fits this net? | FLOP / latency saving at the real token count (8 temporal tokens) | reuse M8 weights? | verdict |
|---|---|---|---|---|
| (a) Local or windowed spatial attention + global tokens (Longformer/BigBird) | **No spatial tokens exist.** The CNN already does local spatial processing. Adding a 7×7 patch grid × 8 steps = 392 tokens would cost ≈ 392 × 3.15 M + 392² × 1 k ≈ 1.4 G MACs per layer, about **25× more** than now. Windowing only cuts the quadratic part, which is not where the cost is. | negative | no | **Reject** |
| (b) Top-k or object-centric sparse attention | Top-k over the 8 temporal tokens: measured top-2 clears 0.23 vs 0.31; top-3/4 nearly lossless (agree 96–98%). Saving ≤ 0.2% of MACs, and in practice the top-k sort makes it **slower**. Object-centric tokens (player, ghosts, pellets) would need RAM or a detector (no longer pixel-only) or a slot-attention encoder, which is a new architecture. | ~0% | top-k yes (with a loss); object tokens no | **Reject** (no compute to save) |
| (c) Hybrid layers: cheap early mixing + one softmax layer | Already the case: conv stem plus 2 attention layers. Replacing layer-0 attention by self-only gives 0.00 clears, so it carries real temporal information. A causal depthwise temporal conv (kernel 8) instead of layer-0 attention removes the qkv/out projections (1.05 M per token, −17% trunk MACs) and keeps temporal mixing. | −17% trunk, ≈ −14% per step | **needs distillation** | Only worth it inside option C; marginal once a KV cache exists |
| (d) Temporal sparsity: dense recent, pooled or cached old | Mean-pooling the 4 oldest tokens gives 0.19 vs 0.31 without retraining (agree 90.5%). As compute, 8 → 5 tokens = −37% trunk with no cache. **With a KV cache, old tokens already cost zero recompute**, so the cache is the right form of (d). | KV cache: trunk 50.3 M → 6.3 M per step (−87%), total −74% | **needs distillation** (relative positions) | **Adopt as the KV-cache form (option B)**; pooling only if the window must grow beyond 8 |

## 6. Options

Speedups are per update of the M8 PPO setup (3.35 s; network ≈ 1.4 s of it) and per eval (~614 s).

**A. Inference/training-path fixes, no retrain (same weights, exact or near-exact)**
- Changes:
  - last-token-only final layer (rollout, training and `Student`), exact;
  - bf16 autocast for the rollout forward;
  - batched GPU evals as one vector env, which frees 6 cores;
  - reference logits computed batched at training time;
  - optionally, ping-pong env halves.
- Expected: training minibatch −18%; rollout net ~2.3 → ~1.2 ms per step; env step 7.3 → 4.6 ms (no CPU evals competing); evals ~10 → ~1–2 min.
- PPO update ≈ 3.35 → ~2.5 s (**~1.3×**), up to ~1.6× with env/GPU overlap. CPU `Student` (confirm and video) 3.5 → 2.4 ms per step.
- Risk to 22%: none for the exact parts. bf16 needs a 100-game check (expected within noise).
- Time: ~2–3 h of coding and checks, ~15 min of PC compute.

**B. Same size, relative positions + KV cache, distilled from M8 (the "real" sliding-window attention)**
- Changes:
  - drop `pe_age` and keep the ALiBi recency bias (or use RoPE), so a token's K/V never change;
  - run the PPO training segment as one causal sequence with a band of 8 (each token computed once);
  - initialise every other weight from M8 and distil on M8's own rollouts plus student-driven DAgger states (KL at τ = 0.25, M8 as teacher), then a short PPO polish.
- Expected: per step 59.8 → 15.7 M MACs (3.8×). Training minibatch ≈ 97 → ~35 ms. Combined with A, update ≈ ~2.0–2.2 s (**~1.6×** vs M8). CPU eval per step ≈ 1 ms (3.5×).
- Risk: medium. The net currently relies on the slot embedding; aim for ≥ 97% agreement and ≥ 20% clears before using it.
- Time: ~0.5 day of code; distillation ~2–3 h on the PC; PPO polish 2–4 h.

**C. Slim student (d = 256, 2 layers, MLP ×2, causal temporal conv in layer 0, KV cache, W = 8), distilled from M8**
- Expected: trunk ≈ 0.9 M MACs per step, total ≈ 10 M (6× vs now). The stem (9.3 M) then dominates, so it's only ~1.5× faster than B in practice. Update ≈ ~1.9 s.
- Risk: higher. Rank 128 truncation already lost 13 points; expect to need DAgger plus PPO to get back to ~20%.
- Time: distillation 3–5 h, PPO 4–8 h.

**Recommendation: do A now. Then B only if more RL is planned.**
- A is exact, quick, and removes the biggest real waste: CPU evals hogging cores, and the unused last-layer outputs.
- B is the principled "sliding-window attention with a KV cache" and gives the larger compute cut for long PPO runs.
- C, and the spatial or sparse variants (a)/(b), don't pay off at 8 temporal tokens. The attention softmax is 0.2% of compute, and the measured ablations show that both the 8-step history and the full width are being used.

**Would training throughput improve?** Yes, but less than the FLOP numbers suggest. Env stepping is only ~25% of an M8 update and the net ~42%; the rest is serial glue. A alone gives ~1.3× (up to ~1.6× with env/GPU overlap). A + B gives ~1.6–1.8×. After that the 14 env workers (~9k steps/s ceiling at 4.6 ms per 42 steps) and the glue are the limit, and more net slimming buys little.
