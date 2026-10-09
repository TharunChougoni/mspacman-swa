"""M7 PPO environments: PixelGame workers (CPU, spawn) that return ONLY the newest 84x84 frame.

Each worker owns G games. The main process keeps the 4-stacks and the SWA token caches; a reset
stack is 4 copies of the first frame (PixelGame.reset), so frame + new-episode flag is enough.
Obs pipeline = M6 exactly (PixelGame: ALE v5 created with frameskip=1, 4-frame skip done once in
PixelGame, max-pool of frames 3/4, 84x84 INTER_AREA, 280 NOOP intro, sticky 0.25 per frame) ->
4 emulator frames per decision, no extra MaxAndSkip.

Autoreset is explicit and correct: the transition that ends an episode is returned with its
own reward and done=1; the frame returned with it is the FIRST frame of the next episode
(new=1). PPO masks the bootstrap at done=1, and the token window restarts at new=1.

Reward (per decision), weights from RW:
  items  = max(0, ram[119]_t+1 - ram[119]_t)   (dots + power pellets; ram[119] resets to 0 on a clear -> clipped)
  score  = ALE game-score delta (ghosts, fruit, dots)                    x score weight
  clear  = 1 on the step the board-1 clear is detected (level byte ram[101] up, dots refilled)
  death  = 1 if lives ram[123] dropped, or the game ended without a clear (the last life
           does not decrement ram[123])
Episode = until board clear, game over, or max_steps (truncation, rare).
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

SEED_BASE = 1_000_000          # training seeds; evals use 0..99 (periodic) and 50000.. (confirm)


def _worker(remote, gids, sticky, max_steps, rw, seed_off):
    from msclear.pixenv import PixelGame
    from msclear import ram as R
    games = [PixelGame(sticky=sticky, max_steps=max_steps) for _ in gids]
    epn = [0] * len(gids)
    seeds = [0] * len(gids)

    def new(i):
        seeds[i] = SEED_BASE + gids[i] * 100_000 + seed_off + epn[i]
        epn[i] += 1
        return games[i].reset(seeds[i])[-1]

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send(np.stack([new(i) for i in range(len(games))]))
            elif cmd == "step":
                n = len(games)
                frames = np.empty((n, 84, 84), np.uint8)
                rew = np.zeros(n, np.float32); done = np.zeros(n, np.uint8)
                comp = np.zeros((n, 4), np.float32)           # items, score, clear, death
                results = []
                for i, (g, a) in enumerate(zip(games, data)):
                    e0 = int(g.ram[R.DOTS_EATEN])
                    _, sc, d, info = g.step(int(a))
                    items = max(0, int(g.ram[R.DOTS_EATEN]) - e0)
                    over = d and g.ale.game_over() and not g.cleared
                    death = float(info["lost_life"] or over)
                    clear = float(d and g.cleared)
                    comp[i] = (items, sc, clear, death)
                    rew[i] = rw[0] * items + rw[1] * sc + rw[2] * clear - rw[3] * death
                    if d:
                        res = g.result(seeds[i]); res["truncated"] = bool(not g.cleared and not g.ale.game_over())
                        results.append(res)
                        frames[i] = new(i); done[i] = 1
                    else:
                        frames[i] = g.frame
                remote.send((frames, rew, done, comp, results))
            elif cmd == "close":
                break
    except (KeyboardInterrupt, EOFError):
        pass


class VecGames:
    def __init__(self, n_workers, per_worker, sticky=0.25, max_steps=6000, rw=(1.0, 0.0, 50.0, 25.0), seed_off=0):
        ctx = mp.get_context("spawn")
        self.n = n_workers * per_worker
        self.remotes, self.procs = [], []
        for w in range(n_workers):
            a, b = ctx.Pipe()
            gids = list(range(w * per_worker, (w + 1) * per_worker))
            p = ctx.Process(target=_worker, args=(b, gids, sticky, max_steps, tuple(rw), seed_off), daemon=True)
            p.start(); b.close()
            self.remotes.append(a); self.procs.append(p)

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return np.concatenate([r.recv() for r in self.remotes])

    def step(self, actions):
        k = len(actions) // len(self.remotes)
        for j, r in enumerate(self.remotes):
            r.send(("step", actions[j * k:(j + 1) * k]))
        out = [r.recv() for r in self.remotes]
        return (np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out]),
                np.concatenate([o[2] for o in out]), np.concatenate([o[3] for o in out]),
                [x for o in out for x in o[4]])

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
